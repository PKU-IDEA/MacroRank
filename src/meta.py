import pandas as pd
import torch
from torch import autograd
from torch._C import Graph
from torch.backends import quantized
import torch.nn.functional as F
import argparse
import os
import time
import numpy as np
from torch.nn.modules import loss
from torch.serialization import load
from tqdm import tqdm
from torch.utils.data import random_split, Subset, ConcatDataset
import torch.nn as nn
import pdb
import sys
import swats
from collections import OrderedDict
from torch_scatter import scatter
import torch_optimizer as optim
import copy
from . import datasets
from . import models 
from .logger import Logger
from .loader import DataLoader
from .util import InversePairs

def get_attr(obj, names):
    if type(names) == str:
        names = names.split('.')
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()
    def __iter__(self):
        return self._modules.values().__iter__()


def _make_functional(module, params_box, params_offset):
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    forward = type(module).forward
    if isinstance(module, nn.Conv2d):
        setattr(self, "_conv_forward", module._conv_forward)
    
    # if isinstance(module, nn.BatchNorm2d):
    #     setattr(self, "_check_input_dim", module._check_input_dim)
    #     setattr(self, "num_batches_tracked", module.num_batches_tracked)
    #     setattr(self, "running_mean", module.running_mean)
    #     setattr(self, "running_var", module.running_var)

    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue
        setattr(self, name, attr)

    child_params_offset = params_offset + num_params
    for name, child in module.named_children():
        child_params_offset, fchild = _make_functional(child, params_box, child_params_offset)
        self._modules[name] = fchild
        setattr(self, name, fchild)

    def fmodule(*args, **kwargs):
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):
            setattr(self, name, param)
        return forward(self, *args, **kwargs)

    return child_params_offset, fmodule


def make_functional(module):
    params_box = [None]
    _, fmodule_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        params_box[0] = kwargs.pop('params')
        return fmodule_internal(*args, **kwargs)

    return fmodule


class META(object):
    def __init__(self, args) -> None:
        super(META).__init__()
        self.args = args
        self.dataset = getattr(datasets,args.dataset)('data')
        self.criterion = self.build_loss(args)
        self.accuracy = self.build_acc(args)
        args.num_features = self.dataset.num_features
        self.args.num_features = self.dataset.num_features
    
        args.num_classes = 1
        self.args.num_classes = 1
        self.meta_model, self.model, self.optimizer, self.schedule = self.build_model(args)
        self.best_model_path, self.last_model_path = self.build_log(args)
        self.net = make_functional(self.meta_model)
        self.train_tasks = [i for i in self.dataset.train_file_names]
        self.test_tasks = [i for i in self.dataset.test_file_names]
        self.sprt_ratio = 0.1
        self.qry_ratio = 0.5
        self.update_lr = 0.0002
        self.update_step = 6
        self.method = 'maml'
        self.param_names = []
        self.build_param_names()
        #print(self.param_names)

        

    def build_param_names(self):
        for (name, w) in list(self.model.named_parameters()):
            if not w.requires_grad:
                continue
            self.param_names.append(name)


    def make_model_functional(self):
        self.build_param_names()
        for (name, w) in list(self.model.named_parameters()):
            #pdb.set_trace()
            if not w.requires_grad:
                continue
            hidden = torch.zeros_like(w.data)
            hidden = w.data
            hidden.requires_grad = True
            del_attr(self.model, name.split("."))
            set_attr(self.model, name.split("."), hidden)
            #print(get_attr(self.model, name.split(".")))


    def build_loss(self,args):
        def MAELoss(out,data):
            y = getattr(data,args.label)
            #y=data.y.new_zeros(1).float()
            return F.l1_loss(out.view(-1),y.view(-1))/out.shape[0]
        def MSELoss(out,data):
            y = getattr(data,args.label)
            return F.mse_loss(out.view(-1),y.view(-1))
        def CMSELoss(out,data):
            y = getattr(data,args.label)
            return F.l1_loss(out,y,reduction='sum')
        def CrossEntropyLoss(out,data):
            y = getattr(data,args.label)
            return F.cross_entropy(out,y.view(-1,))
        def CombLoss(out,data):
            prd = out[0]
            pred_edge = out[1]
            with torch.no_grad():
                cell_pos, edge_index, offset = data.cell_pos, data.edge_index, data.pin_feature

                B = scatter(cell_pos.new_ones(edge_index.size(1)),
                            edge_index[1], dim=0, dim_size=edge_index[1].max()+1, reduce='sum')
                B = 1.0 / B
                B[B == float("inf")] = 0

                pin_pos = torch.index_select(cell_pos,dim=0, index=edge_index[0]) + offset
                maxx = scatter(pin_pos,edge_index[1],dim=0,reduce='max')
                minn = scatter(pin_pos,edge_index[1],dim=0,reduce='min')
                hpwl = (maxx - minn).sum(-1).view(-1)*B.view(-1)
            return MAELoss(prd,data) #+ F.l1_loss(pred_edge.view(-1),hpwl) * 50
        if args.loss == 'MSE':
            return MSELoss
        elif args.loss == 'CMSE':
            return CMSELoss
        elif args.loss == 'CROSS':
            return CrossEntropyLoss
        elif args.loss == 'MAE':
            return MAELoss
        elif args.loss == 'CLOSS':
            return CombLoss
        else:
            print('Invalid loss function!')


    def build_acc(self,args):
        def RelAcc(out,data):
            y = getattr(data,args.label)
            return torch.mean(1-torch.abs((y.view(-1)-out[0].view(-1))/(y.view(-1))))
        def CRelAcc(out,data):
            y = getattr(data,args.label)
            return torch.mean(1-torch.abs((y.view(-1)-out[0].view(-1))/(y.view(-1)+0.00001)))
        def EqAcc(out,data):
            y = getattr(data,args.label)
            return torch.eq(torch.argmax(out[0],dim=1).view(-1),y.view(-1)).float().mean()
        if args.acc == 'rel':
            return RelAcc
        elif args.acc == 'Crel':
            return CRelAcc
        elif args.acc == 'eq':
            return EqAcc
        else:
            print('Invalid acc function!')


    def build_loader(self, args, design, train_ratio=0.8 , test_ratio = 0.2):
        if design == 'all':
            print(self.dataset.train_file_names)
            print(self.dataset.test_file_names)
            train_designs =  self.dataset.train_file_names
            test_designs = self.dataset.test_file_names
            train_sets = []
            test_sets = []
            num_training = 0
            num_testing = 0
            for design in train_designs:
                train_sets.append(Subset(self.dataset,range(self.dataset.ptr[design],
                                                        self.dataset.ptr[design] + self.dataset.file_num[design])))
                num_training += self.dataset.file_num[design]
            for design in self.test_designs:
                test_sets.append(Subset(self.dataset,range(self.dataset.ptr[design],
                                                        self.dataset.ptr[design] + self.dataset.file_num[design])))
                num_testing += self.dataset.file_num[design]
                
            train_set = ConcatDataset(train_sets)
            test_set = ConcatDataset(test_sets)
        else:
            num_training = int(self.dataset.file_num[design] * train_ratio)
            num_testing = int(self.dataset.file_num[design] * test_ratio)
            num_tot  = self.dataset.file_num[design]
            design_set = Subset(self.dataset,range(self.dataset.ptr[design],
                                            self.dataset.ptr[design] + self.dataset.file_num[design]))
            train_set, test_set, _ = random_split(design_set,[num_training, num_testing, num_tot - num_training - num_testing])
        #print("Total %d training data, %d testing data."%(num_training,num_testing),flush=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last=False)
        test_loader = DataLoader(test_set,batch_size=args.test_batch_size,shuffle=True,drop_last=False)
        return train_loader, test_loader


    def build_model(self, args):
        Model = getattr(models,args.model)
        meta_model = Model(args).to(args.device)
        model = Model(args).to(args.device)
        print(model)
        if args.optimizer == 'RAdam':
            optimizer = optim.RAdam(meta_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SWATS':
            optimizer = swats.SWATS(meta_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
        elif args.optimizer == 'SGD+Nestrov':
            optimizer = torch.optim.SGD(meta_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
        else:
            optimizer = getattr(torch.optim,args.optimizer)(meta_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
        return meta_model, model, optimizer, schedule


    def build_log(self, args):
        # make save dir
        st = time.strftime("%b:%d:%X",time.localtime())
        args.save_dir = os.path.join(args.save_dir,'{}_{}_{}_{}'.format(args.model,args.label,args.design,st))
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # redirect to save dir
        sys.stdout = Logger(path=args.save_dir)
        # print args
        print(args)
        # save paths
        best_model_path = os.path.join(args.save_dir,'best.pth'.format(st))
        last_model_path = os.path.join(args.save_dir,'last.pth'.format(st))
        return best_model_path, last_model_path

    
    def test(self, model, loader):
        with torch.no_grad():
            model.eval()
            correct = 0.
            loss = 0.
            reals = []
            preds = []
            for data in loader:
                data = data.to(self.args.device)
                out = model(data)
                preds.append(float(out))
                reals.append(float(getattr(data,self.args.label)))
                correct += self.accuracy(out,data).item()
                loss += self.criterion(out,data).item()
            Rp = np.argsort(preds)
            Rr = np.argsort(np.array(reals)[Rp])
            rankacc = InversePairs(Rr.tolist()) / (len(reals)**2 - len(reals)) * 2
        return correct / len(loader),loss / len(loader), rankacc

    def test_p(self, model, loader, params):
        with torch.no_grad():
            correct = 0.
            loss = 0.
            reals = []
            preds = []
            for data in loader:
                data = data.to(self.args.device)
                out = model(data, params=params)
                preds.extend([float(p) for p in out])
                reals.extend([float(p) for p in getattr(data,self.args.label)])
                correct += self.accuracy(out,data).item()
                loss += self.criterion(out,data).item()
            Rp = np.argsort(preds)
            Rr = np.argsort(np.array(reals)[Rp])
            rankacc = InversePairs(Rr.tolist()) / (len(reals)**2 - len(reals)) * 2
        return correct / len(loader),loss / len(loader), rankacc

    def test_without_p(self, model, loader):
        with torch.no_grad():
            correct = 0.
            loss = 0.
            reals = []
            preds = []
            for data in loader:
                data = data.to(self.args.device)
                out = model(data)
                preds.extend([float(p) for p in out])
                reals.extend([float(p) for p in getattr(data,self.args.label)])
                correct += self.accuracy(out,data).item()
                loss += self.criterion(out,data).item()
            Rp = np.argsort(preds)
            Rr = np.argsort(np.array(reals)[Rp])
            rankacc = InversePairs(Rr.tolist()) / (len(reals)**2 - len(reals)) * 2
        return correct / len(loader),loss / len(loader), rankacc


    def test_design(self, model, design, test_ratio):
        if type(design) is list:
            design = design[0]
        _, loader = self.build_loader(self.args, design, self.sprt_ratio, self.qry_ratio)
        return self.test(model,loader)


    def load(self, args):
        checkp = torch.load(args.checkp)
        self.model.load_state_dict(checkp['model'])
        start = checkp['epoch'] + 1
        print('load model from {}, saved at epoch {}'.format(args.checkp,start - 1))

    def finetune_rep(self):
        print('Finetune test begin : ....')

        def update_once(loader):
            """ update temp model once on sprt set, return average loss"""
            tot_loss = 0
            for i, data in enumerate(loader):
                data = data.to(self.args.device)
                logits = self.model(data)
                loss = self.criterion(logits, data)
                loss.backward()
                tot_loss += float(loss)
                for j, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        param.data = param.data - param.grad * self.update_lr
                        param.grad.zero_()
            return tot_loss / len(loader)

        def save_params(model):
            params = []
            for name in self.param_names:       
                params.append(get_attr(model,name).data.clone())
            return params

        pm = save_params(self.meta_model)
        sprt_loader, qry_loader = self.build_loader(self.args, self.dataset.test_file_names[0], self.sprt_ratio,  1 - self.sprt_ratio)

        for epoch in range(0, self.update_step * 2):
            self.model.train()
            tt = time.time()
            Ave_loss = 0.
            Ave_loss = update_once(sprt_loader)
            val_acc,val_loss, rank_err = self.test(self.model,qry_loader)
            print("[Epoch\t{}]\tTrain loss:\t{:.4f}\tVal loss:\t{:.4f}\tVal accuracy:\t{:.4f}\tRank err:\t{:.4f}\tTotal time:{:.2f}\tlr:{:.5f}".format(
        epoch,Ave_loss,val_loss,val_acc,rank_err,time.time()-tt,self.update_lr),flush=True)

            # state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch,'patience':patience,'min_loss':min_loss}
            # if epoch % 5 == 0:
            #     torch.save(state, self.last_model_path)

            # if val_loss < min_loss:
            #     state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch,'patience':patience,'min_loss':min_loss}
            #     torch.save(state, self.best_model_path + '.finetune')
            #     print("Model saved at epoch{}".format(epoch),flush=True)
            #     min_loss = val_loss
            #     patience = 0
            # elif rank_err < min_err:
            #     state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch,'patience':patience,'min_loss':min_loss}
            #     torch.save(state, self.best_model_path + '.best_rank.finetune')
            #     print("Model saved at epoch{}".format(epoch),flush=True)
            #     min_err = rank_err
            #     patience = 0
            # else:
            #     patience += 1
            # if patience > args.patience:
            #     break 

    def finetune(self):
        self.optimizer.zero_grad()
        print('Finetune test begin : ....')
        def save_params(model):
            params = []
            for name in self.param_names:       
                params.append(get_attr(model,name).data.clone())
            return params

        def load_params(model, params):
            for i, name in enumerate(self.param_names):
                param = get_attr(model,name)
                param.data = params[i]
            return params

        def update_once(loader):
            """ update temp model once on sprt set, return average loss"""
            tot_loss = 0
            for i, data in enumerate(loader):
                data = data.to(self.args.device)
                logits = self.model(data)
                loss = self.criterion(logits, data)
                #loss.backward()
                for j, name in enumerate(self.param_names):
                    param = get_attr(self.model, name)
                    if param.requires_grad:
                        grad0 = torch.autograd.grad(loss, param, retain_graph=True, create_graph=True)
                        new_param = param - grad0[0] * self.update_lr
                        param.data = new_param.data

                tot_loss += float(loss)
            return tot_loss / len(loader)

        def test_once_without_update(loader):
            """ test temp model on qry set """
            with torch.no_grad():
                tot_loss = 0
                for i, data in enumerate(loader):
                    data = data.to(self.args.device)
                    logits = self.model(data)
                    loss = self.criterion(logits, data)
                    tot_loss += float(loss)
            return tot_loss / len(loader)


        # save origin params
        pm = save_params(self.model)
        #pdb.set_trace()
        losses_q = [0 for _ in range(self.update_step + 1)]
        errs_q = [0 for _ in range(self.update_step + 1)]
        #pdb.set_trace()
        for task in self.test_tasks:
            # set up support set and query set
            sprt_loader, qry_loader = self.build_loader(self.args, task, self.sprt_ratio, self.qry_ratio)
            # load parms \phi^0
            load_params(self.model, pm)
            # step 0, do not update
            losses_q[0] += test_once_without_update(qry_loader)
            # step 1
            update_once(sprt_loader)
            losses_q[1] += test_once_without_update(qry_loader)
            print(losses_q)
            for k in range(1, self.update_step):
                update_once(sprt_loader)
                losses_q[k + 1] += test_once_without_update(qry_loader)
                print(losses_q)

            val_acc,val_loss, rank_err = self.test_without_p(self.model,qry_loader)
            print("[{}]\tMAE:\t{:.4f}\tMRE:\t{:.4f}\tIPE:\t{:.4f}".format(task, val_loss,val_acc,rank_err),flush=True)
        self.optimizer.zero_grad()


    def finetune_new(self):
        self.optimizer.zero_grad()
        print('Finetune test begin : ....')
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        for i, task in enumerate(self.test_tasks):
            sprt_loader, qry_loader = self.build_loader(self.args, task, self.sprt_ratio, self.qry_ratio)
            # 1. run the i-th task and compute loss for k=0
            grad = None
            for s in sprt_loader:
                s.to(self.args.device)
                logits = self.meta_model(s)
                loss = self.criterion(logits, s)
                if grad is None:
                    grad = torch.autograd.grad(loss, self.meta_model.parameters())
                else:
                    grad = list(map(lambda p: p[1] + p[0], zip(grad, torch.autograd.grad(loss, self.meta_model.parameters()))))

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.meta_model.parameters())))

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                grad = None
                for s in sprt_loader:
                    s.to(self.args.device)
                    logits = self.net(s, params=fast_weights)
                    loss = self.criterion(logits, s)
                    # 2. compute grad on theta_pi
                    
                    if grad is None:
                        grad = torch.autograd.grad(loss, fast_weights)
                    else:
                        grad = list(map(lambda p: p[1] + p[0], zip(grad, torch.autograd.grad(loss, fast_weights))))
                # 3. theta_pi = theta_pi - train_lr * grad
                
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        
            val_acc,val_loss, rank_err = self.test_p(self.net,qry_loader,fast_weights)
            print("[{}]\tMAE:\t{:.4f}\tMRE:\t{:.4f}\tIPE:\t{:.4f}".format(task, val_loss,val_acc,rank_err),flush=True)
        self.optimizer.zero_grad()

    def train(self, method = None):
        if method is not None:
            self.method = method
        args = self.args
        start = 0
        min_loss = 1e10
        min_err = 1e10
        patience = 0.
        sprt_ratio = 0.


        for epoch in range(start, args.epochs):
            self.meta_model.train()
            #self.model.train()
            tt = time.time()
            if self.method == 'maml':
                losses = self.maml()
            elif self.method == 'fomaml':
                losses = self.fomaml()
            elif self.method == 'reptile':
                losses = self.reptile()
            else:
                raise NotImplementedError
                
            #val_loss,_,_ = self.test_design(self.meta_model, self.dataset.test_file_names, self.qry_ratio)

            print("[Epoch\t{}]\tTrain loss:\t{}\tTotal time:{:.2f}\tlr:{:.5f}".format(
                epoch,[[float(l)/len(self.train_tasks),] for l in losses],time.time()-tt,self.optimizer.param_groups[0]['lr']),flush=True)
            state = {'model': self.meta_model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch,'patience':patience,'min_loss':min_loss}
            if epoch % 5 == 0:
                #if epoch > 50 and epoch % 10 == 0:
                self.finetune()
                torch.save(state, self.last_model_path)

            #elif rank_err < min_err:
            #    state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch,'patience':patience,'min_loss':min_loss}
            #    torch.save(state, self.best_model_path + '.best_rank')
            #    print("Model saved at epoch{}".format(epoch),flush=True)
            #   min_err = rank_err
            #    patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break 

        
    def fomaml(self):
        def save_params(model):
            params = []
            for param in model.parameters():       
                params.append(param.data.clone())
            return params

        def load_params(model, params):
            for i, param in enumerate(model.parameters()):       
                param.data = params[i]
            return params

        def update_once(loader):
            """ update temp model once on sprt set, return average loss"""
            tot_loss = 0
            for i, data in enumerate(loader):
                data = data.to(self.args.device)
                logits = self.model(data)
                loss = self.criterion(logits, data)
                loss.backward()
                tot_loss += float(loss)
                for j, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        param.data = param.data - param.grad * self.update_lr
                        param.grad.zero_()
            return tot_loss / len(loader)

        def test_once_without_update(loader):
            """ test temp model on qry set """
            with torch.no_grad():
                tot_loss = 0
                for i, data in enumerate(loader):
                    data = data.to(self.args.device)
                    logits = self.model(data)
                    loss = self.criterion(logits, data)
                    tot_loss += float(loss)
            return tot_loss / len(loader)

        def test_once(loader):
            """ upddate meta model with grad of temp model on qry set """
            tot_loss = 0
            for i, data in enumerate(loader):
                data = data.to(self.args.device)
                logits = self.model(data)
                loss = self.criterion(logits, data)
                self.optimizer.zero_grad() # reset meta param's grad
                loss.backward()
                tot_loss += float(loss)

            for param, meta_param in zip(self.model.parameters(), \
                                            self.meta_model.parameters()):
                if param.grad is not None:
                    meta_param.grad = param.grad
                    param.grad.zero_()
            self.optimizer.step()
                
            return tot_loss / len(loader)

        # set parms in temp model same as meta model
        for param, meta_param in zip(self.model.parameters(), \
                                        self.meta_model.parameters()):
            if meta_param is not None:
                param.data = meta_param.data

        # save origin params
        pm = save_params(self.model)

        losses_q = [0 for _ in range(self.update_step + 1)]
        errs_q = [0 for _ in range(self.update_step + 1)]
        #pdb.set_trace()
        for task in self.train_tasks:
            # set up support set and query set
            sprt_loader, qry_loader = self.build_loader(self.args, task, self.sprt_ratio, self.qry_ratio)
            # load parms \phi^0

            load_params(self.model, pm)
            # step 0, do not update
            losses_q[0] += test_once_without_update(qry_loader)
            # step 1
            update_once(sprt_loader)
            losses_q[1] += test_once(qry_loader)
            for k in range(1, self.update_step):
                update_once(sprt_loader)
                losses_q[k + 1] += test_once(qry_loader)

        if self.optimizer.param_groups[0]['lr'] > self.args.lr / 100:
            self.schedule.step()
        #accs = np.array(corrects) / (querysz * task_num)
        return losses_q

    def reptile(self):
        def save_params(model):
            params = []
            for param in model.parameters():       
                params.append(param.data.clone())
            return params

        def load_params(model, params):
            for i, param in enumerate(model.parameters()):       
                param.data = params[i]
            return params

        def update_once(loader):
            """ update temp model once on sprt set, return average loss"""
            tot_loss = 0
            for i, data in enumerate(loader):
                data = data.to(self.args.device)
                logits = self.model(data)
                loss = self.criterion(logits, data)
                tot_loss += float(loss)
                loss.backward()
                for j, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        param.data = param.data - param.grad * self.update_lr
                        param.grad.zero_()
            return tot_loss / len(loader)

        def test_once_without_update(loader):
            """ test temp model on qry set """
            with torch.no_grad():
                tot_loss = 0
                for i, data in enumerate(loader):
                    data = data.to(self.args.device)
                    logits = self.model(data)
                    loss = self.criterion(logits, data)
                    tot_loss += float(loss)
            return tot_loss / len(loader)

        def meta_grad_update():
            """ upddate meta model with approximation """
            
            for param, meta_param in zip(self.model.parameters(), \
                                            self.meta_model.parameters()):
                # since param requies grad the same as meta _param,
                # if param.grad is not None,
                # the meta_grad is required
                if param.grad is not None:
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(param.grad)
                    meta_param.grad += (meta_param - param) / len(self.train_tasks)
        # set parms in temp model same as meta model
        for param, meta_param in zip(self.model.parameters(), \
                                        self.meta_model.parameters()):
            if meta_param is not None:
                param.data = meta_param.data

        # save origin params
        pm = save_params(self.model)
        self.optimizer.zero_grad() # reset meta param's grad
        losses_q = [0 for _ in range(self.update_step + 1)]
        errs_q = [0 for _ in range(self.update_step + 1)]
        #pdb.set_trace()ds
        for task in self.train_tasks:
            # set up support set and query set
            sprt_loader, qry_loader = self.build_loader(self.args, task, self.sprt_ratio, self.qry_ratio)
            # load parms \phi^0
            load_params(self.model, pm)
            # step 0, do not update
            losses_q[0] += test_once_without_update(qry_loader)
            # step 1
            update_once(sprt_loader)
            losses_q[1] += test_once_without_update(qry_loader)
            for k in range(1, self.update_step):
                update_once(sprt_loader)
                losses_q[k + 1] += test_once_without_update(qry_loader)
            # \phi <- \phi + \epsilon (\theta_k - \theta) / K
            meta_grad_update()
        self.optimizer.step()
        
        if self.optimizer.param_groups[0]['lr'] > self.args.lr / 100:
            self.schedule.step()
        #accs = np.array(corrects) / (querysz * task_num)
        return losses_q


    def maml(self):
        def save_params(model):
            params = []
            for name in self.param_names:       
                params.append(get_attr(model,name).data.clone())
            return params

        def load_params(model, params):
            for i, name in enumerate(self.param_names):
                param = get_attr(model,name)
                param.data = params[i]
            return params

        def update_once(loader, last_grad):
            """ update temp model once on sprt set, return average loss"""
            tot_loss = 0
            for i, data in enumerate(loader):
                data = data.to(self.args.device)
                logits = self.model(data)
                loss = self.criterion(logits, data)
                #loss.backward()
                for j, name in enumerate(self.param_names):
                    param = get_attr(self.model, name)
                    if param.requires_grad:
                        grad0 = torch.autograd.grad(loss, param, retain_graph=True, create_graph=True,allow_unused=True )
                        #pdb.set_trace()
                        new_param = param - grad0[0] * self.update_lr
                        grad1 = torch.autograd.grad(new_param.sum(), param,create_graph=True,allow_unused=True )[0].detach()
                        with torch.no_grad():
                            last_grad[j] = last_grad[j] * grad1
                        if last_grad[j].sum().isnan():
                            pdb.set_trace()
                            print(grad1)
                        param.data = new_param.data

                tot_loss += float(loss)
            return tot_loss / len(loader)

        def test_once_without_update(loader):
            """ test temp model on qry set """
            with torch.no_grad():
                tot_loss = 0
                for i, data in enumerate(loader):
                    data = data.to(self.args.device)
                    logits = self.model(data)
                    loss = self.criterion(logits, data)
                    tot_loss += float(loss)
            return tot_loss / len(loader)

        def test_once(loader, last_grad):
            """ updatethe grad of meta model with grad of temp model on qry set """
            tot_loss = 0
            for i, data in enumerate(loader):
                data = data.to(self.args.device)
                logits = self.model(data)
                loss = self.criterion(logits, data)
                self.optimizer.zero_grad() # reset meta param's grad
                loss.backward()
                for j, name in enumerate(self.param_names):
                    param = get_attr(self.model, name)
                    meta_param = get_attr(self.meta_model,name)
                    if meta_param.requires_grad and param.requires_grad:
                        if meta_param.grad is None:
                            meta_param.grad = last_grad[j] * param.grad / len(loader)
                        else:
                            meta_param.grad += last_grad[j] * param.grad / len(loader)
                        #pdb.set_trace()
                        param.grad.zero_()
                
                tot_loss += float(loss)
            return tot_loss / len(loader)


        last_grads = []
        # set parms in temp model same as meta model
        for param, meta_param in zip(self.model.parameters(), \
                                        self.meta_model.parameters()):
            if meta_param is not None:
                param.data = meta_param.data
                last_grads.append(torch.ones_like(param))
            else:
                last_grads.append(None)

        self.optimizer.zero_grad()
        # save origin params
        pm = save_params(self.model)
        #pdb.set_trace()
        losses_q = [0 for _ in range(self.update_step + 1)]
        errs_q = [0 for _ in range(self.update_step + 1)]
        #pdb.set_trace()
        for task in self.train_tasks:
            # set up support set and query set
            sprt_loader, qry_loader = self.build_loader(self.args, task, self.sprt_ratio, self.qry_ratio)
            # load parms \phi^0
            load_params(self.model, pm)
            last_grads = []
            for param, meta_param in zip(self.model.parameters(), \
                                        self.meta_model.parameters()):
                if meta_param is not None:
                    last_grads.append(torch.ones(param.shape, dtype=torch.float, device=param.device))
                else:
                    last_grads.append(None)
            # step 0, do not update
            losses_q[0] += test_once_without_update(qry_loader)
            # step 1
            update_once(sprt_loader, last_grads)
            losses_q[1] += test_once_without_update(qry_loader)
            print(losses_q)
            for k in range(1, self.update_step - 1):
                update_once(sprt_loader, last_grads)
                losses_q[k + 1] += test_once_without_update(qry_loader)
                print(losses_q)
            update_once(sprt_loader, last_grads)
            losses_q[self.update_step] += test_once(qry_loader, last_grads)
            print(losses_q)
        #pdb.set_trace()
        self.optimizer.step()
        self.optimizer.zero_grad()
        #pdb.set_trace()
        if self.optimizer.param_groups[0]['lr'] > self.args.lr / 100:
            self.schedule.step()

        return losses_q
    
    def new_maml(self):
        self.optimizer.zero_grad()
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        tot_grad = 0
        for i, task in enumerate(self.train_tasks):
            sprt_loader, qry_loader = self.build_loader(self.args, task, self.sprt_ratio, self.qry_ratio)
            # 1. run the i-th task and compute loss for k=0
            grad = None
            for s in sprt_loader:
                s.to(self.args.device)
                logits = self.meta_model(s)
                loss = self.criterion(logits, s)
                if grad is None:
                    grad = torch.autograd.grad(loss, self.meta_model.parameters())
                else:
                    grad = list(map(lambda p: p[1] + p[0], zip(grad, torch.autograd.grad(loss, self.meta_model.parameters()))))

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.meta_model.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                for q in qry_loader:
                    q.to(self.args.device)
                    logits_q = self.meta_model(q)
                    loss_q = self.criterion(logits_q,q)
                    losses_q[0] += loss_q
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                for q in qry_loader:
                    q.to(self.args.device)
                    logits_q = self.net(q, params=fast_weights)
                    loss_q = self.criterion(logits_q,q)
                    losses_q[1] += loss_q
            # this is the loss and accuracy after

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                grad = None
                for s in sprt_loader:
                    s.to(self.args.device)
                    logits = self.net(s, params=fast_weights)
                    loss = self.criterion(logits, s)
                    # 2. compute grad on theta_pi
                    
                    if grad is None:
                        grad = torch.autograd.grad(loss, fast_weights)
                    else:
                        grad = list(map(lambda p: p[1] + p[0], zip(grad, torch.autograd.grad(loss, fast_weights))))
                # 3. theta_pi = theta_pi - train_lr * grad
                
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                if k < self.update_step - 1:
                    with torch.no_grad():
                        for q in qry_loader:
                            q.to(self.args.device)
                            logits_q = self.net(q, params=fast_weights)
                            # loss_q will be overwritten 
                            loss_q = self.criterion(logits_q, q)
                            losses_q[k + 1] += loss_q
                else:
                    for i, q in enumerate(qry_loader):
                        #print("{} batch".format(i))
                        q.to(self.args.device)
                        logits_q = self.net(q, params=fast_weights)
                        #  keep the loss_q on last update step.
                        loss_q = self.criterion(logits_q, q) / len(self.train_tasks)
                        with torch.no_grad():
                            losses_q[k + 1] += loss_q
                        loss_q.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses_q
