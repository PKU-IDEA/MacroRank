import pdb
import sys
import argparse
import os
import time
import random
import numpy as np
from random import random

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_optimizer import swats
import torch_optimizer as optim
from torch.utils.data import random_split, Subset, ConcatDataset
from torch.utils.tensorboard import writer


import src.hyperdataset as hdatasets
import src.hypermodel as hmodels
from src.logger import Logger
from src.util import InversePairs, mle_loss, spearman, dcg_score

#from src.meta import META
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--device', type=str, default='cuda:0',help='device')
parser.add_argument('--model', type=str, default='GClassifier',help='which mdoel to use')
parser.add_argument('--batch_size', type=int, default=8,help='train batch size')
parser.add_argument('--batch_step', type=int, default=1,help='how many batches per update')
parser.add_argument('--test_batch_size', type=int, default=8,help='test batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step_size', type=int, default=50, help='learning rate decay step')
parser.add_argument('--lr_decay', type=float, default=1., help='learning rate decay ratio')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=16, help='hidden size')
parser.add_argument('--layers',type=int,default=2,help='conv layers')
parser.add_argument('--egnn_layers',type=int,default=3,help='egnn layers')
parser.add_argument('--egnn_nhid',type=int,default=16,help='egnn layers hidden dim')
#parser.add_argument('--pooling_ratio', type=float, default=0.1,help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.1,help='dropout ratio')
parser.add_argument('--group', type=int, default=0, help='which data group to use')
parser.add_argument('--tests', type=str, nargs='+', 
    default=['mgc_des_perf_a', 'mgc_fft_a', 'mgc_matrix_mult_a', 'mgc_matrix_mult_c', 'mgc_superblue14', 'mgc_superblue19'],help='test data')
parser.add_argument('--trains', type=str, nargs='+', 
    default=['mgc_edit_dist_a', 'mgc_fft_b', 'mgc_matrix_mult_b', 'mgc_pci_bridge32_b', 'mgc_superblue11_a', 'mgc_superblue16_a'],help='train data')
parser.add_argument('--dataset_path', type=str, default='data')
parser.add_argument('--dataset', type=str, default='PlainClusterSet')
parser.add_argument('--epochs', type=int, default=400,help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=400,help='patience for earlystopping')
parser.add_argument('--save_dir', type=str, default='save')
parser.add_argument('--goon', action='store_true',help='continue training')
parser.add_argument('--con', action='store_true',help='continue training')
parser.add_argument('--checkp', type=str, default='test.pth')
parser.add_argument('--pos_encode', type=int, default=4, help='whether use pos encoding on position')
parser.add_argument('--size_encode', type=int, default=0, help='whether use pos encoding on size')
parser.add_argument('--offset_encode', type=int, default=0, help='whether use pos encoding on offset')
parser.add_argument('--design', type=str, default='all',help='whitch design to train')
parser.add_argument('--loss', type=str, default='MAE',help='loss func')
parser.add_argument('--acc', type=str, default='rel',help='loss func')
parser.add_argument('--skip_cnt',  action='store_true', default=True ,help='use skip cnt ?')
parser.add_argument('--regresion', action='store_true', help='regression')
parser.add_argument('--classifier', action='store_true', help='classification')
parser.add_argument('--base_model', type=str, default='EGNN',help='which base mdoel to use in classifier')
parser.add_argument('--metric', type=str, default='lambdda',help='which metric to use as lambda, [lambdda (top1 prob), ndcg]')
parser.add_argument('--label', type=list[int],default=[1],help='which label to use, [0~5] = [hpwl, rwl, via, short, score]')
parser.add_argument('--train_ratio', type=float, default=0.8,help='train ratio')
parser.add_argument('--optimizer',type=str,default='Adam')
args = parser.parse_args()
args.betas = [0.005, ]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def build_loss(args):
    def MAELoss(out,data):
        if len(args.label) > 1:
            label = torch.tensor(args.label[0]).long().to(data.device)
            y = data.y[:, label]
            w = data.w[:, label]
            return F.l1_loss(out[0].view(-1) * w,y.view(-1) * w)
        else:
            label = args.label[0]
            y = data.y[:, label]
            w = data.w[:, label]
            return F.l1_loss(out[0].view(-1) * w, y.view(-1) * w) 
    def MSELoss(out,data):
        if len(args.label) > 1:
            label = torch.tensor(args.label[0]).long().to(data.device)
            y = data.y[:, label]
            w = data.w[:, label]
            return F.mse_loss(out[0].view(-1) * w, y.view(-1) * w)
        else:
            label = args.label[0]
            y = data.y[:, label]
            w = data.w[:, label]
            return F.mse_loss(out[0].view(-1) * w,y.view(-1) * w)
    def BCELoss(out,data):
        if len(args.label) > 1:
            label = torch.tensor(args.label[0]).long().to(data.device)
            y = data.y[:, label]
            w = data.w[:, label]
            return F.mse_loss(out[0].view(-1) * w, y.view(-1) * w)
        else:
            label = args.label[0]
            y = data.y[:, label]
            w = data.w[:, label]
            return F.binary_cross_entropy(target=y.view(-1), input=out[0].view(-1), weight=w)
    def MLELoss(out,data):
        if len(args.label) > 1:
            label = torch.tensor(args.label[0]).long().to(data.device)
            y = data.y[:, label]
            w = data.w[:, label]
            return F.mse_loss(out[0].view(-1) * w, y.view(-1) * w)
        else:
            label = args.label[0]
            y = data.y[:, label]
            return mle_loss(y.view(-1), out[0].view(-1))
    def SMAELoss(out,data):
        if len(args.label) > 1:
            label = torch.tensor(args.label[0]).long().to(data.device)
            y = data.y[:, label]
            w = data.w[:, label]
            return F.mse_loss(out[0].view(-1) * w, y.view(-1) * w)
        else:
            label = args.label[0]
            y = data.y[:, label]
            w = data.w[:, label]
            torch.nn.HuberLoss
            return F.smooth_l1_loss(out[0].view(-1) * w, y.view(-1) * w, beta=0.005)
    def CMSELoss(out,data):
        y = getattr(data,args.label)
        return F.l1_loss(out[0],y,reduction='sum')
    def CrossEntropyLoss(out,data):
        label = args.label[0]
        y = data.y[:, label]
        index = torch.arange(0, y.shape[0], 2).to(y.device)
        y0 = torch.index_select(y, dim=0, index=index)
        y1 = torch.index_select(y, dim=0, index=(index+1))
        target = ((y0 - y1) >= 0).long()
        return F.cross_entropy(out[0], target.view(-1))
    def COMBLoss(out,data):
        # get label
        label = args.label[0]
        y = data.y[:, label]
        y1 = data.y1[:, label]
        y2 = data.y2[:, label]
        w = data.w[:, label]
        w1 = data.w1[:, label]
        w2 = data.w2[:, label]
        # get out
        index = torch.arange(0, out[0].shape[0], 2).to(y.device)
        out1 = torch.index_select(out[0], dim=0, index=index)
        out2 = torch.index_select(out[0], dim=0, index=(index+1))
        p = torch.sigmoid(out1 - out2)
        #
        bce_loss = F.binary_cross_entropy(input=p, target=y, weight=w)
        mae_loss = F.l1_loss(input=out1 * w1, target=y1 * w1) + F.l1_loss(input=out2 * w2, target=y2 * w2)
        return bce_loss + mae_loss
    if args.loss == 'MSE':
        return MSELoss
    elif args.loss == 'CMSE':
        return CMSELoss
    elif args.loss == 'BCE':
        return BCELoss
    elif args.loss == 'CROSS':
        return CrossEntropyLoss
    elif args.loss == 'MLE':
        return MLELoss
    elif args.loss == 'MAE':
        return MAELoss
    elif args.loss == 'SMAE':
        return SMAELoss
    elif args.loss == 'COMB':
        return COMBLoss
    else:
        print('Invalid loss function!')


def build_acc(args):
    def RelAcc(out, data):
        label = args.label[0]
        y = data.y[:, label]
        return torch.mean(1-torch.abs((y.view(-1)-out[0].view(-1))/(y.view(-1))))
    def CRelAcc(out, data):
        y = getattr(data,args.label)
        return torch.mean(1-torch.abs((y.view(-1)-out[0].view(-1))/(y.view(-1)+0.00001)))
    def SROCC(out, data):
        label = args.label[0]
        y = data.y[:, label]
        return spearman(y.view(-1), out[0].view(-1))
    def EqAcc(out, data):
        label = args.label[0]
        y = data.y[:, label]
        index = torch.arange(0, y.shape[0], 2).to(y.device)
        y0 = torch.index_select(y, dim=0, index=index)
        y1 = torch.index_select(y, dim=0, index=(index+1))
        target = ((y0 - y1) >= 0).long()
        return torch.eq(torch.argmax(out[0],dim=1).view(-1), target.view(-1)).float().mean()
    def BEQAcc(out, data):
        label = args.label[0]
        y = data.y[:, label]
        mask1, mask5, mask0 = (out[0] > 0.5), (out[0] == 0.5), (out[0] < 0.5)
        mask = 1. * mask1 + 0.5 * mask5
        return torch.eq(mask.view(-1), y.view(-1)).float().mean()
    def COMBAcc(out, data):
        label = args.label[0]
        y1 = data.y1[:, label]
        y2 = data.y2[:, label]
        y = torch.cat((y1,y2))
        index = torch.arange(0, out[0].shape[0], 2).to(y.device)
        out1 = torch.index_select(out[0], dim=0, index=index)
        out2 = torch.index_select(out[0], dim=0, index=(index+1))
        out = torch.cat((out1, out2))
        return torch.mean(1-torch.abs((y.view(-1)-out.view(-1))/(y.view(-1))))
    if args.acc == 'rel':
        return RelAcc
    elif args.acc == 'SROCC':
        return SROCC
    elif args.acc == 'Crel':
        return CRelAcc
    elif args.acc == 'eq':
        return EqAcc
    elif args.acc == 'BEQ':
        return BEQAcc
    elif args.acc == 'COMB':
        return COMBAcc
    else:
        print('Invalid acc function!')
        assert(False)


def build_loader(design,train_ratio=0.8):
    MySet = getattr(hdatasets,args.dataset)

    dataset = MySet(args.dataset_path, mode=args.model, test_files=args.tests, train_files=args.trains, args=args)
    
    if args.model != 'CNN' and args.model != 'Classifier' and args.model != 'RClassifier':
        args.num_node_features = dataset.num_node_features
        args.num_edge_features = dataset.num_edge_features
        args.num_pin_features = dataset.num_pin_features
    if args.model == 'EHGNN':
        args.num_pos_features = dataset.num_pos_features


    if design == 'all':
        print(dataset.train_file_names)
        print(dataset.test_file_names)
        train_designs =  dataset.train_file_names
        test_designs = dataset.test_file_names
        train_sets = []
        test_sets = []
        test_loader = {}
        num_training = 0
        num_testing = 0
        for design in train_designs:
            train_sets.append(Subset(dataset,range(dataset.ptr[design],
                                                    dataset.ptr[design] + dataset.file_num[design])))
            num_training += dataset.file_num[design]
        for design in test_designs:
            test_set = Subset(dataset,range(dataset.ptr[design],
                                                    dataset.ptr[design] + dataset.file_num[design]))
            num_testing += dataset.file_num[design]
            test_loader[design] = DataLoader(test_set,batch_size=args.test_batch_size,shuffle=True)
        train_set = ConcatDataset(train_sets)
        print("Total %d training data, %d testing data."%(num_training,num_testing),flush=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    else:
        num_training = int(dataset.file_num[design] * train_ratio)
        num_testing = dataset.file_num[design] -  num_training
        test_loader = {}
        design_set = Subset(dataset,range(dataset.ptr[design],
                                         dataset.ptr[design] + dataset.file_num[design]))
        train_set, test_set = random_split(design_set,[num_training,num_testing])
        print("Total %d training data, %d testing data."%(num_training,num_testing),flush=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader[design] = DataLoader(test_set,batch_size=args.test_batch_size,shuffle=False)
    return dataset, train_loader, test_loader


def build_model():
    Model = getattr(hmodels,args.model)
    model = Model(args).to(args.device)
    print(model)
    if args.optimizer == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SWATS':
       optimizer = swats.SWATS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Ranger':
        optimizer = optim.Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'Nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
    else:
        optimizer = getattr(torch.optim,args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    #optimizer.zero_grad()
    return model, optimizer, schedule


def build_log():
    # make save dir
    st = time.strftime("%b:%d:%X",time.localtime())
    args.save_dir = os.path.join(args.save_dir,'{}_{}_{}_{}'.format(args.model,args.label,args.group,st))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # rederict to save dir
    sys.stdout = Logger(path=args.save_dir)
    # print args
    print(args)
    # save paths
    best_model_path = os.path.join(args.save_dir,'best.pth'.format(st))
    last_model_path = os.path.join(args.save_dir,'last.pth'.format(st))
    # tensor board logger
    logger = writer.SummaryWriter(args.save_dir)
    return best_model_path, last_model_path, logger


# preparing
torch.set_num_threads(16)

# choose data group
if args.group == 1: 
    tmp = args.tests
    args.tests = args.trains
    args.trains = tmp
if args.model == 'Classifier' or args.model == 'GClassifier':
    args.loss = 'BCE'
    args.acc = 'BEQ'
if args.model == 'RClassifier':
    args.loss = 'COMB'
    args.acc = 'COMB'

args.label = [int(i) for i in args.label]
set_seed(args.seed)
# build up
best_model_path, last_model_path, logger = build_log()
print('loading dataset ...')
dataset, train_loader, test_loader = build_loader(args.design, args.train_ratio)
model, optimizer, schedule = build_model()
criterion = build_loss(args)
accuracy = build_acc(args)
test_designs = dataset.test_file_names

start = 0
min_loss = 1e10
min_train_loss = 1e10
min_err = 1e10
patience = 0

def test(model,loader):
    with torch.no_grad():
        model.eval()
        lenth = len(loader)# if epoch % 5 == 0 else int(len(loader)/5)
        maes = []
        accs = []
        ipes = []
        for i,label in enumerate(args.label):
            correct = 0.
            loss = 0.
            reals = []
            preds = []
            for i, data in enumerate(loader):
                if i >= lenth : break
                data = data.to(args.device)
                out = model(data).view(-1)
                y = data.y[:, label].view(-1)

                preds.extend(out.detach().cpu().numpy().tolist())
                reals.extend(data.y[:, label].cpu().numpy().tolist())
                
                correct += torch.mean(torch.abs((y-out)/y)).item()
                loss += F.l1_loss(out, y).item()
            # rank loss
            Rp = np.argsort(preds)
            Rr = np.argsort(np.array(reals)[Rp])
            rankacc = InversePairs(Rr.tolist()) / (len(reals)**2 - len(reals)) * 2
            #print('[{}]MAE=\t{:4f}\tMRE={:4f}\tIPE={:4f}'.format(label,loss/len(loader),correct/len(loader),rankacc),end='\t')
            maes.append(loss/lenth)
            accs.append(correct/lenth)
            ipes.append(rankacc)
    return np.mean(maes), np.mean(accs), np.mean(ipes)


def test_class(model,loader):
    tmp_mode = dataset.mode
    dataset.mode = 'CNN'
    with torch.no_grad():
        model.eval()
        lenth = len(loader)# if epoch % 5 == 0 else int(len(loader)/5)
        for i,label in enumerate(args.label):
            reals = []
            preds = []
            for i, data in enumerate(loader):
                if i >= lenth : break
                data = data.to(args.device)
                out = model.predict(data).view(-1)
                preds.extend(out.view(-1).detach().cpu().numpy().tolist())
                reals.extend(data.y[:, label].cpu().numpy().tolist())

            # rank loss
            reals = np.array(reals)
            preds = np.array(preds)
            Rp = np.argsort(preds)
            Rr = np.argsort(np.array(reals)[Rp])
            rankacc = InversePairs(Rr.tolist()) / (len(reals)**2 - len(reals)) * 2
            dcg_s = dcg_score(input=preds, target=reals)
    dataset.mode = tmp_mode
    return 0, dcg_s, rankacc


def test_design(model,design, test_loader):
    if(args.model == 'Classifier' or args.model == 'GClassifier' or args.model=='RClassifier'): 
        return test_class(model, test_loader[design])
    return test(model, test_loader[design])

if args.goon:
    checkp = torch.load(args.checkp)
    model.load_state_dict(checkp['model'])
    print('load model from {}, saved at epoch {}'.format(args.checkp,start - 1))
    if args.con:
        optimizer.load_state_dict(checkp['optimizer'])

minn_loss = 10000
minn_errr = 10000

for epoch in range(start, args.epochs):
    model.train()
    tt = time.time()
    Ave_loss = 0.
    Ave_cor = 0.

    for i, data in enumerate(train_loader):
        data = data.to(args.device)

        out = [model(data)]
        loss = criterion(out, data) / args.batch_step
        loss.backward()
        if (i+1) % args.batch_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            Ave_loss += loss.mean().item()
            Ave_cor +=  accuracy(out,data).item()
            
    if optimizer.param_groups[0]['lr'] > args.lr / 100:
        schedule.step()
    val_losses = []
    rank_errs = []
    print("[Epoch\t{}]\tTrain loss:\t{:.4f}\tTrain acc:\t{:.4f}".format(
            epoch, Ave_loss / len(train_loader) * args.batch_step, 
            Ave_cor / len(train_loader)), flush=True,end='\t')
        
    for design in test_designs:
        _, val_loss, rank_err = test_design(model, design, test_loader)
        val_losses.append(val_loss)
        rank_errs.append(rank_err)
    
    mean_val_loss = np.mean(val_losses)
    mean_rank_err = np.mean(rank_errs)

    print("{} mre:\t{:.4f}\t{} ipe:\t{:.4f}\tTime:{:.2f}\tlr:{:.5f}".format(
            'Test',
            mean_val_loss, 
            'Test',
            mean_rank_err,
            time.time() - tt, 
            optimizer.param_groups[0]['lr']))
    
    logger.add_scalar('train loss', Ave_loss / len(train_loader), i)
    logger.add_scalar('train acc', Ave_cor / len(train_loader), i)
    logger.add_scalar('test mre', mean_val_loss, i)
    logger.add_scalar('test rank err', mean_rank_err, i)


    if mean_val_loss < minn_loss:
        minn_loss = mean_val_loss
        state = {'model': model.state_dict(), 'epoch': epoch , 'val_loss' : mean_val_loss, 'rank_err' : mean_rank_err}
        print('model saved {} {}'.format(mean_val_loss, mean_rank_err))
        torch.save(state, best_model_path + '.loss')


    if mean_rank_err < minn_errr:
        minn_errr = mean_rank_err
        state = {'model': model.state_dict(), 'epoch': epoch , 'val_loss' : mean_val_loss, 'rank_err' : mean_rank_err}
        print('model saved {} {}'.format(mean_val_loss, mean_rank_err))
        torch.save(state, best_model_path + '.err')

state = {'model': model.state_dict(), 'val_loss' : mean_val_loss, 'rank_err' : mean_rank_err}
torch.save(state, last_model_path)

