from random import random
from tkinter import Scrollbar
from sklearn.metrics import top_k_accuracy_score
import torch
import torch.nn.functional as F
import argparse
import os
import time
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split, Subset, ConcatDataset
import pdb
import sys
#import swats
from torch_scatter import scatter
import src.hyperdataset as hdatasets
import torch_optimizer as optim
import src.hypermodel as hmodels
from src.logger import Logger
from torch_geometric.loader import DataLoader
from src.util import InversePairs, kendall,mykendall, normalization, rank, top_k_match
from torch.utils.tensorboard import writer
import matplotlib.pyplot as plt 
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def build_test_loader():
    MySet = getattr(hdatasets,args.dataset)

    dataset = MySet(args.dataset_path, mode=args.model, test_files=args.tests, train_files=args.trains, args=args)
    
    if args.model != 'CNN' and args.model != 'Classifier':
        args.num_node_features = dataset.num_node_features
        args.num_edge_features = dataset.num_edge_features
        args.num_pin_features = dataset.num_pin_features
    if args.model == 'EHGNN':
        args.num_pos_features = dataset.num_pos_features

    loader = {}
    for design in dataset.raw_file_names:
        design_set = Subset(dataset,range(dataset.ptr[design],
                                            dataset.ptr[design] + dataset.file_num[design]))
        loader[design] = DataLoader(design_set, batch_size= 1)
    return dataset, loader


def build_model():
    Model = getattr(hmodels,args.model)
    model = Model(args).to(args.device)
    #print(model)
    return model 


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
    return best_model_path, last_model_path


# preparing
torch.set_num_threads(16)
# choose data group
if args.group == 1: 
    tmp = args.tests
    args.tests = args.trains
    args.trains = tmp

label = [int(i) for i in args.label][0]
set_seed(args.seed)
# build up
# print('loading dataset ...')
dataset, loader = build_test_loader()
model = build_model()

os.makedirs('log/{}'.format(args.checkp), exist_ok=True)
logger = writer.SummaryWriter('log/{}'.format(args.checkp))

checkp = torch.load(args.checkp, map_location='cuda')
model.load_state_dict(checkp['model'])
model = model.to(args.device)
#print('load model from {}, loss = {}, err = {}'.format(args.checkp,checkp['val_loss'], checkp['rank_err']))
# golds = {}
# for design in args.tests:
#     test_loader = loader[design]
#     preds = []
#     reals = []
#     origins = []
#     label_p = 'data/raw/{}/labels.txt'.format(design)
#     idx_p = 'data/raw/{}/names.txt'.format(design)
#     golds[design] = np.loadtxt(label_p)[np.loadtxt(idx_p,dtype=int)]

# meann = np.mean(labels, 0 )
# maxx = np.max(labels, 0 )
# minn = np.min(labels, 0 )
# pdb.set_trace()
dataset.mode = 'CNN'
mres = 0
taut = 0
score = 0
print('model =', args.model, ", test group = ", args.group + 1)
print("{:20}\t{:10}\t{:10}\t{}\t{}".format('design', 'mean_score', 'top30_score', 'mre', 'tau'))
with torch.no_grad():
    designs = []
    embdds = []
    model.eval()
    for design in args.tests:
        test_loader = loader[design]
        preds = []
        reals = []
        origins = []
        #print(design, end='\t\t')
        label_p = 'data/raw/{}/labels.txt'.format(design)
        idx_p = 'data/raw/{}/names.txt'.format(design)

        labels_this = np.loadtxt(label_p)[np.loadtxt(idx_p,dtype=int)]
        #pdb.set_trace()
        for i, data in enumerate(test_loader):
            data = data.to(args.device)
            if args.model == 'HGNN' or args.model == 'EHGNN':
                out = model(data)
            else:
                out = model.predict(data)
            #out = out * (maxx[label - 1] - minn[label - 1]) + meann[label - 1]
            #data.y[:, 1 : ] =  data.y[:, 1 : ].cuda() * (maxx - minn) + meann
            reals.append(data.y[:, label].view(-1).item())
            origins.append(dataset.origin[design][i][:, label].item())
            preds.append(out.view(-1).item())
        reals = np.array(reals)
        preds = np.array(preds)
        origins = np.array(origins)
        mre = np.mean(np.abs(reals - preds)/np.abs(reals))
        tau = mykendall(reals, preds)
        taut += tau/len(args.tests)
        mres += mre/len(args.tests)
        top30 = np.argsort(preds)[:30]
        top30_score = origins[top30].mean()
        mean_score = np.mean(origins)
        score += top30_score/mean_score/len(args.tests)

        print("{:20}\t{:>10.4f}\t{:>.4f}\t{:>.3f}\t{:>.3f}".format(design, mean_score, top30_score, mre, tau))

    print("{:20}\t{:>10.4f}\t{:>10.4f}\t{:>.3f}\t{:>.3f}".format('average', 1., score, mres, taut))
    #print('average mre = {:.3f}'.format(mres/len(args.tests)), ', tau = {:.3f}'.format(taut/len(args.tests)))
