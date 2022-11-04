import torch
import numpy as np
from torch._C import Graph
from torch_geometric.nn import HypergraphConv, Sequential
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGPooling as Pool
from torch_geometric.nn.conv.sage_conv import SAGEConv
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as F
import os.path as osp
from uuid import uuid1
from itertools import chain
from inspect import Parameter
from typing import List, Optional, Set
from torch_geometric.typing import Adj, Size
import pdb
from torch import Tensor
from jinja2 import Template
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr
from .layers import V2EConv,V2ELAFConv,E2VConv, HyperConv, HyperSAGPool, AttHyperConv, AttHypergraphConv, AttPNAHyperConv, PNAHyperedgeConv
from torchvision import models
import torch.nn as nn

class SAGPoolh(torch.nn.Module):
    def __init__(self,args):
        super(SAGPoolh, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = args.use_real_pos
        self.negative_slope = 0.1
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = nn.Sequential( 
            HyperConv(self.num_features, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        self.conv2 = nn.Sequential( 
            HypergraphConv(self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        self.conv3 =  nn.Sequential( 
            HypergraphConv(self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope))

        self.mlp = nn.Sequential(
            nn.Linear(self.nhid*2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.nhid//2, self.num_classes)
        )

    def forward(self, data):
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]

        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 

        x= torch.cat((x,fake_pos,ismacro),dim=-1)

        x = self.conv1((x, edge_index, pin_feature))
        x1 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = self.conv2((x, edge_index))
        x2 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = self.conv3((x, edge_index))
        x3 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.mlp(x)
        return x

class SAGPoolh_feature(torch.nn.Module):
    def __init__(self,args):
        super(SAGPoolh_feature, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        #self.max_edge_nums
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = HyperConv(self.num_features, self.nhid)
        self.mlp1 = nn.Sequential(
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True),
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True),
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True))
        #self.pool1 = HyperSAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = HypergraphConv(self.nhid, self.nhid)
        self.mlp2 = nn.Sequential(
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True),
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True),
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True))
        #self.pool2 = HyperSAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = HypergraphConv(self.nhid, self.nhid)
        #self.pool3 = HyperSAGPool(self.nhid, ratio=self.pooling_ratio)

    def forward(self, data):
        #x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        x, edge_index, pin_feature, batch, macro_index, macro_pos = data.x, data.edge_index, data.pin_feature, data.batch, data.macro_index,data.macro_pos
        # add macro pos
        tmp = macro_pos.new_ones((macro_pos.shape[0],1))
        macro_pos = torch.cat((macro_pos,tmp),dim=-1)
        pos = x.new_zeros((x.shape[0],macro_pos.shape[1]))
        pos[macro_index] = macro_pos

        x= torch.cat((x,pos),dim=-1)

        x = F.relu(self.conv1(x, edge_index, pin_feature))
        #x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.mlp1(x)

        x = F.relu(self.conv2(x, edge_index))
        #x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.mlp2(x)

        x = F.relu(self.conv3(x, edge_index))
        #x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        return x

class SAGPoolh_comb(torch.nn.Module):
    def __init__(self,args):
        super(SAGPoolh_comb, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        #self.max_edge_nums
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.graph_feature = SAGPoolh_feature(args=args)
        self.photo_feature = models.vgg16(pretrained=True)
        self.photo_feature.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.nhid*2),  
        )
        self.predict = nn.Sequential(
            nn.Linear(self.nhid*4, self.nhid),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes)
        )

    def forward(self, data):
        #x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        p = torch.cat([data.picture,data.picture,data.picture],dim=1)

        x=self.graph_feature(data)
        p=self.photo_feature(p)

        x = torch.cat([x, p], dim=1)

        x = self.predict(x)

        return x

class MLPlus(torch.nn.Module):
    def __init__(self,args):
        super(MLPlus, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        #self.max_edge_nums
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = HyperConv(self.num_features, self.nhid)
        self.mlp1 = nn.Sequential(
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True))

        self.conv2 = HypergraphConv(self.nhid, self.nhid)
        self.mlp2 = nn.Sequential(
        nn.Linear(self.nhid,self.nhid),
        #nn.Dropout(p=self.dropout_ratio),
        #nn.ReLU(True),
        #nn.Linear(self.nhid,self.nhid),
        nn.ReLU(True))

        self.conv3 = HypergraphConv(self.nhid, self.nhid)

        self.regress = nn.Sequential(
        nn.Linear(self.nhid*2, self.nhid),
        nn.Dropout(p=self.dropout_ratio),
        nn.ReLU(True),
        #nn.BatchNorm1d(self.nhid),
        nn.Linear(self.nhid, self.nhid//2),
        nn.Dropout(p=self.dropout_ratio),
        nn.ReLU(True),
        #nn.BatchNorm1d(self.nhid//2),
        nn.Linear(self.nhid//2, self.num_classes))

    def forward(self, data):
        #x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        x, edge_index, pin_feature, batch, macro_index, macro_pos = data.x, data.edge_index, data.pin_feature, data.batch, data.macro_index,data.macro_pos
        # add macro pos
        tmp = macro_pos.new_ones((macro_pos.shape[0],1))
        macro_pos = torch.cat((macro_pos,tmp),dim=-1)
        pos = x.new_zeros((x.shape[0],macro_pos.shape[1]))
        pos[macro_index] = macro_pos

        x= torch.cat((x,pos),dim=-1)

        x = F.relu(self.conv1(x, edge_index, pin_feature))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.mlp1(x)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.mlp2(x)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.regress(x)

        return x.view(-1)

class CNN(torch.nn.Module):
    def __init__(self,args):
        super(CNN, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.net = models.vgg16(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),  
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128,self.num_classes)
        )



    def forward(self, data):
        x = data.picture
        x = self.net(x)
        return x

class ANet(torch.nn.Module):
    def __init__(self,args):
        super(ANet, self).__init__()
        self.args = args
        self.num_features = args.num_features 
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = args.use_real_pos
        self.attention = args.attention
        self.negative_slope = 0.1
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = AttHyperConv(self.num_features, self.nhid,use_attention=self.attention)
        self.conv2 = AttHypergraphConv(self.nhid, self.nhid,use_attention=self.attention)
        self.conv3 = AttHypergraphConv(self.nhid, self.nhid,use_attention=self.attention)

        self.mlp = nn.Sequential(
            nn.Linear(self.nhid*2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.nhid//2, self.num_classes)
        )

    def forward(self, data):
        
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]

        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 

        x= torch.cat((x,fake_pos,ismacro),dim=-1)

        x = F.leaky_relu(self.conv1(x, edge_index, pin_feature),negative_slope=self.negative_slope)
        
        x1 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = F.leaky_relu(self.conv2(x, edge_index),negative_slope=self.negative_slope)

        x2 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = F.leaky_relu(self.conv3(x, edge_index),negative_slope=self.negative_slope)

        x3 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.mlp(x)
        return x


class BNet(torch.nn.Module):
    def __init__(self,args):
        super(BNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = False
        self.heads = 1
        self.attention = args.attention
        self.negative_slope = 0.1
        #self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.use_pna = args.pna

        self.conv1 =AttPNAHyperConv(self.num_features, self.nhid, 
                                use_attention=self.attention,
                                heads=self.heads, 
                                use_pna=self.use_pna, 
                                dropout=0)

        #    nn.Dropout(p=self.dropout_ratio)])

        #self.conv2 = Sequential('x, hyperegde_index',
        #   [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
        #                       use_attention=self.attention, 
        #                       dropout=self.dropout_ratio),
        #       'x, hyperegde_index -> x'),
        #   nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #   nn.Dropout(p=self.dropout_ratio)])

        #self.conv3 = Sequential('x, hyperegde_index',
        #   [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
        #                       use_attention=self.attention,
        #                       dropout=self.dropout_ratio),
        #       'x, hyperegde_index -> x'),
        #   nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #   nn.Dropout(p=self.dropout_ratio)])

        self.mlp = nn.Sequential(
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.heads * self.nhid*2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes))

    def forward(self, data):
        
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index
        # add macro pos
        batch = x.new_zeros(x.shape[0]).long()
        macro_batch = torch.zeros_like(macro_index).long()

        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 

        x= torch.cat((x,fake_pos,ismacro),dim=-1)

        x = F.leaky_relu(self.conv1(x, edge_index, pin_feature),self.negative_slope)
        
        x1 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv2(x, edge_index)

        #x2 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv3(x, edge_index)

        #x3 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = x1# + x2 + x3

        x = self.mlp(x)
        return x

class CNet(torch.nn.Module):
    def __init__(self,args):
        super(CNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = args.use_real_pos
        self.heads = 1
        self.attention = args.attention
        self.negative_slope = 0.1
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.use_pna = args.pna

        self.conv1 = PNAHyperedgeConv(self.num_features, self.nhid, 
                                use_attention=self.attention,
                                heads=self.heads, 
                                use_pna=self.use_pna, 
                                dropout=self.dropout_ratio)

        self.conv2 = Sequential('x, hyperegde_index',
            [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
                                use_attention=self.attention, 
                                dropout=self.dropout_ratio),
                'x, hyperegde_index -> x'),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio)])

        self.conv3 = Sequential('x, hyperegde_index',
            [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
                                use_attention=self.attention,
                                dropout=self.dropout_ratio),
                'x, hyperegde_index -> x'),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio)])

        self.mlp = nn.Sequential(
            nn.Linear(self.heads * self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes))

    def forward(self, data):
        
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        #macro_index = data.macro_index
        # add macro pos
        #macro_batch = batch[macro_index]
        edge_batch = scatter(batch,edge_index[1],dim=-1,dim_size=edge_index[1].max()+1,reduce='mean').long()
        #ismacro = x.new_zeros((x.shape[0],1))
        #ismacro[macro_index] = 1 
        #pdb.set_trace()
        #x= torch.cat((x,fake_pos,ismacro),dim=-1)
        #x= torch.cat((x,fake_pos),dim=-1)

        x = self.conv1(fake_pos, edge_index, pin_feature)
        
        x1 = gap(x, edge_batch)

        # x = self.conv2(x, edge_index)

        # x2 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        # x = self.conv3(x, edge_index)

        # x3 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = x1# + x2 + x3

        x = self.mlp(x)
        return x


class DNet(torch.nn.Module):
    def __init__(self,args):
        super(DNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = args.use_real_pos
        self.heads = 1
        self.attention = args.attention
        self.negative_slope = 0.1
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.use_pna = args.pna

        self.v2econv = V2EConv(self.num_features, self.nhid, 
                                use_pna=self.use_pna)
        #    [(,
        #        'x, hyperegde_index, pin_feature -> x_j, '),
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
        self.e2vconv = E2VConv(self.nhid, self.nhid, 
                                use_attention=self.attention,
                                heads=self.heads, 
                                dropout=self.dropout_ratio)
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #    nn.Dropout(p=self.dropout_ratio)])

        # self.conv2 = Sequential('x, hyperegde_index',
        #    [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
        #                        use_attention=self.attention, 
        #                        dropout=self.dropout_ratio),
        #        'x, hyperegde_index -> x'),
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        # #   nn.Dropout(p=self.dropout_ratio)])

        # self.conv3 = Sequential('x, hyperegde_index',
        #    [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
        #                        use_attention=self.attention,
        #                        dropout=self.dropout_ratio),
        #        'x, hyperegde_index -> x'),
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        # #   nn.Dropout(p=self.dropout_ratio)])

        self.mlp = nn.Sequential(
            nn.Linear(self.heads * self.nhid*2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes))

        self.mlp2 = nn.Sequential(
            nn.Linear(self.heads * self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.nhid//4),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//4, self.num_classes))

    def forward(self, data):
        
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]

        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 

        x= torch.cat((x,fake_pos,ismacro),dim=-1)

        x, edge_attr = self.v2econv(x, edge_index, pin_feature)

        x = self.e2vconv(x, edge_index, edge_attr)
        
        x1 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv2(x, edge_index)

        #x2 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv3(x, edge_index)

        #x3 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = x1# + x2 + x3

        x = self.mlp(x)
        return x, self.mlp2(edge_attr)


class ENet(torch.nn.Module):
    def __init__(self,args):
        super(ENet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = args.use_real_pos
        self.heads = 1
        self.attention = args.attention
        self.negative_slope = 0.1
        self.meta_feature  = 13
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.use_pna = args.pna

        self.conv1 = Sequential('x, hyperegde_index, pin_feature',
            [(AttPNAHyperConv(self.num_features, self.nhid, 
                                use_attention=self.attention,
                                heads=self.heads, 
                                use_pna=self.use_pna, 
                                dropout=0),
                'x, hyperegde_index, pin_feature -> x'),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #    nn.Dropout(p=self.dropout_ratio)])

        self.conv2 = Sequential('x, hyperegde_index',
           [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
                               use_attention=self.attention, 
                               dropout=self.dropout_ratio),
               'x, hyperegde_index -> x'),
           nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #   nn.Dropout(p=self.dropout_ratio)])

        self.conv3 = Sequential('x, hyperegde_index',
           [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
                               use_attention=self.attention,
                               dropout=self.dropout_ratio),
               'x, hyperegde_index -> x'),
           nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #   nn.Dropout(p=self.dropout_ratio)])

        self.metafc= nn.Sequential(
            nn.Linear(self.meta_feature, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True))


        self.mlp = nn.Sequential(
            nn.Linear(self.heads * self.nhid*2 + self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes))

    def forward(self, data):
        
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index

        meta_feature = data.meta_feature
        
        # add macro pos
        macro_batch = batch[macro_index]

        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 

        x= torch.cat((x,fake_pos,ismacro),dim=-1)

        x = self.conv1(x, edge_index, pin_feature)
        
        x1 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv2(x, edge_index)

        #x2 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv3(x, edge_index)

        #x3 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = x1# + x2 + x3

        meta_feature = self.metafc(meta_feature)
        #pdb.set_trace()
        x = torch.cat([x,meta_feature.view(-1,self.nhid)],dim=-1)

        x = self.mlp(x)
        return x


class LNet(torch.nn.Module):
    def __init__(self,args):
        super(LNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = args.use_real_pos
        self.heads = 1
        self.attention = args.attention
        self.negative_slope = 0.1
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.use_pna = args.pna
        self.units = 4

        self.v2econv = V2ELAFConv(self.num_features, self.nhid, units = self.units)
        #    [(,
        #        'x, hyperegde_index, pin_feature -> x_j, '),
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
        self.e2vconv = E2VConv(self.nhid, self.nhid, 
                                use_attention=self.attention,
                                heads=self.heads, 
                                dropout=self.dropout_ratio)
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #    nn.Dropout(p=self.dropout_ratio)])

        # self.conv2 = Sequential('x, hyperegde_index',
        #    [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
        #                        use_attention=self.attention, 
        #                        dropout=self.dropout_ratio),
        #        'x, hyperegde_index -> x'),
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        # #   nn.Dropout(p=self.dropout_ratio)])

        # self.conv3 = Sequential('x, hyperegde_index',
        #    [(AttHypergraphConv(self.heads * self.nhid, self.heads * self.nhid,
        #                        use_attention=self.attention,
        #                        dropout=self.dropout_ratio),
        #        'x, hyperegde_index -> x'),
        #    nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        # #   nn.Dropout(p=self.dropout_ratio)])

        self.mlp = nn.Sequential(
            nn.Linear(self.heads * self.nhid*2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes))

        self.mlp2 = nn.Sequential(
            nn.Linear(self.heads * self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.nhid//4),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//4, self.num_classes))

    def forward(self, data):
        
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]

        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 

        x= torch.cat((x,fake_pos,ismacro),dim=-1)

        x, edge_attr = self.v2econv(x, edge_index, pin_feature)

        x = self.e2vconv(x, edge_index, edge_attr)
        
        x1 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv2(x, edge_index)

        #x2 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        #x = self.conv3(x, edge_index)

        #x3 = torch.cat([gap(x[macro_index], macro_batch), gap(x, batch)], dim=1)

        x = x1# + x2 + x3

        x = self.mlp(x)
        return x, self.mlp2(edge_attr)


def build_cg_index(macro_pos,size):
    with torch.no_grad():
        num = macro_pos.shape[0]
        edge_index_v = []
        edge_index_h = []
        for i in range(num):
            for j in range(i+1,num):
                pi = macro_pos[i]
                pj = macro_pos[j]
                si = size[i]
                sj = size[j]
                # add  h edge
                if pi[0] + si[0] <= pj[0]:
                    edge_index_v.append([i,j])
                if pj[0] + sj[0] <= pi[0]:
                    edge_index_v.append([j,i])
                if pi[1] + si[1] <= pj[1]:
                    edge_index_h.append([i,j])
                if pj[1] + sj[1] <= pi[1]:
                    edge_index_h.append([j,i])
    return torch.from_numpy(np.array(edge_index_v).T), torch.from_numpy(np.array(edge_index_h).T)

class TOPONet(torch.nn.Module):
    def __init__(self,args):
        super(TOPONet, self).__init__()
        self.device = args.device
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.use_real_pos = args.use_real_pos
        self.heads = 1
        self.attention = args.attention
        self.negative_slope = 0.1
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.use_pna = args.pna

        self.conv1 = Sequential('x, hyperegde_index, pin_feature',
            [(AttPNAHyperConv(self.num_features, self.nhid, 
                                use_attention=self.attention,
                                heads=self.heads, 
                                use_pna=self.use_pna, 
                                dropout=0),
                'x, hyperegde_index, pin_feature -> x'),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)])
        #    nn.Dropout(p=self.dropout_ratio)])

        self.conv2 = SAGEConv(self.nhid,self.nhid)
        self.conv3 = SAGEConv(2 * self.nhid, self.nhid)
        
        #   nn.Dropout(p=self.dropout_ratio)])

        self.mlp = nn.Sequential(
            nn.Linear(self.nhid*2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            #nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes))

    def forward(self, data):
        
        x, edge_index, pin_feature, batch = data.x, data.edge_index, data.pin_feature, data.batch
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index
        # add macro pos
        macro_size = x[macro_index]
        macro_pos = fake_pos[macro_index]
        macro_batch = batch[macro_index]

        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 

        x= torch.cat((x,fake_pos,ismacro),dim=-1)

        x = self.conv1(x, edge_index, pin_feature)
        
        x = x[macro_index]

        v_index, h_index =  build_cg_index(macro_pos, macro_size)
        #pdb.set_trace()
        v_index = v_index.to(self.device).long()
        h_index = h_index.to(self.device).long()

        x_v, x_h = F.leaky_relu(self.conv2(x, v_index)), F.leaky_relu(self.conv2(x, h_index))

        x = torch.cat([x_v,x_h],dim=-1)

        x_v, x_h = F.leaky_relu(self.conv3(x,v_index)), F.leaky_relu(self.conv3(x, h_index))

        x = torch.cat([x_v,x_h],dim=-1)

        x = self.mlp(x)
        return x

class CGNet(torch.nn.Module):
    def __init__(self,args):
        super(CGNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = 32
        self.num_classes = 1
        self.use_real_pos = False
        self.heads = 1
        self.attention = args.attention
        self.negative_slope = 0.1
        self.dropout_ratio = 0.5
        self.use_pna = True
        self.num_bins = 224
        self.bin_size = 1./self.num_bins

        self.gnn = AttPNAHyperConv(self.num_features, self.nhid, 
                        use_attention=self.attention,
                        heads=self.heads, 
                        use_pna=self.use_pna, 
                        dropout=0)

        # self.cnn = models.resnet18(pretrained=True)
        # #print(self.cnn)
        # self.cnn.conv1 = nn.Conv2d(
        #     self.nhid, 64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2), bias=False)
        # self.cnn.fc = nn.Sequential(
        #     nn.Linear(512, 128),  
        #     nn.ReLU(True),
        #     nn.Dropout(self.dropout_ratio),
        #     nn.Linear(128,32),
        #     nn.ReLU(True),
        #     nn.Dropout(self.dropout_ratio),
        #     nn.Linear(32,self.num_classes)
        # )
        self.cnn = models.vgg11(pretrained=True)
        self.cnn.features[0] = nn.Conv2d(
            self.nhid, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.cnn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),  
            nn.ReLU(True),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(2048,128),
            nn.ReLU(True),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(128,self.num_classes)
        )

    def forward(self, data):
        # gnn part
        x, edge_index, pin_feature = data.x, data.edge_index, data.pin_feature
        if self.use_real_pos:
            fake_pos = data.cell_pos
        else:
            fake_pos = data.fake_pos
        macro_index = data.macro_index
        density = data.density
        # add macro pos
        ismacro = x.new_zeros((x.shape[0],1))
        ismacro[macro_index] = 1 
        x = torch.cat((x,fake_pos,ismacro),dim=-1)
        x = self.gnn(x, edge_index, pin_feature)
        x = F.leaky_relu(x, self.negative_slope)
        # cnn part
        x = x[macro_index]

        minn,_ = torch.min(x,dim=0)
        maxx,_ = torch.max(x,dim=0)
        x = (x - minn)/(maxx-minn)
        #pdb.set_trace()
        x = torch.matmul(density.float(),x).transpose(0,2)
        x= x - x.mean(dim=0)
        
        x = x.view(1,self.nhid,self.num_bins,self.num_bins)

        x = self.cnn(x)

        return x
