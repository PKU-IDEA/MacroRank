from genericpath import exists
import torch
import numpy as np
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gsp
from torch_geometric.nn import SAGPooling as Pool
from torch_geometric.nn import SAGEConv, GATv2Conv, GINConv, ResGatedGraphConv
import torch.nn.functional as F
import pdb
from torch import Tensor, dropout
from torch_scatter import scatter, scatter_mean
import torch.nn as nn
from torchvision import models
import src.egnn as egnn_models
from torch.nn.utils.rnn import pad_sequence

def split_batch(x, ptr):
    if len(ptr) == 1: return [x]
    split_x = list(torch.split_with_sizes(x, ptr.cpu().numpy().tolist(), dim=0))
    return split_x

def unpad_sequence(padded_sequences, masks):
    unpadded_sequences = []

    for seq, mask in zip(padded_sequences, masks):
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences

class CNN(torch.nn.Module):
    """ cnn baseline """
    def __init__(self,args):
        super(CNN, self).__init__()
        self.args = args
        self.dropout = args.dropout_ratio
        self.num_classes = 1
        self.net = models.vgg11(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),  
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 1000),  
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(1000, 1),
        )

    def forward(self, data):

        x = data.density
        x = self.net(x)
        return x.view(-1)
    
    def predict(self, data):
        x = data.density
        x = self.net(x)
        return x.view(-1)

class RClassifier(torch.nn.Module):
    """ cnn baseline """
    def __init__(self,args):
        super(RClassifier, self).__init__()
        self.args = args
        self.num_classes = 1
        self.dropout = args.dropout_ratio
        self.net = models.vgg11(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 1),
        )

        self.out = nn.Sigmoid()

    def forward(self, data):
        x = data.density
        x = self.net(x)
        return x.view(-1)

    def predict(self, data):
        x = data.density
        return  self.net(x).view(-1)

class Classifier(torch.nn.Module):
    """ cnn baseline """
    def __init__(self,args):
        super(Classifier, self).__init__()
        self.args = args
        self.num_classes = 1
        self.dropout = args.dropout_ratio
        self.net = models.vgg11(pretrained=False)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 1),
        )

        self.out = nn.Sigmoid()

    def forward(self, data):
        x = data.density
        index = torch.arange(0, x.shape[0], 2).to(x.device)
        x0 = torch.index_select(x,dim=0,index=index)
        x1 = torch.index_select(x,dim=0,index=(index+1))
        s0 = self.net(x0)
        s1 = self.net(x1)

        x = self.out(s0 - s1)
        return x.view(-1)

    def predict(self, data):
        x = data.density
        return  self.net(x).view(-1)

class GClassifier(torch.nn.Module):
    """ cnn baseline """
    def __init__(self, args):
        super(GClassifier, self).__init__()
        self.args = args
        self.net = EHGNN(args=args)
        self.out = nn.Sigmoid()

    def forward(self, data):
        s0 = self.net.predict(data, data.macro_pos1)
        s1 = self.net.predict(data, data.macro_pos2)
        x = self.out(s0 - s1)
        return x.view(-1)

    def predict(self, data):
        return  self.net(data).view(-1)
    
    def test(self, data):
        return  self.net(data).view(-1)


class V2PLayer(torch.nn.Module):
    def __init__(self):
        super(V2PLayer, self).__init__()

    def forward(self, node_feat : Tensor, pin_feat : Tensor, edge_index : Tensor)->Tensor:
        # creare edges
        node_pin_feat = torch.index_select(node_feat, dim=0, index=edge_index[0])
        pin_feat = torch.cat([node_pin_feat, pin_feat], dim=-1)
        return pin_feat


class P2VLayer(torch.nn.Module):
    def __init__(self):
        super(P2VLayer, self).__init__()

    def forward(self, pin_feat : Tensor, edge_index : Tensor)->Tensor:
        # creare edges
        node_feat = scatter_mean(pin_feat, edge_index[0], dim=0)
        return node_feat


class HyperGATConv(torch.nn.Module):
    def __init__(self,in_nch=6, in_pch=2, in_ech=1, nhid=16, out_ch=16, dropout=0, leaky_relu=0.1):
        super(HyperGATConv, self).__init__()
        self.in_node_ch = in_nch
        self.in_pin_ch = in_pch
        self.in_edge_ch = in_ech
        self.nhid = nhid
        self.out_ch = out_ch
        self.dropout = dropout
        self.leaky_relu = leaky_relu

        self.Dropout = nn.Dropout(p=self.dropout, inplace=True)
        self.act = nn.LeakyReLU(self.leaky_relu)
        self.v2p = V2PLayer()
        self.p2e = GATv2Conv(in_channels=(in_nch + in_pch, in_ech), out_channels=nhid, dropout=dropout)
        self.e2p = SAGEConv(in_channels=(nhid, in_nch + in_pch), out_channels=out_ch)
        self.p2v = P2VLayer()

    def forward(self, node_feat : Tensor, pin_feat : Tensor,  edge_index : Tensor, edge_attr : Tensor):
        # creare edges
        pins_index = torch.arange(0, pin_feat.shape[0], 1).to(node_feat.device)
        if edge_attr.dim() == 1: edge_attr = edge_attr.view(-1, 1)
        pin_edge = torch.stack((pins_index, edge_index[1]),dim=0)
        edge_pin = torch.stack((edge_index[1], pins_index),dim=0)
        # forward, v2p
        pin_feat = self.v2p(node_feat = node_feat, pin_feat = pin_feat, edge_index = edge_index)
        # p2e
        pin_feat = self.Dropout(pin_feat)
        edge_feat = self.p2e(x=(pin_feat, edge_attr), edge_index=pin_edge)
        edge_feat = self.act(edge_feat)
        # e2p
        edge_feat = self.Dropout(edge_feat)
        pin_feat = self.e2p(x=(edge_feat, pin_feat), edge_index=edge_pin, size=(edge_attr.shape[0], pin_feat.shape[0]))
        pin_feat = self.act(pin_feat)
        # p2v
        node_feat = self.p2v(pin_feat=pin_feat, edge_index=edge_index)
        return node_feat, pin_feat


class EGNNet(torch.nn.Module):
    """ plain gnn baseline """
    def __init__(self, layers = 3, feat_dim=32, pos_dim=2, nhid=32, position_encoding = 0, num_nearest_neighbors=0, dropout = 0.,edge_dim=0, args=None):
        super(EGNNet, self).__init__()
        self.layers = layers
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.nhid = nhid
        self.position_encoding = position_encoding
        self.num_nearest_neighbors = num_nearest_neighbors
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.embedd = nn.Sequential(
            nn.Linear(feat_dim, nhid),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.convs = nn.ModuleList([])
        
        base_model = getattr(egnn_models, args.base_model)
        for i in range(layers):  
            self.convs.append(base_model(
                dim = nhid,                                     # input dimension
                edge_dim = edge_dim,                            # dimension of the edges, if exists, should be > 0
                m_dim = nhid,                                   # hidden model dimension
                fourier_features = position_encoding,           # number of fourier features for encoding of relative distance - defaults to none as in paper
                num_nearest_neighbors = num_nearest_neighbors,  # cap the number of neighbors doing message passing by relative distance
                dropout = dropout,                              # dropout
                norm_feats = True if args.model == 'GClassifier' else False,                             # whether to layernorm the features
                norm_coors = True,                              # whether to normalize the coordinates, using a strategy from the SE(3) Transformers paper    
                update_feats = True,                            # whether to update features - you can build a layer that only updates one or the other
                update_coors = True,                            # whether ot update coordinates
                only_sparse_neighbors = False,                  # using this would only allow message passing along adjacent neighbors, using the adjacency matrix passed in 
                valid_radius = float('inf'),                    # the valid radius each node considers for message passing
                m_pool_method = 'sum',                          # whether to mean or sum pool for output node representation
                soft_edges = True,                              # extra GLU on the edges, purportedly helps stabilize the network in updated version of the paper
                coor_weights_clamp_value = None,                # clamping of the coordinate updates, again, for stabilization purposes
                act=nn.LeakyReLU(negative_slope=0.1),
                use_rel_coord = True if (args.model == 'GClassifier' and args.label[0] == '3') else False
            ))
        

    def forward(self,  feat_ : Tensor, coor_ : Tensor,  batch : Tensor, edge_index : Tensor=None, edge_attr : Tensor=None):
        if self.embedd is not None : feat_ = self.embedd(feat_)
        # if batch.max() <= 200 :
        #     feats = split_batch(feat_, batch)
        #     feats = pad_sequence(feats, batch_first=True)
        #     coors = split_batch(coor_, batch)
        #     coors = pad_sequence(coors, batch_first=True)
        #     masks = split_batch(feat_.new_ones(feat_.shape[0], dtype=bool), batch)
        #     masks = pad_sequence(masks, batch_first=True)

        #     for i, conv in enumerate(self.convs):
        #         feats, coors = conv(feats, coors, mask=masks)

        #     feats = unpad_sequence(feats, masks)
        #     feats = torch.cat(feats, dim=0)
        # else:
        feats = split_batch(feat_, batch)
        coors = split_batch(coor_, batch)
        feats = [p.view(1, -1, self.feat_dim) for p in feats]
        coors = [p.view(1, -1, self.pos_dim) for p in coors]
        zipped = list(zip(feats, coors))

        for i, conv in enumerate(self.convs):
            zipped = list(map(lambda p: conv(p[0], p[1], num_nearest=128 if p[0].shape[1] > 128 else 0), zipped))

        feats = [p[0].view(-1, self.feat_dim) for p in zipped]
        feats = torch.cat(feats, dim=0)
        return feats


class HGNN(torch.nn.Module):
    """ plain gnn baseline """
    def __init__(self,args=None):
        super(HGNN, self).__init__()
        self.args = args
        self.out_ch = 1
        self.num_node_features = args.num_node_features
        self.num_pin_features = args.num_pin_features
        self.num_edge_features = args.num_edge_features
        self.nhid = args.nhid
        self.negative_slope = 0.1
        self.dropout_ratio = args.dropout_ratio
        self.conv_layers = args.layers
        self.skip_cnt = args.skip_cnt

        self.convs = nn.ModuleList([HyperGATConv(in_nch=self.num_node_features,in_pch=self.num_pin_features,
                                    in_ech=self.num_edge_features, nhid=self.nhid,
                                    out_ch=self.nhid, dropout=self.dropout_ratio)])
        for i in range(self.conv_layers - 1):
            self.convs.append(HyperGATConv(in_nch=self.nhid,in_pch=self.nhid, 
                                    in_ech=self.num_edge_features, nhid=self.nhid, 
                                    out_ch=self.nhid, dropout=self.dropout_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(self.nhid * 2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True),
            nn.Linear(self.nhid, self.out_ch))

    def forward(self, data):
        #pdb.set_trace()
        x, edge_index = data.x, data.edge_index
        pin_feat, edge_weight  = data.pin_offset, data.edge_weight
        batch, macro_index = data.batch, data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]
        # model forward
        for i in range(self.conv_layers):
            last_x, last_pin_feat = x, pin_feat
            x, pin_feat = self.convs[i](x, pin_feat, edge_index, edge_weight)
            if self.skip_cnt and i > 0: x, pin_feat = x + last_x, pin_feat + last_pin_feat
        # 
        macro_feature = x[macro_index]
        x = torch.cat([gap(macro_feature, macro_batch), gap(x, batch)], dim=-1)
        # mlp
        x = self.mlp(x)
        return x


class EHGNN(torch.nn.Module):
    """ egnn + gnn """
    def __init__(self,args=None):
        super(EHGNN, self).__init__()
        self.args = args
        self.out_ch = 1
        self.num_node_features = args.num_node_features
        self.num_pin_features = args.num_pin_features
        self.num_edge_features = args.num_edge_features
        self.nhid = args.nhid
        self.negative_slope = 0.1
        self.dropout_ratio = args.dropout_ratio
        self.conv_layers = args.layers
        self.skip_cnt = args.skip_cnt
        self.pos_encode = args.pos_encode
        self.pos_dim = 4
        self.num_egnn = args.egnn_layers
        self.egnn_dim = args.egnn_nhid

        self.convs = nn.ModuleList([HyperGATConv(in_nch=self.num_node_features,in_pch=self.num_pin_features,
                                    in_ech=self.num_edge_features, nhid=self.nhid,
                                    out_ch=self.nhid, dropout=self.dropout_ratio)])
        for i in range(self.conv_layers - 1):
            self.convs.append(HyperGATConv(in_nch=self.nhid,in_pch=self.nhid, 
                                    in_ech=self.num_edge_features, nhid=self.nhid, 
                                    out_ch=self.nhid, dropout=self.dropout_ratio))

        self.posnet = EGNNet(self.num_egnn, self.nhid, self.pos_dim, self.egnn_dim, position_encoding=self.pos_encode, dropout=self.dropout_ratio, args=args)

        self.mlp = nn.Sequential(
            nn.Linear(self.egnn_dim, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_ratio, inplace=True),
            nn.Linear(self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.nhid, self.out_ch))

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        pin_feat, edge_weight  = data.pin_offset, data.edge_weight
        batch, macro_index = data.batch, data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]
        macro_pos = data.macro_pos
        # model forward
        for i, conv in enumerate(self.convs):
            last_x, last_pin_feat = x, pin_feat
            x, pin_feat = conv(x, pin_feat, edge_index, edge_weight)
            if self.skip_cnt and i > 0: x, pin_feat = x + last_x, pin_feat + last_pin_feat
        # 
        macro_feature = x[macro_index]
        # EGNN for position feature
        feat = self.posnet(macro_feature, macro_pos, data.macro_num)
        # mlp
        #x = torch.cat([x, gap(feat, macro_batch)], dim=-1)
        x = gap(feat, macro_batch)
        x = self.mlp(x)
        return x

    def predict(self, data, macro_pos):
        """ eplicitly input macro_pos, since other info are all the same within a netlist """
        x, edge_index = data.x, data.edge_index
        pin_feat, edge_weight  = data.pin_offset, data.edge_weight
        batch, macro_index = data.batch, data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]
        # model forward
        for i, conv in enumerate(self.convs):
            last_x, last_pin_feat = x, pin_feat
            x, pin_feat = conv(x, pin_feat, edge_index, edge_weight)
            if self.skip_cnt and i > 0: x, pin_feat = x + last_x, pin_feat + last_pin_feat
        # macro feature
        macro_feature = x[macro_index]
        # EGNN for position feature
        feat = self.posnet(macro_feature, macro_pos, data.macro_num)
        # mlp
        x = gap(feat, macro_batch)
        x = self.mlp(x)
        return x


class CEHGNN(torch.nn.Module):
    """ plain gnn baseline """
    def __init__(self,args=None):
        super(CEHGNN, self).__init__()
        self.args = args
        self.out_ch = 1
        self.num_node_features = args.num_node_features
        self.num_pin_features = args.num_pin_features
        self.num_edge_features = args.num_edge_features
        self.nhid = args.nhid
        self.negative_slope = 0.1
        self.dropout_ratio = args.dropout_ratio
        self.conv_layers = args.layers
        self.skip_cnt = args.skip_cnt
        self.pos_encode = args.pos_encode
        self.pos_dim = 4
        self.num_egnn = args.egnn_layers
        self.egnn_dim = args.egnn_nhid

        self.convs = nn.ModuleList([HyperGATConv(in_nch=self.num_node_features,in_pch=self.num_pin_features,
                                    in_ech=self.num_edge_features, nhid=self.nhid,
                                    out_ch=self.nhid, dropout=self.dropout_ratio)])
        for i in range(self.conv_layers - 1):
            self.convs.append(HyperGATConv(in_nch=self.nhid,in_pch=self.nhid, 
                                    in_ech=self.num_edge_features, nhid=self.nhid, 
                                    out_ch=self.nhid, dropout=self.dropout_ratio))

        self.posnet = EGNNet(self.num_egnn, self.nhid, self.pos_dim, self.egnn_dim, position_encoding=self.pos_encode, dropout=self.dropout_ratio, args=args)

        self.net = models.vgg11(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512 * 7 * 7, self.egnn_dim),  
            nn.ReLU(True),
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.egnn_dim * 2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.nhid, self.out_ch))

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        pin_feat, edge_weight  = data.pin_offset, data.edge_weight
        batch, macro_index = data.batch, data.macro_index
        density = data.pic
        # add macro pos
        macro_batch = batch[macro_index]
        macro_pos = data.macro_pos
        # model forward
        for i, conv in enumerate(self.convs):
            last_x, last_pin_feat = x, pin_feat
            x, pin_feat = conv(x, pin_feat, edge_index, edge_weight)
            if self.skip_cnt and i > 0: x, pin_feat = x + last_x, pin_feat + last_pin_feat
        # 
        macro_feature = x[macro_index]
        x = torch.cat([gap(macro_feature, macro_batch), gap(x, batch)], dim=-1)
        # EGNN for position feature
        feat = self.posnet(macro_feature, macro_pos, data.macro_num)
        # mlp
        #x = torch.cat([x, gap(feat, macro_batch)], dim=-1)
        feat = gap(feat, macro_batch)
        # density feature
        density_feat = self.net(density)
        x = torch.cat([feat, density_feat], dim=-1)
        x = self.mlp(x)
        return x