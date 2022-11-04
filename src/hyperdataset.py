from copy import deepcopy
from itertools import combinations
import os.path as osp
import pandas as pd
import torch
import numpy as np
import os
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data
from torchvision import transforms
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.util import mean_dist, position_encoding, draw_rect, get_ensity_map


class BipartiteData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x.size(0)], [self.edge_weight.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class PlainClusterSet(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, mode ='graph', pos_encoding=True, test_files=['mgc_fft_a','mgc_matrix_mult_b'], train_files=['mgc_fft_b'], device='cpu', args=None):
        self.args = args
        self.tot_file_num = None # int
        self.file_num = None # dict, file nums for each design
        self.ptr = None
        self.num_bins = 224
        self.bin_size = 1./224
        self.train_file_names = train_files
        self.test_file_names = test_files
        self.device = device
        self.mode = mode
        self.pos_encoding = pos_encoding
        # info
        self.labels = ['hpwl', 'rwl','vias','short', 'score']
        self.weight = {}
        # for label statistics
        self.stats = {}
        self.tot_labels = None
        self.tot_means = None
        self.tot_stds = None
        self.means = {}
        self.stds = {}
        # 
        super(PlainClusterSet, self).__init__(root, transform, pre_transform)
        # data prefech
        self.netlist = {}
        self.data = []
        self.y = torch.load(osp.join(self.processed_dir, 'labels.pt'))
        self.weight = torch.load(osp.join(self.processed_dir, 'weight.pt'))
        self.lambdda = torch.load(osp.join(self.processed_dir, 'lambda.pt'))
        self.dcg = torch.load(osp.join(self.processed_dir, 'dcg.pt'))
        self.origin = {}
        
        for design in self.raw_file_names:
            self.origin[design] = []
            self.netlist[design] = torch.load(osp.join(self.processed_dir, '{}.pt'.format(design))).to(device)
            
        for i in range(len(self.processed_file_names)):
            self.data.append(self.pre_load_data(i).to(device))

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_plain')

    @property
    def raw_file_names(self):
        names_path = osp.join(self.root,'raw','all.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return names.tolist()
    
    @property
    def num_node_features(self):
        if self[0].x is not None: return self[0].x.size(1)
        return 1
    
    @property
    def num_pin_features(self):
        if self[0].pin_offset is not None:return self[0].pin_offset.size(1)
        return 1
    
    @property
    def num_edge_features(self):
        if self[0].edge_weight is not None:return self[0].edge_weight.size(1)
        return 1

    @property
    def num_pos_features(self):
        if self[0].macro_pos is not None:return self[0].macro_pos.size(1)
        return 1

    @property
    def processed_file_names(self):
        if self.tot_file_num is None:
            self.tot_file_num = 0
            self.file_num = {}
            self.ptr = {}
            for design in self.raw_file_names:
                path = osp.join(self.raw_dir,design)
                name_path = osp.join(path,'names.txt')
                names = np.array(pd.read_table(name_path,header=None)).reshape(-1)
                self.tot_file_num += names.shape[0]
                self.file_num[design] = names.shape[0]
            self.ptr[self.raw_file_names[0]] = 0
            for i in range(1,int(len(self.raw_file_names))):
                self.ptr[self.raw_file_names[i]] = self.ptr[self.raw_file_names[i-1]] + self.file_num[self.raw_file_names[i-1]]
        return ['data_%d.pt'%i for i in range(0, self.tot_file_num)]


    def process(self):

        self.tot_labels = []
        i = 0
        for design in self.raw_file_names:
            # paths
            path = osp.join(self.raw_dir,design)
            size_path = osp.join(path,'node_size.txt')
            name_path = osp.join(path,'names.txt')
            pos_root = osp.join(path,'node_pos')
            pin_path = osp.join(path,'pins.txt')
            region_path = osp.join(path,'region.txt')
            macro_path = osp.join(path,'macro_index.txt')
            hpwl_path = osp.join(path,'hpwl.txt')
            meta_path = osp.join(path,'meta.txt')
            label_path = osp.join(path,'labels.txt')
            hedge_w_path = osp.join(path, 'edge_weights.txt')
            # loading ...
            pins = np.loadtxt(pin_path)
            size = np.loadtxt(size_path)
            hedge_w = torch.from_numpy(np.loadtxt(hedge_w_path)).float()
            
            incidence = pins[:,:2]
            pin_feature = pins[:,2:]
            xl,yl,xh,yh = np.loadtxt(region_path)
            
            macro_index = torch.tensor(np.loadtxt(macro_path),dtype=torch.long)
            names = np.loadtxt(name_path,dtype=int)

            hpwls = np.loadtxt(hpwl_path)
            meta_data = np.loadtxt(meta_path)
            labels = np.loadtxt(label_path)
            
            rWLs = labels[:,0]
            vias = labels[:,1]
            short = labels[:,2]
            score = labels[:,3]
            mask = (rWLs != 0)
            # labels statics
            self.stats[design] = np.stack([hpwls[mask], rWLs[mask], vias[mask], short[mask], score[mask]], axis=0)
            self.tot_labels.append(self.stats[design])

            meta_data[5] = meta_data[5]/(yh-yl)
            meta_data[8] = meta_data[8]/(yh-yl)/(xh-xl)
            meta_data[9] = meta_data[9]/(yh-yl)/(xh-xl)
            meta_data[10] = meta_data[10]/(yh-yl)/(xh-xl)

            meta_data = torch.from_numpy(meta_data).float()
            size[:,0] = size[:,0]/(xh-xl)
            size[:,1] = size[:,1]/(yh-yl)
            pin_feature[:,0] = pin_feature[:,0]/(xh-xl)
            pin_feature[:,1] = pin_feature[:,1]/(yh-yl)
            # std
            rWLs = rWLs/(xh-xl+yh-yl)*2
            rWLs = rWLs
            hpwls = hpwls/(xh-xl+yh-yl)*2
            hpwls = hpwls

            cell_size = torch.tensor(size, dtype=torch.float)
            edge_index = torch.tensor(incidence.T, dtype=torch.long)
            pins = torch.tensor(pin_feature,dtype=torch.float)

            num_nodes = cell_size.shape[0]
            num_egdes = hedge_w.shape[0]
            num_pins  = pins.shape[0]
            # node_degree
            D = scatter(torch.ones(num_pins), edge_index[0], dim=0, dim_size=num_nodes, reduce='sum')
            # add self loop to no edge block
            block_index = torch.where(D == 0)[0] # no edge connected
            if len(block_index) >0:
                self_loop_edge = torch.arange(num_egdes, num_egdes + block_index.shape[0], 1).long()
                self_loop_edge = torch.stack([block_index, self_loop_edge],dim=0)
                edge_index = torch.cat([edge_index, self_loop_edge],dim=-1)
                self_loop_pin = torch.zeros((block_index.shape[0], 2)).float()
                pins = torch.cat([pins, self_loop_pin], dim=0)
                self_loop_edge_w = torch.zeros(block_index.shape[0]).float()

                hedge_w =  torch.cat([hedge_w, self_loop_edge_w], dim=-1)

                num_egdes += block_index.shape[0]
                num_pins += block_index.shape[0]
            # edge_degree
            B = scatter(torch.ones(num_pins), edge_index[1], dim=0, dim_size=num_egdes, reduce='sum')
            B = torch.index_select(B, dim=-1, index=edge_index[1]).clamp(0,50)
            

            macro_mask = torch.zeros(cell_size.shape[0]).float()
            macro_mask[macro_index] = 1

            node_attr = torch.cat((cell_size, D.view(-1, 1), macro_mask.view(-1,1)),dim=-1)
            # netlist is the same
            data = Data(
                    # x = [size[2 or 16], degree[1], pins[1]]
                    node_attr=node_attr, 
                    edge_index=edge_index,
                    edge_weight=hedge_w.view(-1,1),
                    pin_offset=pins,
                    macro_index=macro_index)
            if osp.exists(osp.join(self.processed_dir, '{}.pt'.format(design))): continue
            torch.save(data, osp.join(self.processed_dir, '{}.pt'.format(design)))
            for name in tqdm(names):
                if osp.exists(osp.join(self.processed_dir, 'data_{}.pt'.format(i))):continue
                if hpwls[name] == 0: print('{}-{}'.format(design,name))
                pos_path = osp.join(pos_root,'%d.txt'%name)
                node_pos = torch.tensor(np.loadtxt(pos_path),dtype=torch.float) 
                # normalize
                node_pos[:,0] = (node_pos[:,0]-xl)/(xh-xl)
                node_pos[:,1] = (node_pos[:,1]-yl)/(yh-yl)
                # fill zero
                fake_pos = torch.zeros_like(node_pos)
                fake_pos[macro_index] = node_pos[macro_index]
                # density map
                pic = get_ensity_map(macro_index,self.num_bins,self.bin_size,node_pos, cell_size, edge_index, pins, B)

                data = Data(# position[ll][2 or 16]
                            pos=fake_pos.float(),
                            # label = [hpwl, rwl, vias, short, score]
                            y=torch.tensor([hpwls[name], rWLs[name], vias[name], short[name], score[name]],dtype=torch.float).view(1, -1), 
                            # density_map
                            pic = pic.float(),
                            # design
                            design = design)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

        self.tot_labels = np.hstack(self.tot_labels)
        self.tot_means = torch.from_numpy(np.mean(self.tot_labels, axis=-1))
        self.tot_maxs = torch.from_numpy(np.max(self.tot_labels, axis=-1))
        self.tot_mins = torch.from_numpy(np.min(self.tot_labels, axis=-1))
        self.tot_stds = torch.from_numpy(np.std(self.tot_labels, axis=-1))

        self.means = {}
        self.stds = {}
        self.lambdda = {}
        self.dcg = {}
        ws = []

        for design in self.raw_file_names:
            tmp = torch.from_numpy(self.stats[design])
            tmp = (tmp - tmp.mean(dim=-1).view(-1,1))/tmp.std(dim=-1).view(-1,1)
            logi = torch.argsort(tmp, dim=-1)
            logi = torch.argsort(logi, dim=-1)
            logi = torch.log2(logi+2)
            self.lambdda[design] = torch.softmax(-tmp, dim=-1) * tmp.shape[-1]
            self.dcg[design] = tmp / logi
            meann = mean_dist(self.dcg[design])
            self.dcg[design] = self.dcg[design] / meann
            print(design, self.dcg[design].max(), self.dcg[design].min(), self.dcg[design].std())
        

        for design in self.raw_file_names:
            self.stats[design] = (torch.from_numpy(self.stats[design]) - self.tot_means.view(-1,1))/(self.tot_maxs.view(-1,1) - self.tot_mins.view(-1,1))
            self.means[design] = torch.mean(self.stats[design], dim=-1)
            self.stds[design] = torch.std(self.stats[design], dim=-1)
            if design == 'mgc_pci_bridge32_b': 
                self.stds[design] = self.stds['mgc_fft_a']
            ws.append(self.stds[design].view(-1, 1))
        
        ws = torch.cat(ws, dim=-1)
        ws = 1 / ws
        mws = torch.mean(ws, dim=-1)
        ws = ws / mws.view(-1, 1)

        labes = []
        for i, design in enumerate(self.raw_file_names):
            self.weight[design] = ws[:, i].float()
        
        for i, design in enumerate(self.raw_file_names):
            labes.append(self.stats[design])
        
        labes = torch.cat(labes, dim=-1).float()

        torch.save(labes, osp.join(self.processed_dir, 'labels.pt'))
        torch.save(self.weight, osp.join(self.processed_dir, 'weight.pt'))
        torch.save(self.lambdda, osp.join(self.processed_dir, 'lambda.pt'))
        torch.save(self.dcg, osp.join(self.processed_dir, 'dcg.pt'))

            
    def len(self):
        return len(self.processed_file_names)

    def pre_load_data(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx))).to(self.device)
        design = data.design
        netlist = self.netlist[design]
        #
        path = osp.join(self.raw_dir,design)
        region_path = osp.join(path,'region.txt')
        xl,yl,xh,yh = np.loadtxt(region_path)
        data.y[:, 1] *= (xh+yh-xl-yl)/2
        self.origin[design].append(data.y)
        # normalize
        y = self.y[:, idx].view(1, -1)
        if self.mode == 'HGNN':
            w = self.weight[design].view(1, -1)
            size = netlist.node_attr[:,2]
            pe = position_encoding(data.pos)
            x = torch.cat([pe, netlist.node_attr], dim=-1)
            bipartdata = BipartiteData(x=x, edge_index=netlist.edge_index, y=y, pic = data.pic, \
                                    edge_weight=netlist.edge_weight, pin_offset=netlist.pin_offset,
                                    macro_index=netlist.macro_index, design = design, w=w)
        elif self.mode == 'EHGNN' or self.mode == 'CEHGNN':
            w = self.weight[design].view(1, -1)
            size = netlist.node_attr[netlist.macro_index, :2]
            pos = data.pos[netlist.macro_index]
            d4pos = torch.cat([pos, pos + size], dim=-1)
            x = netlist.node_attr
            offset = netlist.pin_offset
            bipartdata = BipartiteData(x=x, edge_index=netlist.edge_index, y=y, pic = data.pic, \
                                    edge_weight=netlist.edge_weight, pin_offset=offset,
                                    macro_index=netlist.macro_index, design = design, w=w, macro_num=netlist.macro_index.shape[0], macro_pos = d4pos)
        elif self.mode == 'CNN':
            w = self.weight[design].view(1, -1)
            bipartdata = Data(y=y, density = data.pic, design = design, w=w)
        elif self.mode == 'Classifier' or self.mode == 'RClassifier':
            w = self.weight[design].view(1, -1)
            bipartdata = Data(y=y, density = data.pic, design = design, w=w)
        elif self.mode == 'GClassifier':
            w = self.weight[design].view(1, -1)
            size = netlist.node_attr[netlist.macro_index, :2]
            pos = data.pos[netlist.macro_index]
            d4pos = torch.cat([pos, pos + size], dim=-1)
            x = netlist.node_attr
            offset = netlist.pin_offset
            bipartdata = BipartiteData(x=x, edge_index=netlist.edge_index, y=y, pic = data.pic, \
                                    edge_weight=netlist.edge_weight, pin_offset=offset,
                                    macro_index=netlist.macro_index, design = design, w=w, macro_num=netlist.macro_index.shape[0], macro_pos = d4pos)
        else:
            assert(False)
        return bipartdata

    def get(self, idx):
        if self.mode == 'Classifier' or self.mode == 'RClassifier':
            design = self.data[idx].design
            begin = self.ptr[design]
            lenth = self.file_num[design]
            select_pair = np.random.randint(begin, begin + lenth, 2)
            data1 = self.data[select_pair[0]]
            data2 = self.data[select_pair[1]]
            mask1, mask5, mask0 = (data1.y > data2.y), (data1.y == data2.y), (data1.y < data2.y)
            target = mask1 * 1 + 0.5 * mask5
            lambdd1 = self.lambdda[design][:, select_pair[0] - begin]
            lambdd2 = self.lambdda[design][:, select_pair[1] - begin]
            w = (lambdd1 - lambdd2).abs().view(1, -1)

            bidata = Data(y=target, density=torch.cat((data1.density, data2.density),dim=0), w=w, y1=data1.y, y2=data2.y, w1=data1.w, w2=data2.w)
            return bidata
        elif self.mode == 'GClassifier':
            # select data
            design = self.data[idx].design
            begin = self.ptr[design]
            lenth = self.file_num[design]
            select_pair = np.random.randint(begin, begin + lenth, 2)
            data1 = self.data[select_pair[0]]
            data2 = self.data[select_pair[1]]
            # get weight
            mask1, mask5, mask0 = (data1.y > data2.y), (data1.y == data2.y), (data1.y < data2.y)
            target = mask1 * 1 + 0.5 * mask5
            lambdd1 = self.lambdda[design][:, select_pair[0] - begin]
            lambdd2 = self.lambdda[design][:, select_pair[1] - begin]
            w = (lambdd1 - lambdd2).abs().view(1, -1)
            #
            netlist = self.netlist[design]
            bidata = BipartiteData(x=netlist.node_attr, edge_index=netlist.edge_index,  y1=data1.y, y2=data2.y, 
                                    y=target, pic1 = data1.pic, pic2 = data2.pic, edge_weight=netlist.edge_weight, 
                                    pin_offset=netlist.pin_offset, macro_index=netlist.macro_index, design = design, w=w, 
                                    macro_num=netlist.macro_index.shape[0], macro_pos1 = data1.macro_pos, macro_pos2 = data2.macro_pos)
            return bidata
        else :return self.data[idx]
    

    def pre_trans():
        return 

if __name__=='__main__':
    #from torch.utils.tensorboard import writer
    #logger = writer.SummaryWriter('visual')
    Set = PlainClusterSet('data', mode='HGNN')
    target = 1
    for m, design in enumerate(Set.raw_file_names):
        labels = []
        positions = []
        for i in range(Set.ptr[design], Set.file_num[design] + Set.ptr[design]):
            data = Set[i]
            pos = data.x[data.macro_index,:2]
            positions.append(pos.view(-1))
            labels.append(torch.load(osp.join(Set.processed_dir, 'data_{}.pt'.format(i))).y[:, target].view(-1).item())
        a = torch.stack(positions, dim=0)
        print(design, np.std(labels)/np.mean(labels))
        #logger.add_embedding(a, metadata=torch.tensor(labels), global_step=m)

    


