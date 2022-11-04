import torch
from torch_scatter import scatter
from tqdm import tqdm
import pylab
import os.path as osp
import os
import numpy as np
import scipy.stats as stats

def position_encoding(position : torch.Tensor, L : int=4) -> torch.Tensor:
    """ inputs : position [n, d] 
    outputs : position [n, d * L * 2]"""
    n = position.shape[0]
    arr = torch.arange(0, L, 1)
    arr = torch.pow(2, arr) * np.pi
    arr = torch.stack((arr, arr + np.pi/2)).T.reshape(-1)
    encoding = position.view(n, -1, 1) * arr.view(1, 1, -1)
    encoding = torch.sin(encoding)
    encoding = encoding.view(n, -1)
    return encoding

def draw_rect(coord, size, path = 'figs'):
    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, aspect='equal')
    plt.axis('off')
    plt.xlim(xmax=1.2,xmin=-0.2)
    plt.ylim(ymax=1.2,ymin=-0.2)

    c , s  = coord, size
    #patches = [matplotlib.patches.Rectangle((x, y),w, h, alpha=0.2,color='blue') for x,y,w,h in zip(coord[0], coord[1], size[0], size[1])]
    #ax.add_collection(PatchCollection(patches))
    [ax.add_patch(plt.Rectangle((x, y),w, h, alpha=0.2,facecolor='blue')) for x,y,w,h in zip(c[0], c[1], s[0], s[1])]
    fig.savefig(os.path.join(path,"draw.png"),bbox_inches='tight')
    plt.close(fig)
    plt.cla()
    plt.clf()


def get_ensity_map(macro_index, num_bins, bin_size, node_pos, cell_size, edge_index, pins, B):
    density = []
    ox = macro_index.new_zeros(num_bins,num_bins).float()
    oy = macro_index.new_zeros(num_bins,num_bins).float()
    for idx in macro_index:
        pos = node_pos[idx]
        size = cell_size[idx]
        ox = torch.arange(0,1,bin_size,dtype=float).view(1,-1).repeat(num_bins,1)
        oy = torch.arange(0,1,bin_size,dtype=float).view(-1,1).repeat(1,num_bins)

        ox = torch.clamp((size[0]/2 + bin_size/2 - torch.abs(pos[0] - ox + size[0]/2 - bin_size/2)) / bin_size,0,1)
        oy = torch.clamp((size[1]/2 + bin_size/2 - torch.abs(pos[1] - oy + size[1]/2 - bin_size/2)) / bin_size,0,1)

        density.append((ox * oy).view(num_bins,num_bins,1))

    density = torch.cat(density,dim = -1)
    density_map = density.sum(dim=-1)

    pin_density = torch.zeros_like(density_map).view(-1)
    cnt_density = torch.zeros_like(density_map).view(-1)


    all_pin_pos = ((torch.index_select(node_pos,dim=0,index=edge_index[0]) + pins) /bin_size).long().clamp(0, num_bins - 1)
    def dd2d(index):
        return index[:,1] * num_bins + index[:,0]
    pin_mask = torch.zeros(all_pin_pos.shape[0]).bool()
    for pidx in macro_index:
        pin_mask |= (edge_index[0]==pidx)
    pin_pos = all_pin_pos[pin_mask] 
    indx = dd2d(pin_pos)

    pin_density = pin_density + scatter(all_pin_pos.new_ones(pin_pos.shape[0]), \
            indx, dim=0, dim_size=num_bins * num_bins, reduce='sum')
        
    cnt_density = cnt_density + scatter(B[pin_mask], \
            indx, dim=0, dim_size=num_bins * num_bins, reduce='sum')

    pin_density /= pin_density.max()
    cnt_density /= cnt_density.max()

    pin_density = pin_density.view(num_bins,num_bins)
    cnt_density = cnt_density.view(num_bins,num_bins)

    pic = torch.cat([density_map.view(1,1,num_bins,num_bins),pin_density.view(1,1,num_bins,num_bins),cnt_density.view(1,1,num_bins,num_bins)],dim=1)

    return pic

def diameter(hyperedge_index):
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1

    maxx = 1000
    for i in tqdm(range(0,num_nodes)):

        vec = torch.zeros(num_nodes,dtype=torch.long).cuda()
        vec[i] = 1
        cnt = 0
        while vec.sum()/num_nodes <= 0.9:
            vec = vec.index_select(-1, hyperedge_index[0])
            vec = scatter(vec,hyperedge_index[1], dim=0, dim_size=num_edges,reduce='max')
            vec = vec.index_select(-1,hyperedge_index[1])
            vec = scatter(vec,hyperedge_index[1], dim=0, dim_size=num_edges,reduce='max')
            cnt += 1
            if cnt > maxx:
                break
        if cnt < maxx:
            maxx = cnt 
            print(maxx)
    print(maxx)


def k_shortest(hyperedge_index,macro_index):
    macro_num = int(len(macro_index))
    node_num = int(hyperedge_index[0].max()+1)
    edge_num = int(hyperedge_index[1].max()+1)
    shortest_length = []
    for i in range(0,macro_num):
        macro_id = macro_index[i]
        steps = torch.zeros(node_num,dtype=torch.long).to('cuda:0')
        visited = torch.zeros(node_num,dtype=torch.long).to('cuda:0')
        visited[macro_id] = 1
        cnt = 0
        # newly added nodes mask
        new_node = torch.zeros(node_num,dtype=torch.long).to('cuda:0')
        new_node[macro_id] = 1
        steps[macro_id] = 10
        while torch.sum(new_node) > 0 :
            cnt += 1
            tmp_vec = visited.index_select(-1, hyperedge_index[0])
            tmp_vec = scatter(tmp_vec,hyperedge_index[1], dim=0, dim_size=edge_num,reduce='max')
            tmp_vec = tmp_vec.index_select(-1,hyperedge_index[1])
            tmp_vec = scatter(tmp_vec,hyperedge_index[0], dim=0, dim_size=node_num,reduce='max')
            new_node = (tmp_vec - visited).long()
            steps[new_node.bool()] = cnt
            visited = tmp_vec
        #print(visited)
        steps = torch.where(visited == 0, node_num + edge_num,steps)
        shortest_length.append(steps.view(node_num,1))
    shortest = torch.cat(shortest_length,dim=1)
    return shortest
        


def standardization(x):
    mean = torch.mean(x)
    std = torch.std(x)
    return (x-mean)/std

def normalization(x):
    minn = torch.min(x)[0]
    maxx = torch.max(x)[0]
    return (x-minn)/(maxx-minn)


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
                if pi[0] + si[0] < pj[0]:
                    edge_index_v.append([i,j])
                if pj[0] + sj[0] < pi[0]:
                    edge_index_v.append([j,i])
                if pi[1] + si[1] < pj[1]:
                    edge_index_h.append([i,j])
                if pj[1] + sj[1] < pi[1]:
                    edge_index_h.append([j,i])
    return torch.from_numpy(np.array(edge_index_v).T), torch.from_numpy(np.array(edge_index_h).T)
    

def MergeSort(data):
    n=len(data)
    #递归基
    if n==1:return data, 0
    #分两半来排序
    part1,part2=data[:n//2],data[n//2:]
    sorted_part1,s1=MergeSort(part1)
    sorted_part2,s2=MergeSort(part2)
    #排序后拼接这两半，拼接后先计数，然后将两个有序序列合并
    s,sorted_temp=0,sorted_part1+sorted_part2
    #用p、q两个指针指向两段，计算q中每个元素离插入点的index差
    p,q,len1,len_all=0,sorted_temp.index(sorted_part2[0]),len(sorted_part1),len(sorted_temp)
    while p<len1 and q<len_all:
        #移动p使p成为插入排序的插入点，计算要移动多少个位置
        while p<len1:
            if sorted_temp[q]<sorted_temp[p]:
                s+=len1-p
                break
            p+=1
        q+=1
    #完成排序，并把排序后的内容回溯给上一级做准备
    l=[]
    p,q=0,sorted_temp.index(sorted_part2[0])
    while p<len1 and q<len_all:
        if sorted_temp[p]<sorted_temp[q]:
            l.append(sorted_temp[p])
            p+=1
        else:
            l.append(sorted_temp[q])
            q+=1
    if p==len1:l+=sorted_temp[q:]
    if q==len_all:l+=sorted_part1[p:]
    return l,s+s1+s2

def InversePairs(data):
    # write code here
    #用归并排序，归并拼接后用计算排序时元素的index变动了少
    _,s=MergeSort(data)
    return s


def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean(dim=-1)
    target_n = target - target.mean(dim=-1)
    inv_norm = 1 / target_n.norm(dim=-1)
    inv_norm[inv_norm.isinf()] = 0
    pred_n = pred_n / pred_n.norm(dim=-1)
    target_n = target_n * inv_norm
    #pdb.set_trace()
    return (pred_n * target_n).sum(dim=-1).mean()


def spearman(
    target : torch.Tensor,
    pred: torch.Tensor,
) -> torch.Tensor:
    target = torch.argsort(target)
    target = torch.argsort(target).float()

    pred = torch.argsort(pred)
    pred = torch.argsort(pred).float()
    
    return corrcoef(target, pred / pred.shape[-1])


def kendall(target, pred):
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    return stats.kendalltau(target, pred)[0]

def mykendall(target, pred):
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    Rp = np.argsort(pred)
    Rr = np.argsort(np.array(target)[Rp])
    return 1 - 2 * InversePairs(Rr.tolist()) / (len(target)**2 - len(target)) * 2


def mle_loss(target : torch.Tensor, pred : torch.Tensor) -> torch.Tensor:
    perm = torch.argsort(target)
    pred = pred[perm]
    exp_pred = torch.exp(pred)
    sum_exp_pred = torch.cumsum(exp_pred, dim=-1)
    prob = exp_pred / sum_exp_pred
    log_prob = torch.log(prob)
    return  - torch.sum(log_prob)

def dcg_score(input, target):
    perm = input.argsort()
    out = target[perm]
    logi = np.arange(0, len(perm), 1) + 2
    logi = np.log2(logi)
    out = out / logi
    return out.mean()

def idcg_score(input):
    return dcg_score(-input, input)

def ndcg_score(input, target):
    return dcg_score(input, target) / idcg_score(target)


def top_k_match(input, target, k=30):
    p_idx = np.argsort(input)
    r_idx = np.argsort(target)
    pk = p_idx[:k]
    rk = r_idx[:k]
    cross = np.intersect1d(pk, rk)
    return len(cross) / k

def rank(input):
    perm = input.argsort()
    return perm.argsort()

def mean_dist(data):
    n = data.shape[0]
    data= data.view(n, -1, 1)
    dist = torch.cdist(data, data, p=1)
    dist = dist.view(n, -1)
    return torch.mean(dist, dim=-1).view(-1, 1)

if __name__ == '__main__':
    a = torch.randn(5, 10)
    meann  = mean_dist(a)
    print(meann)




