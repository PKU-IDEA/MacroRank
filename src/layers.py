from os import fsdecode
import numpy
from torch_geometric.nn import SAGEConv, GCNConv, HypergraphConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
import torch
from typing import Optional
import math
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter, scatter_std
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
import pdb


def global_sum_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='sum')


def diag_offdiag_max_pool(input):
    """ input : (batch , channal, node_num, node_num) or  (channal, node_num, node_num) """
    N = input.shape[-1]
    max_diag = torch.max(torch.diagonal(input,dim1=-2,dim2=-1),dim=-1)[0]
    max_val = torch.max(max_diag)
    min_val = torch.max(- input)
    val = torch.abs(torch.add(max_val,min_val))
    if input.dim() == 4:
        min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1,N,N)
    else:
        min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1,N,N)
    max_offdiag = torch.max(torch.max(input - min_mat,dim=-1)[0],dim=-1)[0]
    return torch.cat([max_diag, max_offdiag],dim=-1)

def diag_offdiag_mean_pool(input):
    """ input : (batch , channal, node_num, node_num) or  (channal, node_num, node_num) """
    N = input.shape[-1]
    mean_diag = torch.mean(torch.diagonal(input,dim1=-2,dim2=-1),dim=-1)
    mean_offdiag = (torch.sum(input,dim=[-1,-2]) - mean_diag * N) / (N*N - N)
    return torch.cat([mean_diag,mean_offdiag],dim=-1)

def dist_matrix(input):
    """ input : (batch , channal, node_num) or  (channal, node_num) """
    N = input.shape[-1]
    if input.dim() == 3:
        A = input.view(input.shape[0],input.shape[1],N,1).repeat(1,1,1,N)
        D = A - A.transpose(-1,-2)
    else :
        A = input.view(input.shape[0],N,1).repeat(1,1,N)
        D = A - A.transpose(-1,-2)
    return D

def dist_matrix_l2_angle(input):
    """ input : (batch , channal, node_num) or  (channal, node_num) """
    N = input.shape[-1]
    if input.dim() == 3:
        A = input.view(input.shape[0],input.shape[1],N,1).repeat(1,1,1,N)
        D = A - A.transpose(-1,-2)
        L = torch.sum(torch.sqrt(D**2),dim=-3)
        G = torch.arctan(D[-3,1]/D[-3,0]).view(-1,1,N,N)
    else :
        A = input.view(input.shape[0],N,1).repeat(1,1,N)
        D = A - A.transpose(-1,-2)
        L = torch.norm(D,p=2,dim=-3).view(1,N,N)
        G = torch.arctan(D[0]/D[1]).view(1,N,N)/numpy.pi
    G = torch.nan_to_num(G,0)
    return torch.cat([L,G],dim=-3)

def dir_matrix(n):
    D = torch.triu(torch.ones(n,n),diagonal=1)
    return torch.cat([D.view(1,n,n),D.t().view(1,n,n),(D + D.t()).view(1,n,n)],dim=0)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def filter_inst(hyperedge_index, hyperedge_attr, perm, num_nodes=None, num_edges=None):
    #num_nodes = maybe_num_nodes(edge_index, num_nodes)
    # keep node which in perm, keep edge degree > 0
    node_mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    node_mask[perm] = i

    # shape = [num_pins]
    nodes, edges = hyperedge_index
    # shape=[num_pins], if the node(connecte to pin) in topk, give an index, else -1
    nodes = node_mask[nodes] 
    # shepe=[num_new_edges], the left edge indeces(deg > 0)
    edge_perm = torch.nonzero(scatter_add((nodes >= 0),hyperedge_index[1], dim=0, dim_size=num_edges)).view(-1)
    edge_mask = perm.new_full((num_edges, ), -1)
    i = torch.arange(edge_perm.size(0), dtype=torch.long, device=perm.device)
    # shape=[num_edges], if the edgedeg > 0, give an index, else -1
    edge_mask[edge_perm] = i
    # shape=[num_pins], if the edge(connecte to pin) deg > 0, give an index, else -1
    edges = edge_mask[edges]
    mask = (nodes >= 0) & (edges >= 0)

    nodes, edges = nodes[mask], edges[mask]



    if hyperedge_attr is not None:
        hyperedge_attr = hyperedge_attr[mask]

    return torch.stack([nodes, edges], dim=0), hyperedge_attr


class HyperSAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=HypergraphConv,non_linearity=torch.tanh):
        super(HyperSAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, hyperedge_index, hyperedge_attr=None, batch=None):
        if batch is None:
            batch = hyperedge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,hyperedge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        hyperedge_index, hyperedge_attr = filter_inst(
            hyperedge_index, hyperedge_attr, perm, num_nodes=score.size(0), num_edges=hyperedge_index[1].max().item()+1)

        return x, hyperedge_index, hyperedge_attr, batch, perm




class HyperConv(HypergraphConv):
    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            pin_feature(LongTensor): Pin feature matrix, *i.e.*
                the sparse  matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        # modified
        #x = torch.matmul(x, self.weight)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = torch.matmul(hyperedge_attr, self.weight)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            # modified
            # x_i = x[hyperedge_index[0]]
            x_i = torch.cat((x[hyperedge_index[0]],pin_feature),dim=-1)
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        #self.flow = 'source_to_target'
        #out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
        #                     size=(num_nodes, num_edges))
        # passing message to edges
        # modified

        x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature),dim=-1)
        B_i = B.index_select(self.node_dim, hyperedge_index[1])

        out = self.message( x_j = x_j, norm_i = B_i, alpha = alpha)
        out = scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,
                           reduce=self.aggr)
        
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))

        out = torch.matmul(out.view(-1,self.in_channels), self.weight)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        

        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.in_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


class AttHyperConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True, use_self_loop=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(AttHyperConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
    
    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            pin_feature(LongTensor): Pin feature matrix, *i.e.*
                the sparse  matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        # modified
        #x = torch.matmul(x, self.weight)

        alpha = None
   
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        
        # modified
        self.flow = 'source_to_target'
        x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature),dim=-1)

        x_j = torch.matmul(x_j, self.weight)#( len(edge_index) , heads, out_channel )

        B_i = B.index_select(self.node_dim, hyperedge_index[1])

        out = self.message( x_j = x_j, norm_i = B_i, alpha = alpha)
        out = scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,
                           reduce=self.aggr)
        # there's no edge attr
        if self.use_attention:
            # dst hyperedge feature (gathered from nodes)
            x_dst = out.view(-1, self.heads, self.out_channels)[hyperedge_index[1]]
            # src node feature
            x_src = x_j.view(-1, self.heads, self.out_channels)[hyperedge_index[0]]
            alpha = torch.cat([x_src, x_dst], dim=-1) 
            alpha = (F.leaky_relu(alpha, self.negative_slope)* self.att).sum(dim=-1)
            alpha = torch.tanh(alpha)
            # alpha = (torch.cat([x_src, x_dst], dim=-1) * self.att).sum(dim=-1)
            # alpha = F.leaky_relu(alpha, self.negative_slope)
            # alpha = torch.tanh(alpha)
            # alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        
        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class AttPNAHyperConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True, use_self_loop=True,
                 aggrs_list = ['mean','max','min'],use_pna = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(AttPNAHyperConv, self).__init__(node_dim=0, **kwargs)
        in_channels = in_channels + 1
        self.use_pna = use_pna
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.aggrs = aggrs_list
        self.loop = use_self_loop

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if use_pna:
            self.pna_weight = Parameter(torch.Tensor(heads * out_channels * int(len(self.aggrs)),heads * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        if self.use_pna:
            glorot(self.pna_weight)
        zeros(self.bias)
    
    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            pin_feature(LongTensor): Pin feature matrix, *i.e.*
                the sparse  matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        # use self_loop
        if self.loop:
            loop_node = torch.arange(0, num_nodes, 1, dtype=torch.long).to(hyperedge_index.device)
            loop_edge = torch.arange(num_edges,num_nodes + num_edges, 1, dtype=torch.long).to(hyperedge_index.device)
            loop = torch.cat([loop_node.view(1,-1),loop_edge.view(1,-1)],dim=0)
            hyperedge_index = torch.cat([hyperedge_index,loop],dim=1)
            pin_feature = torch.cat([pin_feature,pin_feature.new_zeros((num_nodes,2))],dim=0)
            num_edges = num_edges + num_nodes

        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        # modified
        #x = torch.matmul(x, self.weight)
 
        alpha = None
   
        D = x.new_ones(num_nodes)
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        
        # modified
        self.flow = 'source_to_target'
        B_i = B.index_select(self.node_dim, hyperedge_index[1])
        x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature, B_i.view(-1,1)),dim=-1)

        #x_j  = x.index_select(self.node_dim, hyperedge_index[0])

        #x_j[:,2:4] = x_j[:,2:4] + pin_feature

        x_j = torch.matmul(x_j, self.weight)#( len(edge_index) , heads, out_channel )

        

        #out = self.message( x_j = x_j, norm_i = B_i, alpha = alpha)
        out = B_i.view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)

        if self.use_pna:
            outs = [scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=aggr) for aggr in self.aggrs]
            out = torch.cat(outs,dim=-1)
            out  = torch.matmul(out,self.pna_weight)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        else:
            out = scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=self.aggr)
        # there's no edge attr
        if self.use_attention:
            # dst hyperedge feature (gathered from nodes)
            x_dst = out.view(-1, self.heads, self.out_channels)[hyperedge_index[1]]
            # src node feature
            x_src = x_j.view(-1, self.heads, self.out_channels)[hyperedge_index[0]]
            alpha = torch.cat([x_src, x_dst], dim=-1) 
            alpha = (F.leaky_relu(alpha, self.negative_slope) * self.att).sum(dim=-1)
            alpha = torch.softmax(alpha,dim=-1)
            # alpha = (torch.cat([x_src, x_dst], dim=-1) * self.att).sum(dim=-1)
            # alpha = F.leaky_relu(alpha, self.negative_slope)
            # alpha = torch.tanh(alpha)
            #alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        
        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class AttPNAHyperConv3(MessagePassing):
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True, use_self_loop=True,
                 aggrs_list = ['mean','max','min'],use_pna = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(AttPNAHyperConv3, self).__init__(node_dim=0, **kwargs)
        in_channels = in_channels + 1
        self.use_pna = use_pna
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.aggrs = aggrs_list
        self.loop = use_self_loop

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if use_pna:
            self.pna_weight = Parameter(torch.Tensor(heads * out_channels * int(len(self.aggrs)),heads * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        if self.use_pna:
            glorot(self.pna_weight)
        zeros(self.bias)
    
    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        num_nodes, num_edges = x.size(0), 0
        pdb.set_trace()
        num_edges = int(hyperedge_index[1].max()) + 1
        # use self_loop
        if self.loop:
            loop_node = torch.arange(0, num_nodes, 1, dtype=torch.long).to(hyperedge_index.device)
            loop_edge = torch.arange(num_edges,num_nodes + num_edges, 1, dtype=torch.long).to(hyperedge_index.device)
            loop = torch.cat([loop_node.view(1,-1),loop_edge.view(1,-1)],dim=0)
            hyperedge_index = torch.cat([hyperedge_index,loop],dim=1)
            pin_feature = torch.cat([pin_feature,pin_feature.new_zeros((num_nodes,2))],dim=0)
            num_edges = num_edges + num_nodes

        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        # modified
        #x = torch.matmul(x, self.weight)
 
        alpha = None
   
        D = x.new_ones(num_nodes)
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        
        # modified
        self.flow = 'source_to_target'
        B_i = B.index_select(self.node_dim, hyperedge_index[1])
        x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature, B_i.view(-1,1)),dim=-1)

        #x_j  = x.index_select(self.node_dim, hyperedge_index[0])

        #x_j[:,2:4] = x_j[:,2:4] + pin_feature

        x_j = torch.matmul(x_j, self.weight)#( len(edge_index) , heads, out_channel )

        

        #out = self.message( x_j = x_j, norm_i = B_i, alpha = alpha)
        out = B_i.view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)

        if self.use_pna:
            outs = [scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=aggr) for aggr in self.aggrs]
            out = torch.cat(outs,dim=-1)
            out  = torch.matmul(out,self.pna_weight)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        else:
            out = scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=self.aggr)
        # there's no edge attr
        if self.use_attention:
            # dst hyperedge feature (gathered from nodes)
            x_dst = out.view(-1, self.heads, self.out_channels)[hyperedge_index[1]]
            # src node feature
            x_src = x_j.view(-1, self.heads, self.out_channels)[hyperedge_index[0]]
            alpha = torch.cat([x_src, x_dst], dim=-1) 
            alpha = (F.leaky_relu(alpha, self.negative_slope) * self.att).sum(dim=-1)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        
        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class AttPNAHyperConv2(MessagePassing):
    def __init__(self, in_node_channels, in_pin_channels, out_node_channels, out_pin_channels, 
                 use_attention=False, use_pin =False,
                 negative_slope=0.2, dropout=0, bias=True, use_self_loop=True,
                 aggrs_list = ['mean','max','min'], use_pna = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(AttPNAHyperConv2, self).__init__(node_dim=0, **kwargs)
        self.use_pin = use_pin
        self.use_pna = use_pna
        self.in_node_channels = in_node_channels
        self.in_pin_channels = in_pin_channels
        self.out_node_channels = out_node_channels
        self.out_pin_chennels = out_pin_channels
        self.use_attention = use_attention
        self.aggrs = aggrs_list
        self.loop = use_self_loop

        self.weight_v2e = Parameter(
                torch.Tensor(in_node_channels + in_pin_channels, out_node_channels))
        if self.use_pin:
            self.weight_e2p = Parameter(
                 torch.Tensor(2 * out_node_channels, out_pin_channels))

        
        if self.use_attention:
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight_p2v = Parameter(torch.Tensor(2 * out_node_channels, 1))

        if use_pna:
            self.pna_weight = Parameter(torch.Tensor(out_node_channels * int(len(self.aggrs)), out_node_channels))

        if bias:
            self.bias_v = Parameter(torch.Tensor(out_node_channels))
            self.bias_p = Parameter(torch.Tensor(out_pin_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_v2e)
        if self.use_attention:
            glorot(self.weight_p2v)
        if self.use_pna:
            glorot(self.pna_weight)
        if self.use_pin:
            glorot(self.weight_e2p)
        zeros(self.bias_v)
        zeros(self.bias_p)

    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            pin_feature(LongTensor): Pin feature matrix, *i.e.*
                the sparse  matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        # use self_loop
        if self.loop:
            loop_node = torch.arange(0, num_nodes, 1, dtype=torch.long).to(hyperedge_index.device)
            loop_edge = torch.arange(num_edges,num_nodes + num_edges, 1, dtype=torch.long).to(hyperedge_index.device)
            loop = torch.cat([loop_node.view(1,-1),loop_edge.view(1,-1)],dim=0)
            hyperedge_index = torch.cat([hyperedge_index,loop],dim=1)
            pin_feature = torch.cat([pin_feature,pin_feature.new_zeros((num_nodes,self.in_pin_channels))],dim=0)
            num_edges = num_edges + num_nodes

        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        # modified
        #x = torch.matmul(x, self.weight)
 
        alpha = None
   
        D = x.new_ones(num_nodes)
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        # B = scatter_add(x.new_ones(hyperedge_index.size(1)),
        #                 hyperedge_index[1], dim=0, dim_size=num_edges)
        # B = 1.0 / B
        # B[B == float("inf")] = 0
        # modified
        self.flow = 'source_to_target'
        x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature),dim=-1)

        x_j = torch.matmul(x_j, self.weight_v2e)

        if self.use_pna:
            outs = [scatter(x_j, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=aggr) for aggr in self.aggrs]
            out = torch.cat(outs,dim=-1)
            out  = torch.matmul(out,self.pna_weight)
        else:
            out = scatter(x_j, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=self.aggr)
        # there's no edge attr
        if self.use_attention:
            # dst hyperedge feature (gathered from nodes)
            x_dst = out[hyperedge_index[1]]
            # src node feature
            x_src = x_j[hyperedge_index[0]]
            alpha = torch.cat([x_src, x_dst], dim=-1)
            if self.use_pin:
                pin_feature = torch.matmul(alpha, self.weight_e2p)
            alpha = (torch.matmul(F.leaky_relu(alpha, self.negative_slope),self.weight_p2v))
            #alpha = torch.softmax(alpha,dim=-1)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))

        if self.bias_v is not None:
            out = out + self.bias_v
            pin_feature = pin_feature + self.bias_p
        if self.use_pin:
            return F.leaky_relu(out, self.negative_slope), F.leaky_relu(pin_feature[:pin_feature.shape[0] - x.shape[0]],self.negative_slope)
        return F.leaky_relu(out,self.negative_slope)


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        F = self.out_node_channels

        out = norm_i.view(-1, 1) * x_j.view(-1, F)

        if alpha is not None:
            out = alpha.view(-1,1) * out

        return out
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_node_channels,
                                   self.out_node_channels)

class AttHypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(AttHypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = torch.matmul(x, self.weight)

        alpha = None

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))

        # there's no edge attr
        if self.use_attention:
            # dst hyperedge feature (gathered from nodes)
            x_dst = out.view(-1, self.heads, self.out_channels)[hyperedge_index[1]]
            # src node feature
            x_src = x.view(-1, self.heads, self.out_channels)[hyperedge_index[0]]
            alpha = torch.cat([x_src, x_dst], dim=-1) 
            alpha = (F.leaky_relu(alpha, self.negative_slope) * self.att).sum(dim=-1)
            alpha = torch.tanh(alpha)
            # alpha = (torch.cat([x_src, x_dst], dim=-1) * self.att).sum(dim=-1)
            # alpha = F.leaky_relu(alpha, self.negative_slope)
            # alpha = torch.tanh(alpha)
            #alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))


        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class PNAHyperedgeConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 aggrs_list = ['max','min'],use_pna = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PNAHyperedgeConv, self).__init__(node_dim=0, **kwargs)
        self.use_pna = use_pna
        self.in_channels = in_channels - 5
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.aggrs = aggrs_list


        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if use_pna:
            self.pna_weight = Parameter(torch.Tensor(self.in_channels * int(len(self.aggrs)),heads * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        if self.use_pna:
            glorot(self.pna_weight)
        zeros(self.bias)
    
    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            pin_feature(LongTensor): Pin feature matrix, *i.e.*
                the sparse  matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        # modified
        #x = torch.matmul(x, self.weight)

        alpha = None
   
        # D = scatter_add(hyperedge_weight[hyperedge_index[1]],
        #                 hyperedge_index[0], dim=0, dim_size=num_nodes)
        # D = 1.0 / D
        # D[D == float("inf")] = 0

        # B = scatter_add(x.new_ones(hyperedge_index.size(1)),
        #                 hyperedge_index[1], dim=0, dim_size=num_edges)
        # B = 1.0 / B
        # B[B == float("inf")] = 0

        
        # modified
        self.flow = 'source_to_target'
        #x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature),dim=-1)

        #( len(edge_index) , heads, out_channel )
        #print(x[0])
        x_j = x.index_select(self.node_dim, hyperedge_index[0])
        #print(x_j[0])
        x_j = x_j + pin_feature
        #print(pin_feature[0])


        #B_i = B.index_select(self.node_dim, hyperedge_index[1])

        #out = self.message( x_j = x_j, norm_i = B_i, alpha = alpha)


        if self.use_pna:
            outs = [scatter(x_j, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=aggr) for aggr in self.aggrs]
            out = torch.cat(outs,dim=-1)
            #pdb.set_trace()
            out  = torch.matmul(out,self.pna_weight)
            #out = F.dropout(out, p=self.dropout, training=self.training)
            #x_j = torch.matmul(x_j, self.weight)
        #else:
        #    out = scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=self.aggr)
        # there's no edge attr
        # if self.use_attention:
        #     # dst hyperedge feature (gathered from nodes)
        #     x_dst = out.view(-1, self.heads, self.out_channels)[hyperedge_index[1]]
        #     # src node feature
        #     x_src = x_j.view(-1, self.heads, self.out_channels)[hyperedge_index[0]]
        #     alpha = torch.cat([x_src, x_dst], dim=-1) 
        #     alpha = (F.leaky_relu(alpha, self.negative_slope)* self.att).sum(dim=-1)
        #     alpha = torch.tanh(alpha)
        #     # alpha = (torch.cat([x_src, x_dst], dim=-1) * self.att).sum(dim=-1)
        #     # alpha = F.leaky_relu(alpha, self.negative_slope)
        #     # alpha = torch.tanh(alpha)
        #     alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # self.flow = 'target_to_source'
        # out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
        #                      size=(num_edges, num_nodes))

        out = out.view(-1, self.heads * self.out_channels)

        if self.bias is not None:
            out = out + self.bias
        
        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# gather from nodes
class V2EConv(MessagePassing):
    def __init__(self, in_channels, out_channels, 
                 negative_slope=0.2, bias=True,
                 aggrs_list = ['mean','max','min','add'],
                 use_pna = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(V2EConv, self).__init__(node_dim=0, **kwargs)
        in_channels = in_channels
        self.use_pna = use_pna
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggrs = aggrs_list
        self.negative_slope = negative_slope

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if use_pna:
            self.pna_weight = Parameter(torch.Tensor(out_channels * int(len(self.aggrs)), out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_pna:
            glorot(self.pna_weight)
        zeros(self.bias)
    
    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
   
        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0
        # modified
        self.flow = 'source_to_target'
        x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature),dim=-1)
        #x_j  = x.index_select(self.node_dim, hyperedge_index[0])
        #x_j[:,2:4] = x_j[:,2:4] + pin_feature
        x_j = torch.matmul(x_j, self.weight)#( len(edge_index) , heads, out_channel )
        B_i = B.index_select(self.node_dim, hyperedge_index[1])
        #out = self.message( x_j = x_j, norm_i = B_i, alpha = alpha)
        out = B_i.view(-1, 1) * x_j.view(-1, self.out_channels)
        if self.use_pna:
            outs = [scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=aggr) for aggr in self.aggrs]
            out = torch.cat(outs,dim=-1)
            out  = torch.matmul(out,self.pna_weight)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        else:
            out = scatter(out, hyperedge_index[1], dim=self.node_dim, dim_size=num_edges,reduce=self.aggr)


        if self.bias is not None:
            out = out + self.bias
        
        return x_j, F.leaky_relu(out,negative_slope=self.negative_slope,inplace=True)


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# aggregate from edges
class E2VConv(MessagePassing):
    def __init__(self, in_channels, out_channels, 
                 use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 aggrs_list = ['mean','max','min', 'add'], use_pna = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(E2VConv, self).__init__(node_dim=0, **kwargs)
        in_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.aggrs = aggrs_list
        self.negative_slope = negative_slope

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            #self.weight = Parameter(
            #    torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            #self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
    
    def forward(self, x_j: Tensor, 
                hyperedge_index: Tensor,
                hyperedge_attr: Tensor = None,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:
        num_nodes, num_edges = int(hyperedge_index[0].max()) + 1, 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x_j.new_ones(num_edges)
        # modified
        #x = torch.matmul(x, self.weight)

        alpha = None
   
        #D = x_j.new_ones(num_nodes)
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0
        # there's no edge attr
        if self.use_attention:
            # dst hyperedge feature (gathered from nodes)
            x_dst = hyperedge_attr.view(-1, self.heads, self.out_channels)[hyperedge_index[1]]
            # src node feature
            x_src = x_j.view(-1, self.heads, self.out_channels)[hyperedge_index[0]]
            alpha = torch.cat([x_src, x_dst], dim=-1) 
            alpha = (F.leaky_relu(alpha, self.negative_slope) * self.att).sum(dim=-1)
            alpha = 2 * torch.tanh(alpha)
            # alpha = (torch.cat([x_src, x_dst], dim=-1) * self.att).sum(dim=-1)
            # alpha = F.leaky_relu(alpha, self.negative_slope)
            # alpha = torch.tanh(alpha)
            #alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=hyperedge_attr, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        
        return F.leaky_relu(out,negative_slope=self.negative_slope,inplace=True)


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class LAFLayer(torch.nn.Module):
    def __init__(self, units = 4, eps=1e-7,**kwargs):
        super(LAFLayer, self).__init__()
        self.units = units
        self.eps = eps
        self.weights = Parameter(torch.Tensor(12,self.units))
        self.e = Parameter(torch.tensor([1,-1,1,-1]).float(), requires_grad=False)
        self.num_idx = Parameter(torch.tensor([1,1,0,0]).float(), requires_grad=False)
        self.den_idx = Parameter(torch.tensor([0,0,1,1]).float(), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weights)

    def forward(self, data, index, dim=0, **kwargs):
        r"""
        @Param
            data: node_feature
            index: egde_index
        """
        eps = self.eps
        sup = 1.0 - eps 
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1,1,-1)        

        exps = (1. - e)/2. + x*e 
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1,1,4,-1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)
        
        multiplier = 2.0*torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier*eps, den)

        res = num / den
        return res


# gather from nodes
class V2ELAFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, 
                 negative_slope=0.2, 
                 bias=True,
                 units = 4,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(V2ELAFConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.units = units 
        self.LAF = LAFLayer(units)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels // units))
        self.laf_weight = Parameter(torch.Tensor(out_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.laf_weight)
        zeros(self.bias)
    
    def forward(self, x: Tensor, 
                hyperedge_index: Tensor,
                pin_feature: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
   
        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0
        # modified
        self.flow = 'source_to_target'
        x_j = torch.cat((x.index_select(self.node_dim, hyperedge_index[0]),pin_feature),dim=-1)

        x_j = torch.matmul(x_j, self.weight)
        B_i = B.index_select(self.node_dim, hyperedge_index[1])
        out = B_i.view(-1, 1) * x_j.view(-1, self.out_channels // self.units)

        out = self.LAF(out, hyperedge_index[1]).view(-1,self.out_channels)
        #pdb.set_trace()
        out = torch.matmul(out, self.laf_weight)

        if self.bias is not None:
            out = out + self.bias
        
        return x_j, F.leaky_relu(out,negative_slope=self.negative_slope,inplace=True)


    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)