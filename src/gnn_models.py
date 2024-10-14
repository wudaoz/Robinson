import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the GCN model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.
        """
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
# #以下是胶囊图卷积网络代码
# class GCN(torch.nn.Module):
#     def __init__(
#         self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
#     ):
#         """
#         This constructor method initializes the GapsGCN model

#         Arguments:
#         nfeat: (int) - Number of input features
#         nhid: (int) - Number of hidden features in the hidden layers of the network
#         nclass: (int) - Number of output classes
#         dropout: (float) - Dropout probability
#         NumLayers: (int) - Number of GapsGCN layers in the network.
#         """
#         super(GCN, self).__init__() #这里把GapsGCN改为了GCN

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
#         for _ in range(NumLayers - 2):
#             self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
#         self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True)) #nclass改成了nhid
#         self.linear = Linear(nhid, nclass)

#         self.dropout = dropout
#         self.num_capsules = nclass  # 定义胶囊的数量为输出类别数

#     def reset_parameters(self) -> None:
#         """
#         This function is available to cater to weight initialization requirements as necessary.
#         """
#         for conv in self.convs:
#             conv.reset_parameters()
#         self.linear.reset_parameters()
#         return None

#     def dynamic_routing(self, x, num_iterations=3):
#         """
#         实现动态路由机制
#         x: (torch.Tensor) - 输入特征
#         num_iterations: (int) - 动态路由迭代次数
#         """
#         batch_size = x.size(0)
#         b_ij = torch.zeros(batch_size, self.num_capsules, 1).to(x.device)
#         for _ in range(num_iterations):
#             c_ij = F.softmax(b_ij, dim=1)
#             s_j = (c_ij * x.unsqueeze(1)).sum(dim=0)
#             v_j = self.squash(s_j)
#             b_ij = b_ij + (x.unsqueeze(1) * v_j).sum(dim=-1, keepdim=True)
#         return v_j

#     @staticmethod
#     def squash(s_j):
#         """
#         实现squash函数
#         s_j: (torch.Tensor) - 输入向量
#         """
#         s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
#         v_j = (s_j_norm / (1 + s_j_norm**2)) * s_j
#         return v_j

#     def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
#         for conv in self.convs[:-1]:
#             x = conv(x, adj_t)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
        
#         # 动态路由机制
#         x_caps = self.dynamic_routing(x)
        
#         # 将胶囊输出转换为适合分类的形式
#         x_caps = x_caps.view(x_caps.size(0), -1)
#         x = self.linear(x_caps)  # 映射到两个类别
#         return torch.log_softmax(x, dim=-1)  # 返回对数概率



class GCN_products(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the GCN_products model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.
        """

        super(GCN_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=False))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=False))
        self.convs.append(GCNConv(nhid, nclass, normalize=False))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE_products(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network
        """
        super(SAGE_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nclass))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


# +
class GCN_arxiv(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network
        """
        super(GCN_arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GCNConv(nhid, nclass, cached=True))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
