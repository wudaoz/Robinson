import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import MessagePassing

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


class AMGapsGNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers, K=5):
        super(AMGapsGNN, self).__init__()
        
        self.K = K  # 图记忆模块中的记忆槽数量
        self.convs = torch.nn.ModuleList()
        self.convs.append(MessagePassing(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(MessagePassing(nhid, nhid))
        self.convs.append(MessagePassing(nhid, nclass))

        self.memory = torch.nn.Parameter(torch.randn(K, nhid))
        self.dropout = dropout

    def adaptive_filtering(self, x_i, x_j):
        return torch.sigmoid(torch.cat([x_i, x_j], dim=-1))

    def graph_memory(self, x):
        attention = F.softmax(x @ self.memory.T, dim=-1)
        return torch.einsum('nk,kd->nd', attention, self.memory)

    def dynamic_routing(self, x, adj_t, iterations=3):
        for _ in range(iterations):
            x = torch.einsum('ij,jd->id', adj_t, x)
            x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = self.adaptive_filtering(x, x)
            x = self.graph_memory(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, adj_t)
        x = self.dynamic_routing(x, adj_t)
        return torch.log_softmax(x, dim=-1)