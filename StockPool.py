import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn

# define base 3-layer GCN model
class GNN(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, normalize=True):
    super(GNN, self).__init__()
    self.convs = torch.nn.ModuleList()
    self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
    self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
    self.convs.append(GCNConv(hidden_channels, out_channels, normalize))

  def forward(self, x, adj, mask=None):
    num_nodes, in_channels = x.size()[-2:]
    for step in range(len(self.convs)):
      # x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))
      x = F.relu(self.convs[step](x, adj, mask))
    return x

# define stock pooling operation
def stock_pool(x, adj, s,  mask=None):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    batch_size, num_nodes, _ = x.size()
    s = torch.softmax(s, dim=-1)  # obtain matrix S, i.e. the assignment matrix
    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask
    out = torch.matmul(s.transpose(1, 2), x)  # output feature matrix X'
    # output adjacency matrix A'
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    return out, out_adj, None, None

# Define Stock Pooling Class
class StockPool(torch.nn.Module):
  def __init__(self, hidden_nodes, num_features, num_nodes, S_subindustry, S_industry):
    super(StockPool, self).__init__()
    
    # define S_1 and S_2
    self.S_1 = S_subindustry
    self.S_2 = S_industry

    # define pooling layer 1
    self.gnn1_embed = GNN(num_features, hidden_nodes, hidden_nodes)

    # define pooling layer 2
    self.gnn2_embed = GNN(hidden_nodes, hidden_nodes, hidden_nodes)

    # define pooling layer 3
    self.gnn3_embed = GNN(hidden_nodes, hidden_nodes, hidden_nodes)

    # define linear layers
    self.linear_layers = nn.Sequential(
        nn.Dropout(),
        torch.nn.Linear(hidden_nodes, num_nodes * 2),
        nn.ReLU(),
        torch.nn.Linear(num_nodes * 2, num_nodes)
    )

    # define sigmoid function for binary classification
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x, edge_index, adj, edge_weight=None, mask=None):
    # apply pooling layer 1
    s = self.S_1
    x = self.gnn1_embed(x, edge_index, mask)
    x, adj, _, _ = stock_pool(x, adj, s, mask)
    x = x[0]
    edge_index = adj[0].nonzero().t().contiguous()

    # apply pooling layer 2
    s = self.S_2
    x = self.gnn2_embed(x, edge_index, mask)
    x, adj, _, _ = stock_pool(x, adj, s, mask)
    x = x[0]
    edge_index = adj[0].nonzero().t().contiguous()

    # apply final convolution
    x = self.gnn3_embed(x, edge_index, mask)

    # apply linear function (leave for regression, otherwise add softmax)
    x = x.mean(dim=0)
    x = self.linear_layers(x)

    # return x # for regression
    return self.sigmoid(x)  # for binary classification
