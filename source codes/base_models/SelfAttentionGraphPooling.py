import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader



from torch_geometric.data import Dataset as TGDataset, Data as TGData
from torch_geometric.loader import DataLoader as TGDataLoader
from torchvision import transforms, utils
from torch_geometric.utils.convert import from_networkx
from torch_geometric import transforms as T
from torch_geometric.nn import GCNConv,Linear,GATConv,GATv2Conv,SAGEConv, GATConv,ChebConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.pool.topk_pool import topk,filter_adj

LAYERS = {
    GCNConv:"GCNConv",
    GATConv: "GATConv",
    SAGEConv:"SAGEConv",
    ChebConv:"ChebConv"
}



class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.2,non_linearity=torch.tanh,**karg):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = karg["Conv"](in_channels,**karg)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x,edge_index).squeeze()
        perm = topk(score, self.ratio, batch)

        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
    
class SAGPoolNet(torch.nn.Module):
    def __init__(self,SAGPoolNet_dataset_features=10,is_hierarchical=True,pooling_ratio=0.2,p_dropout=0.2,hidden_features=128,send_feature=False,use_w_for_concat=False,**karg):
        super(SAGPoolNet, self).__init__()
        from torch.nn.init import xavier_uniform_,zeros_
        self.num_features = SAGPoolNet_dataset_features
        self.hidden_features = hidden_features
        self.num_classes = 10
        self.pooling_ratio = pooling_ratio
        self.p = p_dropout

        self.send_feature = send_feature
        self.is_hierarchical = is_hierarchical
        self.use_w_for_concat = use_w_for_concat

        if self.use_w_for_concat:
            W = torch.Tensor(3,1)
            W = nn.Parameter(W)
            self._att = W
            xavier_uniform_(self._att)

        if is_hierarchical:
            self.lin1 = torch.nn.Linear(self.hidden_features*2, self.hidden_features)
            self.conv1 = GCNConv(self.num_features, self.hidden_features)
            self.pool1 = SAGPool(self.hidden_features, ratio=self.pooling_ratio,**karg)
            self.conv2 = GCNConv(self.hidden_features, self.hidden_features)
            self.pool2 = SAGPool(self.hidden_features, ratio=self.pooling_ratio,**karg)
            self.conv3 = GCNConv(self.hidden_features, self.hidden_features)
            self.pool3 = SAGPool(self.hidden_features, ratio=self.pooling_ratio,**karg)
        else:
            self.conv1 = GCNConv(self.num_features, self.hidden_features)
            self.conv2 = GCNConv(self.hidden_features, self.hidden_features)
            self.conv3 = GCNConv(self.hidden_features, self.hidden_features)
            self.pool = SAGPool(self.hidden_features*3, ratio=self.pooling_ratio,**karg)
            self.lin1 = torch.nn.Linear(self.hidden_features*2*3, self.hidden_features)

        self.lin2 = torch.nn.Linear(self.hidden_features, self.hidden_features//2)
        if not self.send_feature:
            self.lin3 = torch.nn.Linear(self.hidden_features//2, self. num_classes)

    def forward(self, data):
        if self.is_hierarchical:
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            if self.use_w_for_concat:
                x = self._att[0][0]*x1 + self._att[1][0]*x2 + self._att[2][0]*x3
            else:
                x = x1 + x2 + x3

        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x1 = F.relu(self.conv1(x, edge_index))
            x2 = F.relu(self.conv2(x1, edge_index))
            x3 = F.relu(self.conv3(x2, edge_index))

            if self.use_w_for_concat:
                x1 = self._att[0][0] * x1
                x2 = self._att[1][0] * x2
                x3 = self._att[2][0] * x3

            x = torch.concat([x1,x2,x3],dim=1)
            x, edge_index, _, batch, _ = self.pool(x, edge_index, None, batch)
            x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.lin2(x))
        
        if self.send_feature:
            return x
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
    