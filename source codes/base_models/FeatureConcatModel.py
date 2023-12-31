import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader
from torch_geometric.data import Dataset as TGDataset, Data as TGData
from torch_geometric.data.dataloader import DataLoader as TGDataLoader
from torch_geometric import transforms as T
from torch_geometric.nn import GCNConv,GATConv,SAGEConv, GATConv,ChebConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
LAYERS = {
    GCNConv:"GCNConv",
    GATConv: "GATConv",
    SAGEConv:"SAGEConv",
    ChebConv:"ChebConv"
}
from pre_process.GraphPreprocessing import *
from base_models.PointNet import *
from base_models.SelfAttentionGraphPooling import * 

class FeatureConcatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        MAINargs = {
        "SAGPoolNet_dataset_features":10,
        "out_channels":1,
        "is_hierarchical":True,
        "use_w_for_concat":True,
        "pooling_ratio":0.25,
        "p_dropout":0.25,
        "Conv":GATConv,
        "heads":6,
        "concat":False,
        "send_feature":True,
        "hidden_features":128
        }
        
        self.graph_pool_model = SAGPoolNet(**MAINargs)
        
        self.pointnet_model = PointNet(send_feature=True,input_dim=10)
        self.fc_point_net = nn.Linear(1024, 64)
        self.bn_point_net = nn.BatchNorm1d(64)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        
    def forward(self, data):
        data_graph = ConvertBatchToGraph(data)
        data_graph = data_graph.to("cuda")
        out_graph = self.graph_pool_model(data_graph)
        
        data_pointnet = data['graph_features'].float().transpose(1,2).to("cuda")
        out_pointnet,m3,m64 = self.pointnet_model(data_pointnet)
        out_pointnet = nn.MaxPool1d(out_pointnet.size(-1))(out_pointnet)
        out_pointnet = nn.Flatten(1)(out_pointnet)
        out_pointnet = F.relu(self.bn_point_net(self.fc_point_net(out_pointnet)))
        
        final_out = torch.cat([out_graph,out_pointnet],dim=1)
        final_out = F.relu(self.bn1(self.fc1(final_out)))
        final_out = F.relu(self.bn2(self.dropout(self.fc2(final_out))))
        final_out = self.fc3(final_out)
        return self.logsoftmax(final_out), m3, m64