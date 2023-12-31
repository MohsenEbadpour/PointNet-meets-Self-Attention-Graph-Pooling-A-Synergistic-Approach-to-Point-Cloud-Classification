import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader
from torch.utils.data import random_split


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import os
import plotly.graph_objects as go
import math
import random
import threading
from tqdm import tqdm
import pickle

import torch_geometric
from torch_geometric.data import Dataset as TGDataset, Data as TGData
from torch_geometric.data.dataloader import DataLoader as TGDataLoader
from torchvision import transforms, utils
from torch_geometric.utils.convert import from_networkx
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


from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from sklearn.metrics import confusion_matrix,accuracy_score
import scipy.spatial.distance
import networkx as nx

from base_models.PointNet import *
from base_models.SelfAttentionGraphPooling import * 


class PointNetBasedGraphPoolingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        MAINargs = {
        "SAGPoolNet_dataset_features":32,
        "out_channels":1,
        "is_hierarchical":True,
        "use_w_for_concat":True,
        "pooling_ratio":0.25,
        "p_dropout":0.25,
        "Conv":GATConv,
        "heads":6,
        "concat":False,
        "send_feature":False,
        "hidden_features":256
        }
        
        self.graph_pool_model = SAGPoolNet(**MAINargs)
        
        self.pointnet_model = PointNet(send_feature=True,input_dim=10)
        self.conv = nn.Conv1d(1024,32,1)
        self.bn = nn.BatchNorm1d(32)
        
        
    def forward(self, data):
        data_pointnet = data['graph_features'].float().transpose(1,2).to("cuda")
        out_pointnet,m3,m64 = self.pointnet_model(data_pointnet)
        
        out_pointnet = self.bn(self.conv(out_pointnet))
        graph_batch = self.get_graph_structure(out_pointnet,data) 
        graph_batch = graph_batch.to("cuda")
        graph_out = self.graph_pool_model(graph_batch)
        
        return graph_out,m3,m64
    
    def get_graph_structure(self,pointnet_out,batch):
        list_of_graphs = []
        for index in range(len(batch["edge_list"])):
            edge_index = batch["edge_list"][index].T
            x = pointnet_out[index].float().transpose(0,1)
            y = batch["category"][index]
            tgdata = TGData(x=x,y=y,edge_index=edge_index)
            list_of_graphs.append(tgdata)
            
        return torch_geometric.data.Batch.from_data_list(list_of_graphs)