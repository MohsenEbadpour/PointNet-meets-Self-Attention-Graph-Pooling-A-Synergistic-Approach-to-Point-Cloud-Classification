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


from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from sklearn.metrics import confusion_matrix,accuracy_score
import scipy.spatial.distance
import networkx as nx

from CloudPointsPreprocessing import * 
from FeatureConcatModel import * 
from PointNet import *
from PointNetBasedGraphPoolingModel import *
from ReportVisualization import * 
from SelfAttentionGraphPooling import * 

class PointCloudGraph(TGDataset):
    def __init__(self, point_cloud_dataset ,root="./", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.point_cloud_dataset = point_cloud_dataset         
        
    def len(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, idx):
        sample = self.point_cloud_dataset[idx]
        edge_index = sample["edge_list"].T
        x = sample["graph_features"].float()
        y = sample["category"]
        tgdata = TGData(x=x,y=y,edge_index=edge_index)
        return tgdata
    
    def __len__(self):
        return len(self.point_cloud_dataset)
    
    
def get_graph_features(point_cloud,N=6):
    connection_matrix = kneighbors_graph(point_cloud,N,mode='distance').toarray()
    graph = nx.DiGraph()
    X,Y,Z = {},{},{}
    for node in range(connection_matrix.shape[0]):
        for neighbor in range(connection_matrix.shape[1]):
            if neighbor == node:
                graph.add_edge(node,neighbor,weight=0)
                continue
            if connection_matrix[node][neighbor] == 0 :
                continue
            graph.add_edge(neighbor,node)
            graph[neighbor][node]["weight"] = connection_matrix[node][neighbor]
        X[node] = np.float64(point_cloud[node][0])
        Y[node] = np.float64(point_cloud[node][1])
        Z[node] = np.float64(point_cloud[node][2])

    features = [X,Y,Z]
    features_name = ["X","Y","Z"]

    betweenness_centrality = nx.betweenness_centrality(graph,weight="weight")
    features.append(betweenness_centrality)
    features_name.append("betweenness_centrality")

    katz_centrality = nx.katz_centrality(graph,weight="weight")
    features.append(katz_centrality)
    features_name.append("katz_centrality")

    closeness_centrality = nx.closeness_centrality(graph,distance="weight",)
    features.append(closeness_centrality)
    features_name.append("closeness_centrality")

    eigenvector_centrality = nx.eigenvector_centrality(graph,weight="weight",max_iter=100,tol=1e-3)
    features.append(eigenvector_centrality)
    features_name.append("eigenvector_centrality")

    harmonic_centrality = nx.harmonic_centrality(graph,distance="weight",)
    features.append(harmonic_centrality)
    features_name.append("harmonic_centrality")

    load_centrality = nx.load_centrality(graph,weight="weight")
    features.append(load_centrality)
    features_name.append("load_centrality")

    pagerank = nx.pagerank(graph,weight='weight')
    features.append(pagerank)
    features_name.append("pagerank")

    for idx in range(len(features)):
        nx.set_node_attributes(graph,features[idx],features_name[idx])

    nodes = nx.nodes(graph)
    features = []
    for node_indx in range(len(nodes)): 
        features.append(np.array(list(nodes[node_indx].values())))
    features = np.array(features)
    
    features[:,3:] = features[:,3:] - np.mean(features[:,3:], axis=0)
    features[:,3:] /= np.max(np.linalg.norm(features[:,3:], axis=1))
    
    edge_list = np.array(nx.edges(graph))
    
    return torch.from_numpy(features),torch.from_numpy(edge_list), graph


def ConvertBatchToGraph(batch):
    list_of_graphs = []
    for index in range(len(batch["edge_list"])):
        edge_index = batch["edge_list"][index].T
        x = batch["graph_features"][index].float()
        y = batch["category"][index]
        tgdata = TGData(x=x,y=y,edge_index=edge_index)
        list_of_graphs.append(tgdata)
    return torch_geometric.data.Batch.from_data_list(list_of_graphs)
    


def GetSets(dataset,train=0.99,valid=0.01):
    train_ratio = int(len(dataset)*train)
    validation_ratio = int(len(dataset)*valid)
    training_set,validation_set,test_set = random_split(dataset,[train_ratio , validation_ratio,len(dataset) - (train_ratio + validation_ratio)])
    return training_set,validation_set,test_set
