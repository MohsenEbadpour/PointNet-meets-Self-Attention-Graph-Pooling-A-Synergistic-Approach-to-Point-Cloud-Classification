import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# from torch_geometric.loader import DataLoader
# from torch.utils.data import Dataset as TDataset, DataLoader 
from torch_geometric.data import DataLoader
# from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import plotly.graph_objects as go
from tqdm import tqdm
# from torch_geometric.data import Dataset as TGDataset, Data as TGData
# from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric import transforms as T
from torch_geometric.nn import GCNConv,Linear,GATConv,GATv2Conv,SAGEConv, GATConv,ChebConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from pre_process.GraphPreprocessing import *
from base_models.SelfAttentionGraphPooling import *

from visualization.ReportVisualization import *


DATASET_NAME="PROTEINS_full"

graph_dataset , num_classes = load_graph(DATASET_NAME)
print(num_classes)
TrainSet,ValidationSet,TestSet = GetSets(graph_dataset,0.8,0.2)
BatchSize = 32


TrainLoader = DataLoader(TrainSet, batch_size=BatchSize, shuffle=True)
ValidationLoader = DataLoader(ValidationSet,batch_size=BatchSize,shuffle=False)
TestLoader = DataLoader(TestSet,batch_size=BatchSize,shuffle=False)


def TestPerformance(model,loader):
    with torch.no_grad():
        model.eval()
        correct = 0.
        loss = 0.
        for data in loader:
            data = ConvertBatchToGraph(data)
            data = data.to("cpu")
            model = model.to("cpu")
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.cross_entropy(out,data.y).item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


def Train(model,TrainLoader,ValidationLoader,epoch:int,lr=0.01,weight_decay=5e-4,show=True,name="Self-Attention Graph Pooling"):
    device = "cpu"
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    loss_train = []
    acc_train = []

    loss_val = []
    acc_val = []

    acc_test = []
    min_loss = 1e10
    patience = 0
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_mb = round(size_all_mb,3)
    print("Model Size: {0} MB".format(size_all_mb))
    for ite in range(epoch):
        model.train()
        for i, data in enumerate(TrainLoader):
            data = ConvertBatchToGraph(data)
            opt.zero_grad()
            data = data.to("cpu")
            model = model.to("cpu")
            out = model(data)
            # print(len(out))
            # print(len(data.y))

            loss = F.cross_entropy(out, data.y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        val_acc,val_loss = TestPerformance(model,ValidationLoader)
        train_acc,train_loss = TestPerformance(model,TrainLoader)

        acc_val.append(val_acc)
        loss_val.append(val_loss)

        acc_train.append(train_acc)
        loss_train.append(train_loss)


        print("Epoch: {0} | Train Loss: {1} | Train Acc: {2} | Val Loss: {3} | Val Acc: {4}".format(ite,train_loss,train_acc,val_loss,val_acc,size_all_mb))

    test_acc = max(acc_val)
    if show:
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize']= (21,5)
        h,w = 1,2
        plt.subplot(h,w,1)
        plt.plot(loss_train,label="Train loss")
        plt.plot(loss_val,label="Validation loss")
        plt.title("Loss Report | {0} | ModelSize: {1} MB".format(name,size_all_mb))
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        #plt.show()

        plt.subplot(h,w,2)
        plt.plot(acc_train,label="Train Accuracy")
        plt.plot(acc_val,label="Validation Accuracy")
        plt.title("Accuracy Report | Test Accuracy: {0}%".format(round(test_acc*100,2)))
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plt.savefig("../results/self-attention-graph-pooling/graph_dataset/{0},{1}_with_label.png".format(name,DATASET_NAME))
        plt.show()
        plt.clf()

    return round(test_acc*100,2),model


MAINargs = {
    "SAGPoolNet_dataset_features":8,
    "out_channels":1,
    "is_hierarchical":True,
    "use_w_for_concat":True,
    "pooling_ratio":0.25,
    "p_dropout":0.25,
    "Conv":GATConv,
    "heads":6,
    "concat":False,
    "send_feature":False,
    "num_classes":num_classes
}


model = SAGPoolNet(**MAINargs)
acc,model = Train(model,TrainLoader=TrainLoader,ValidationLoader=ValidationLoader ,
            epoch=60,lr=0.01,weight_decay=0.0005,show=True,name="Self-Attention Graph Pooling")

