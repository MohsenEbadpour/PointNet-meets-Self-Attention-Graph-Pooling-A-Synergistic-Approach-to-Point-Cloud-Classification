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

from PointNet import *


def TestPerfomanceCustom(model,loader):
    with torch.no_grad():
        model.eval()
        correct = 0.
        loss = 0.
        model = model.to("cuda")
        for data in loader:
            outputs, m3x3, m64x64 = model(data)
            labels = data['category'].to("cuda")
            pred = outputs.max(dim=1)[1]
            correct += pred.eq(labels).sum().item()

            loss += PointNetLoss(outputs, labels, m3x3, m64x64,defualt_dim=10).item()

    return correct / len(loader.dataset),loss / len(loader.dataset)


def TrainCustom(model, train_loader, val_loader,lr=0.01,weight_decay=0.0005, epochs=30, name="PointNet"):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_mb = round(size_all_mb,3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_train = []
    acc_train = []

    loss_val = []
    acc_val = []
    for epoch in range(epochs):
        model.train()

        for i, data in tqdm(enumerate(train_loader, 0)):
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(data)
            labels = data['category'].to(device)
            loss = PointNetLoss(outputs, labels, m3x3, m64x64,defualt_dim=10)
            loss.backward()
            optimizer.step()


        val_acc,val_loss = TestPerfomanceCustom(model,val_loader)
        train_acc,train_loss = TestPerfomanceCustom(model,train_loader)

        acc_val.append(val_acc)
        loss_val.append(val_loss)

        acc_train.append(train_acc)
        loss_train.append(train_loss)

        print("Epoch: {0} | Train Loss: {1} | Train Acc: {2} | Val Loss: {3} | Val Acc: {4}".format(epoch,train_loss,train_acc,val_loss,val_acc,size_all_mb))

    test_acc = max(acc_val)


    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize']= (21,5)
    h,w = 1,2
    plt.subplot(h,w,1)
    plt.plot(loss_train,label="Train loss")
    plt.plot(loss_val,label="Validation loss")
    plt.title("Loss Report | {0} | ModelSize: {1} MB".format(name,size_all_mb))
    plt.xlabel("Epoch")
    plt.ylabel("NLLLoss")
    plt.legend()
    #plt.show()

    plt.subplot(h,w,2)
    plt.plot(acc_train,label="Train Accuracy")
    plt.plot(acc_val,label="Validation Accuracy")
    plt.title("Accuracy Report | Test Accuracy: {0}%".format(round(test_acc*100,2)))
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./{0}.png".format(name))
    plt.show()
    plt.clf()

    return round(test_acc*100,2),model