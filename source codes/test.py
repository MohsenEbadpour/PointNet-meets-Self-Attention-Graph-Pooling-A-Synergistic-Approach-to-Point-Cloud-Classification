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
import plotly.graph_objects as go
from tqdm import tqdm
from torch_geometric.data import Dataset as TGDataset, Data as TGData
# from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.data import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric import transforms as T
from torch_geometric.nn import GCNConv,Linear,GATConv,GATv2Conv,SAGEConv, GATConv,ChebConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from pre_process.CloudPointsPreprocessing import *
from pre_process.PointCloudGraphPreprocessing import *

from base_models.PointNet import *
from base_models.SelfAttentionGraphPooling import *

from visualization.ReportVisualization import *


path_global = Path("../datasets/pointcloud/raw/ModelNet10")

dataset_pointcloud_test = PointCloudData(path_global, valid=True, folder='test',force_to_cal=False)
dataset_pointcloud_train = PointCloudData(path_global, force_to_cal=False)



dataset_graph_test = PointCloudGraph(dataset_pointcloud_test)

dataset_graph_train = PointCloudGraph(dataset_pointcloud_train)


TrainSet,ValidationSet,TestSet = GetSets(dataset_graph_train,0.99,0.01)

BatchSize = 64
TrainLoader = DataLoader(TrainSet, batch_size=BatchSize, shuffle=True)
ValidationLoader = DataLoader(ValidationSet,batch_size=BatchSize,shuffle=False)
TestLoader = DataLoader(dataset_graph_test,batch_size=BatchSize,shuffle=False)


        

def save_checkpoint(path:str,epoch:int,model,optimizer)->None:
    """function for save checkpoint of model

    Args:
        path (str): path for save checkpoint
        epoch (int): number of epoch
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
def load_checkpoint(path:str,model,optimizer)->None:
    """function for load checkpoint of model

    Args:
        path (str): path for load checkpoint
    """


    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer





def TestPerformance(model,loader):
    with torch.no_grad():
        model.eval()
        correct = 0.
        loss = 0.
        for data in loader:
            # data = ConvertBatchToGraph(data)
            data = data.to("cuda")
            model = model.to("cuda")
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.cross_entropy(out,data.y).item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


def Train(model,TrainLoader,ValidationLoader,TestLoader,epoch:int,lr=0.01,weight_decay=5e-4,show=True,name="Self-Attention Graph Pooling"):
    device = "cuda"
    # print(weight_decay)


    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model,opt = load_checkpoint("../checkpoints/pointcloud/{1}_{0}.pt",model,opt)
    model.train()
    loss_train = []
    acc_train = []

    # loss_val = []
    # acc_val = []

    acc_test = []
    loss_test=[]

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
            # print(data)
            # data = ConvertBatchToGraph(data)
            opt.zero_grad()

            data = data.to("cpu")
            model = model.to("cpu")
            out = model(data)
            # print(out,data.y)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # val_acc,val_loss = TestPerformance(model,ValidationLoader)
        train_acc,train_loss = TestPerformance(model,TrainLoader)
        test_acc,test_loss = TestPerformance(model,TestLoader)

        # acc_val.append(val_acc)
        # loss_val.append(val_loss)

        acc_train.append(train_acc)
        loss_train.append(train_loss)
        acc_test.append(test_acc)
        loss_test.append(test_loss)
        # save_checkpoint(path= "../checkpoints/pointcloud/{1}_{0}.pt".format(ite,name),epoch=ite,model=model,optimizer=opt)


        print("Epoch: {0} | Train Loss: {1} | Train Acc: {2} | Test Loss: {3} | Test Acc: {4}".format(ite,train_loss,train_acc,test_loss,test_acc,size_all_mb))
    save_checkpoint(path="../checkpoints/pointcloud/{1}_{0}.pt".format(epoch,name),epoch=epoch,model=model,optimizer=opt)
    test_acc = max(acc_test)
    if show:
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize']= (21,5)
        h,w = 1,2
        plt.subplot(h,w,1)
        plt.plot(loss_train,label="Train loss")
        plt.plot(loss_test,label="Test loss")
        plt.title("Loss Report | {0} | ModelSize: {1} MB".format(name,size_all_mb))
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        #plt.show()

        plt.subplot(h,w,2)
        plt.plot(acc_train,label="Train Accuracy")
        plt.plot(acc_test,label="Test Accuracy")
        plt.title("Accuracy Report | Test Accuracy: {0}%".format(round(test_acc*100,2)))
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plt.savefig("./{0}.png".format(name))
        plt.show()
        plt.clf()

    return round(test_acc*100,2),model


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
    "send_feature":False,
    "num_classes":10
}
# WD=0.0005,lr=0.01, "p_dropout":0.25,pooling_ratio":0.25,

#beetween
#KATz
#closeness
#eigen
#harmonic
#load
#page

model = SAGPoolNet(**MAINargs)

acc,model = Train(model,TrainLoader=TrainLoader,ValidationLoader=ValidationLoader,TestLoader=TestLoader,
            epoch=100,lr=0.01,weight_decay=0.0002,show=True,name="Self-Attention Graph Pooling-ModelNet10-100epoch-test")


# def TestPerfomancePointNet(model,loader):
#     with torch.no_grad():
#         model.eval()
#         correct = 0.
#         loss = 0.
#         for data in loader:
#             inputs, labels = data['pointcloud'].to("cuda:0").float(), data['category'].to("cuda:0")
#             inputs = inputs.to("cuda")
#             labels = labels.to("cuda")
#             model = model.to("cuda")
#             out,_,__ = model(inputs.transpose(1,2))
#             pred = out.max(dim=1)[1]
#             correct += pred.eq(labels).sum().item()
#             loss += PointNetLoss(out, labels, _, __).item()

#     return correct / len(loader.dataset),loss / len(loader.dataset)


# def TrainPointNet(model, train_loader, val_loader,lr=0.01,weight_decay=0.0005, epochs=30, name="PointNet"):
#     param_size = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#     buffer_size = 0
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()
#     size_all_mb = (param_size + buffer_size) / 1024**2
#     size_all_mb = round(size_all_mb,3)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
#     model = model.to(device)

#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     loss_train = []
#     acc_train = []

#     loss_val = []
#     acc_val = []
#     for epoch in range(epochs):
#         model.train()

#         for i, data in enumerate(train_loader, 0):
#             inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
#             optimizer.zero_grad()
#             outputs, m3x3, m64x64 = model(inputs.transpose(1,2))
#             loss = PointNetLoss(outputs, labels, m3x3, m64x64)
#             loss.backward()
#             optimizer.step()


#         val_acc,val_loss = TestPerfomancePointNet(model,val_loader)
#         train_acc,train_loss = TestPerfomancePointNet(model,train_loader)

#         acc_val.append(val_acc)
#         loss_val.append(val_loss)

#         acc_train.append(train_acc)
#         loss_train.append(train_loss)

#         print("Epoch: {0} | Train Loss: {1} | Train Acc: {2} | Val Loss: {3} | Val Acc: {4}".format(epoch,train_loss,train_acc,val_loss,val_acc,size_all_mb))

#     test_acc = max(acc_val)


#     sns.set_style("whitegrid")
#     plt.rcParams['figure.figsize']= (21,5)
#     h,w = 1,2
#     plt.subplot(h,w,1)
#     plt.plot(loss_train,label="Train loss")
#     plt.plot(loss_val,label="Validation loss")
#     plt.title("Loss Report | {0} | ModelSize: {1} MB".format(name,size_all_mb))
#     plt.xlabel("Epoch")
#     plt.ylabel("NLLLoss")
#     plt.legend()
#     #plt.show()

#     plt.subplot(h,w,2)
#     plt.plot(acc_train,label="Train Accuracy")
#     plt.plot(acc_val,label="Validation Accuracy")
#     plt.title("Accuracy Report | Test Accuracy: {0}%".format(round(test_acc*100,2)))
#     plt.xlabel("Epoch")
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig("./{0}.png".format(name))
#     plt.show()
#     plt.clf()

#     return round(test_acc*100,2),model


# pointnet = PointNet()
# acc, model = TrainPointNet(pointnet, dataset_pointcloud_train_loader, dataset_pointcloud_test_loader, lr=0.001, weight_decay=0.0005, epochs=60, name="PointNet")

