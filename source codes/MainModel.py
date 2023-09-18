# import libraries
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

# Create class for training and testing all models
class MainModel(torch.nn):
    def __init__(self,model,dataset_path,data_loader,loss_function):
        super().__init__()
        self.dataset_path=dataset_path
        self.model=model
        self.data_loader=data_loader
        self.data_set={
            "train":None,
            "test":None
        }
        self.epoch=0
        self.train_losses=[]
        self.test_losses=[]
        self.loss_function=loss_function
        self.train_accuracy=[]
        self.test_accuracy=[]
        self.train_f1_score=[]
        self.test_f1_score=[]
        self.train_precision=[]
        self.test_precision=[]
        
        
    def read_dataset(self,reader,valid,force_to_cal=False):
        train=reader(self.dataset_path,valid=True,train=True,force_to_cal=force_to_cal)
        test=reader(self.dataset_path,valid=True,train=False,force_to_cal=force_to_cal)
        return train,test

    def procced_dataset(self,proccer,train,test):
        train=proccer(train)
        test=proccer(test)
        return train,test
        
    
    def load_dataset(self,procced_data_set,batch_size,shuffle=True,train=True):
        if train:
            self.data_set["train"]=self.data_loader(procced_data_set,train=True,batch_size=batch_size,shuffle=shuffle)
        else:
            self.data_set["test"]=self.data_loader(procced_data_set,train=False,batch_size=batch_size,shuffle=shuffle)
        
    def train(self):
        pass
    

    
    def save_weights(self):
        pass
    
    def load_weights(self):
        pass
    
    def calculate_accuracy(self,Train=True):
        if Train:
            loader=self.data_set["train"]
        else:
            loader=self.data_set["test"]
        with torch.no_grad():
            self.model.eval()
            correct = 0.
            model = model.to("cuda")
            for data in loader:
                outputs, m3x3, m64x64 = model(data)
                labels = data['category'].to("cuda")
                pred = outputs.max(dim=1)[1]
                correct += pred.eq(labels).sum().item()
        return correct / len(loader.dataset)
    
    def calculate_loss(self,Train=True ):
        if Train:
            loader=self.data_set["train"]
        else:
            loader=self.data_set["test"]
        with torch.no_grad():
            self.model.eval()
            loss = 0.
            model = model.to("cuda")
            for data in loader:
                outputs, m3x3, m64x64 = model(data)
                labels = data['category'].to("cuda")
                loss += self.loss_function(outputs, labels).item()

        return loss / len(loader.dataset)
        
    
    def calculate_confusion_matrix(self):
        pass
    
    def calculate_f1_score(self):
        pass
    
    def calculate_precision(self):
        pass
    
    def plot_confusion_matrix(self):
        pass
    
    def plot_loss(self,epochs,train_losses,test_losses):
        plt.plot(epochs,train_losses,label="train")
        plt.plot(epochs,test_losses,label="test")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        
        
    
    def plot_accuracy(self,epochs,train_accuracy,test_accuracy):
        plt.plot(epochs,train_accuracy,label="train")
        plt.plot(epochs,test_accuracy,label="test")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
    
    def plot_f1_score(self,epochs,train_f1_score,test_f1_score):
        plt.plot(epochs,train_f1_score,label="train")
        plt.plot(epochs,test_f1_score,label="test")
        plt.xlabel("epochs")
        plt.ylabel("f1_score")
        plt.legend()
        plt.show()
        
    
    


    

        
        
        
    
    
    

