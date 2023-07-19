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
from GraphPreprocessing import *
from PointNet import *
from PointNetBasedGraphPoolingModel import *
from SelfAttentionGraphPooling import * 

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data, layout=go.Layout(    updatemenus=[dict(type='buttons', showactive=False,y=1,x=0.8,xanchor='left',yanchor='bottom',pad=dict(t=45, r=10),
                                                    buttons=[dict(label='Play', method='animate',args=[None, dict(frame=dict(duration=50, redraw=True),transition=dict(duration=0),fromcurrent=True,mode='immediate')])])
                                                    ]), frames=frames)
    return fig

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.show()
