from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader
from pathlib import Path

from CloudPointsPreprocessing import *
from FeatureConcatModel import *
from GraphPreprocessing import *
from PointNet import *
from PointNetBasedGraphPoolingModel import *
from ReportVisualization import *
from SelfAttentionGraphPooling import *
from TrainingUtils import *

path_global = Path("ModelNet10")
dataset_pointcloud_test = PointCloudData(path_global, valid=True, folder='test', force_to_cal=False)
dataset_pointcloud_train = PointCloudData(path_global, force_to_cal=False)

dataset_pointcloud_train_loader = TDataloader(dataset=dataset_pointcloud_train, batch_size=32, shuffle=True)
dataset_pointcloud_test_loader = TDataloader(dataset=dataset_pointcloud_test, batch_size=64)

model = FeatureConcatModel()
acc, model = TrainCustom(model, dataset_pointcloud_train_loader, dataset_pointcloud_test_loader, lr=0.005, weight_decay=0.0005,epochs=60, name="FeatureConcatModel")