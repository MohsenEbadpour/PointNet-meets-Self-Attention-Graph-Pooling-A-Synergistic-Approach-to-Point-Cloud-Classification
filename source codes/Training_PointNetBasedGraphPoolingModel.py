from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader
from pathlib import Path

from pre_process.CloudPointsPreprocessing import *
from pre_process.PointCloudGraphPreprocessing import *

from base_models.PointNet import *
from base_models.PointNetBasedGraphPoolingModel import *
from base_models.SelfAttentionGraphPooling import *


from TrainingUtils import *

path_global = Path("../datasets/pointcloud/raw/ModelNet40")
dataset_pointcloud_test = PointCloudData(path_global, valid=True, folder='test', force_to_cal=False)
dataset_pointcloud_train = PointCloudData(path_global, force_to_cal=False)
TrainSet,ValidationSet,TestSet = GetSets(dataset_pointcloud_train,train=0.97,valid=0.03)

dataset_pointcloud_train_loader = TDataloader(dataset=TrainSet, batch_size=64, shuffle=True)
dataset_pointcloud_test_loader = TDataloader(dataset=ValidationSet, batch_size=64)

model = PointNetBasedGraphPoolingModel(num_classes=40)
acc, model = TrainCustom(model, dataset_pointcloud_train_loader, dataset_pointcloud_test_loader, lr=0.01, weight_decay=0.0005, epochs=100, name="PointNetBasedGraphPoolingModel", file_name="PB-40-val")
