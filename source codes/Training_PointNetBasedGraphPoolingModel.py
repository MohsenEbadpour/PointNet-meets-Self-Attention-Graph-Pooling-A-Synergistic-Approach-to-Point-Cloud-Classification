from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader
from pathlib import Path

from pre_process.CloudPointsPreprocessing import *
from pre_process.PointCloudGraphPreprocessing import *

from base_models.FeatureConcatModel import *
from base_models.PointNet import *
from base_models.PointNetBasedGraphPoolingModel import *
from base_models.SelfAttentionGraphPooling import *

from visualization.ReportVisualization import *

from TrainingUtils import *

path_global = Path("ModelNet10")
dataset_pointcloud_test = PointCloudData(path_global, valid=True, folder='test', force_to_cal=False)
dataset_pointcloud_train = PointCloudData(path_global, force_to_cal=False)

dataset_pointcloud_train_loader = TDataloader(dataset=dataset_pointcloud_train, batch_size=32, shuffle=True)
dataset_pointcloud_test_loader = TDataloader(dataset=dataset_pointcloud_test, batch_size=64)

model = PointNetBasedGraphPoolingModel()
acc, model = TrainCustom(model, dataset_pointcloud_train_loader, dataset_pointcloud_test_loader, lr=0.005, weight_decay=0.0005, epochs=60, name="PointNetBasedGraphPoolingModel")
