from torch.utils.data import Dataset as TDataset, DataLoader as TDataloader
from pathlib import Path

from pre_process.CloudPointsPreprocessing import *
from pre_process.PointCloudGraphPreprocessing import *


from base_models.FeatureConcatModel import *
from base_models.PointNet import *
from base_models.SelfAttentionGraphPooling import *
from torch_geometric.data import DataLoader


from TrainingUtils import *

path_global = Path("../datasets/pointcloud/raw/ModelNet10")
dataset_pointcloud_test = PointCloudData(path_global, valid=True, folder='test', force_to_cal=False)
dataset_pointcloud_train = PointCloudData(path_global, force_to_cal=False)
torch.manual_seed(42)
np.random.seed(42)

# dataset_graph_test = PointCloudGraph(dataset_pointcloud_test)
# dataset_graph_train = PointCloudGraph(dataset_pointcloud_train)

dataset_pointcloud_train_loader = DataLoader(dataset=dataset_pointcloud_train, batch_size=64, shuffle=True)
dataset_pointcloud_test_loader = DataLoader(dataset=dataset_pointcloud_test, batch_size=64)

model = FeatureConcatModel()
acc, model = TrainCustom(model, dataset_pointcloud_train_loader, dataset_pointcloud_test_loader, lr=0.005, weight_decay=0.0005,epochs=100, name="FeatureConcatModel")