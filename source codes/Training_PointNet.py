import torch
from torch import optim

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import DataLoader
from pre_process.CloudPointsPreprocessing import *
from pre_process.PointCloudGraphPreprocessing import *


from base_models.PointNet import *

from visualization.ReportVisualization import *


path_global = Path("../datasets/pointcloud/raw/modelnet-10/ModelNet10")
dataset_pointcloud_test = PointCloudData(path_global, valid=True, folder='test',force_to_cal=False)
dataset_pointcloud_train = PointCloudData(path_global, force_to_cal=False)


dataset_pointcloud_train_loader = DataLoader(dataset=dataset_pointcloud_train, batch_size=32, shuffle=True)
dataset_pointcloud_test_loader = DataLoader(dataset=dataset_pointcloud_test, batch_size=64)

for data,i in dataset_pointcloud_train_loader:
    print("hi")


exit
def TestPerfomancePointNet(model,loader):
    with torch.no_grad():
        model.eval()
        correct = 0.
        loss = 0.
        for data in loader:
            inputs, labels = data['pointcloud'].to("cuda:0").float(), data['category'].to("cuda:0")
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            model = model.to("cuda")
            out,_,__ = model(inputs.transpose(1,2))
            pred = out.max(dim=1)[1]
            correct += pred.eq(labels).sum().item()
            loss += PointNetLoss(out, labels, _, __).item()

    return correct / len(loader.dataset),loss / len(loader.dataset)


def TrainPointNet(model, train_loader, val_loader,lr=0.01,weight_decay=0.0005, epochs=30, name="PointNet"):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_mb = round(size_all_mb,3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_train = []
    acc_train = []

    loss_val = []
    acc_val = []
    for epoch in range(epochs):
        model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))
            loss = PointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()


        val_acc,val_loss = TestPerfomancePointNet(model,val_loader)
        train_acc,train_loss = TestPerfomancePointNet(model,train_loader)

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


pointnet = PointNet()
acc, model = TrainPointNet(pointnet, dataset_pointcloud_train_loader, dataset_pointcloud_test_loader, lr=0.001, weight_decay=0.0005, epochs=60, name="PointNet")

