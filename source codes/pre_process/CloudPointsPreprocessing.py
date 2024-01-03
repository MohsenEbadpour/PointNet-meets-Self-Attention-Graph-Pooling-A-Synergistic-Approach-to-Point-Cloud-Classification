import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import os
import numpy as np
import math
import random
import os
import torch
from multiprocessing import Process
from torch.utils.data import Dataset
from torchvision import transforms
import threading
from tqdm import tqdm
from sklearn.neighbors import  kneighbors_graph
import pickle




path_global = Path("/home/ehsan/Desktop/Mohadeseh/ModelNet10")


def load_data(path):
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    print("Name of classes: ", classes)

    with open(path/"bed/train/bed_0001.off", 'r') as file:
        if 'OFF' != file.readline().strip():
            for i in range(10000000000):
                print(file)
            raise('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces,classes


# verts, faces, classes = load_data(path_global)
# i,j,k = np.array(faces).T
# x,y,z = np.array(verts).T


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],verts[faces[i][1]],verts[faces[i][2]]))
        sampled_faces = (random.choices(faces, weights=areas,cum_weights=None, k=self.output_size))
        sampled_points = np.zeros((self.output_size, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]], verts[sampled_faces[i][1]],verts[sampled_faces[i][2]]))
        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return  norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        return torch.from_numpy(pointcloud)


def DefaultTransforms():
    return transforms.Compose([ PointSampler(1024), Normalize(),ToTensor()])


def get_graph_features(point_cloud,N=6):
    connection_matrix = kneighbors_graph(point_cloud,N,mode='distance').toarray()
    graph = nx.DiGraph()
    X,Y,Z = {},{},{}
    for node in range(connection_matrix.shape[0]):
        for neighbor in range(connection_matrix.shape[1]):
            if neighbor == node:
                graph.add_edge(node,neighbor,weight=0)
                continue
            if connection_matrix[node][neighbor] == 0 :
                continue
            graph.add_edge(neighbor,node)
            graph[neighbor][node]["weight"] = connection_matrix[node][neighbor]
        X[node] = np.float64(point_cloud[node][0])
        Y[node] = np.float64(point_cloud[node][1])
        Z[node] = np.float64(point_cloud[node][2])

    features = [X,Y,Z]
    features_name = ["X","Y","Z"]

    betweenness_centrality = nx.betweenness_centrality(graph,weight="weight")
    features.append(betweenness_centrality)
    features_name.append("betweenness_centrality")

    katz_centrality = nx.katz_centrality(graph,weight="weight")
    features.append(katz_centrality)
    features_name.append("katz_centrality")

    closeness_centrality = nx.closeness_centrality(graph,distance="weight",)
    features.append(closeness_centrality)
    features_name.append("closeness_centrality")

    eigenvector_centrality = nx.eigenvector_centrality(graph,weight="weight",max_iter=100,tol=1e-3)
    features.append(eigenvector_centrality)
    features_name.append("eigenvector_centrality")

    harmonic_centrality = nx.harmonic_centrality(graph,distance="weight",)
    features.append(harmonic_centrality)
    features_name.append("harmonic_centrality")

    load_centrality = nx.load_centrality(graph,weight="weight")
    features.append(load_centrality)
    features_name.append("load_centrality")

    pagerank = nx.pagerank(graph,weight='weight')
    features.append(pagerank)
    features_name.append("pagerank")

    for idx in range(len(features)):
        nx.set_node_attributes(graph,features[idx],features_name[idx])

    nodes = nx.nodes(graph)
    features = []
    for node_indx in range(len(nodes)):
        features.append(np.array(list(nodes[node_indx].values())))
    features = np.array(features)

    features[:,3:] = features[:,3:] - np.mean(features[:,3:], axis=0)
    features[:,3:] /= np.max(np.linalg.norm(features[:,3:], axis=1))

    edge_list = np.array(nx.edges(graph))

    return torch.from_numpy(features),torch.from_numpy(edge_list), graph


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=DefaultTransforms(),force_to_cal = False):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else DefaultTransforms()
        self.valid = valid
        self.files = []
        self.force_to_cal = force_to_cal
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def read_off(self,file):
        if 'OFF' != file.readline().strip():
            print(file)
            raise('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return verts, faces

    def __preproc__(self, file):
        verts, faces = self.read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        name = str(pcd_path).split("/")[-1].split(".")[0]
        pointcloud_path = str(pcd_path).replace(name+".off",name+"_pointcloud.npz")
        graph_feature_path = str(pcd_path).replace(name+".off",name+"_graph_features.npz")
        graph_edge_list_path = str(pcd_path).replace(name+".off",name+"_graph_edge_list.npz")
        graph_path = str(pcd_path).replace(name+".off",name+"_graph.pickle")
        torch_graph_path = str(pcd_path).replace(name + ".off", name + "_torch_graph.pickle")

        if not(os.path.exists(pointcloud_path) and os.path.exists(graph_feature_path)) and self.force_to_cal:
            with open(pcd_path, 'r') as f:
                pointcloud = self.__preproc__(f)
                graph_features,edge_list,graph = get_graph_features(pointcloud)

                pointcloud_np = pointcloud.numpy()
                graph_features_np = graph_features.numpy()
                edge_list_np = edge_list.numpy()

                np.savez_compressed(pointcloud_path, pointcloud_np)
                np.savez_compressed(graph_feature_path, graph_features_np)
                np.savez_compressed(graph_edge_list_path, edge_list_np)

                with open(graph_path, 'wb') as handle:
                    pickle.dump(graph, handle)

        else:   
                pointcloud = torch.from_numpy(np.load(pointcloud_path)["arr_0"])
                graph_features = torch.from_numpy(np.load(graph_feature_path)["arr_0"])
                edge_list = torch.from_numpy(np.load(graph_edge_list_path)["arr_0"])

                with open(graph_path, 'rb') as handle:
                    graph = pickle.load(handle)
        
        # The edge is not a tensor, so dataloader cannot work. As it is not used in the training process, I have deleted it.
        return {'pointcloud': pointcloud,"edge_list":edge_list, 'category': self.classes[category],
                'graph_features': graph_features}
    
        # return {'pointcloud': pointcloud,"edge_list":edge_list,"graph":graph, 'category': self.classes[category],
        #         'graph_features': graph_features}

        # return {'pointcloud': pointcloud, "edge_list": edge_list, "graph": graph, 'category': self.classes[category],
        #         'graph_features': graph_features, "torch_graph_path": torch_graph_path}


def prepare_dataset(num,_cut,dataset):
    for i in range(len(dataset)):
        # print("Start ->",i)
        if i%num==_cut:
            sample = dataset[i]
            # print("Done! ->",i)


def handle_threads(num,dataset):
    threads = []
    for idx in tqdm(range(num)):
        thread = threading.Thread(target=prepare_dataset, args=(num,idx,dataset,))
        threads.append(thread)

    for thread in tqdm(threads):
        thread.start()

    for thread in threads:
        thread.join()

def multi_process(num,dataset):
    procs = []
    for idx in tqdm(range(num)):
        proc = Process(target=prepare_dataset, args=(num,idx,dataset,))
        procs.append(proc)
    for process in tqdm(procs):
        process.start()
    for proc in procs:
        proc.join()


# custom_transforms = transforms.Compose([PointSampler(1024),Normalize(), RandRotation_z(), RandomNoise(),ToTensor()])
# train_dataset = PointCloudData(path_global,force_to_cal=True)
# valid_dataset = PointCloudData(path_global, valid=True, folder='test',force_to_cal=True)

# multi_process(20,train_dataset)
# multi_process(20,valid_dataset)

