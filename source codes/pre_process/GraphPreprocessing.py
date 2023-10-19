import os.path as path
import networkx as nx
import torch
from torch_geometric.data import Dataset as TGDataset, Data as TGData
import numpy as np
import zipfile
import os
import pickle
class Graph(TGData):

    def __init__(self,ds_name,root="./", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dataset_name = ds_name
        self.graph = nx.DiGraph()

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx):
        return self.graph[idx]
    
    def get_sets(self,dataset,train=0.99,test=0.1):
        train_size = int(train*len(dataset))
        test_size = int(test*len(dataset))
        validation_size = len(dataset) - train_size - test_size
        return torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
        
    def unzip(self):
        with zipfile.ZipFile("../../datasets/graph/" + self.dataset_name + ".zip", 'r') as zip_ref:
            zip_ref.extractall("../../datasets/graph/")
        zip_ref.close()
        # create raw folder
        os.system("mkdir ../../datasets/graph/" + self.dataset_name + "/raw")
        # copy files to raw folder
        os.system("cp -r ../../datasets/graph/" + self.dataset_name + "/*.txt ../../datasets/graph/" + self.dataset_name + "/raw/")
        # create raw and procceced folders
        os.system("mkdir ../../datasets/graph/" + self.dataset_name + "/processed")
        # remove unzipped folder
        os.system("rm -r ../../datasets/graph/" + self.dataset_name+"/*.txt")
        

    def tud_to_networkx(self):
        pre = self.dataset_name + "/"

        with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_graph_indicator.txt", "r") as f:
            graph_indicator = [int(i) - 1 for i in list(f)]
        f.closed

        # Nodes.
        num_graphs = max(graph_indicator)
        node_indices = []
        offset = []
        c = 0

        for i in range(num_graphs + 1):
            offset.append(c)
            c_i = graph_indicator.count(i)
            node_indices.append((c, c + c_i - 1))
            c += c_i

        graph_db = []
        for i in node_indices:
            g = nx.Graph()
            for j in range(i[1] - i[0]+1):
                g.add_node(j)

            graph_db.append(g)

        # Edges.
        with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_A.txt", "r") as f:
            edges = [i.split(',') for i in list(f)]
        f.closed
        
        edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]
        edge_list = []
        edgeb_list = []
        for e in edges:
            g_id = graph_indicator[e[0]]
            g = graph_db[g_id]
            off = offset[g_id]

            # Avoid multigraph (for edge_list)
            if ((e[0] - off, e[1] - off) not in list(g.edges())) and ((e[1] - off, e[0] - off) not in list(g.edges())):
                g.add_edge(e[0] - off, e[1] - off)
                edge_list.append((e[0] - off, e[1] - off))
                edgeb_list.append(True)
            else:
                edgeb_list.append(False)

        # Node labels.
        if path.exists("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_node_labels.txt"):
            with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_node_labels.txt", "r") as f:
                node_labels = [str.strip(i) for i in list(f)]
            f.closed
            
            node_labels = [i.split(',') for i in node_labels]
            int_labels = [];
            for i in range(len(node_labels)):
                int_labels.append([int(j) for j in node_labels[i]])
            
            i = 0
            for g in graph_db:
                for v in range(g.number_of_nodes()):
                    g.nodes[v]['labels'] = int_labels[i]
                    i += 1

        # Node Attributes.
        if path.exists("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_node_attributes.txt"):
            with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_node_attributes.txt", "r") as f:
                node_attributes = [str.strip(i) for i in list(f)]
            f.closed
            
            node_attributes = [i.split(',') for i in node_attributes]
            float_attributes = [];
            for i in range(len(node_attributes)):
                float_attributes.append([float(j) for j in node_attributes[i]])
            i = 0
            for g in graph_db:
                for v in range(g.number_of_nodes()):
                    g.nodes[v]['attributes'] = float_attributes[i]
                    i += 1

        # Edge Labels.
        if path.exists("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_edge_labels.txt"):
            with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_edge_labels.txt", "r") as f:
                edge_labels = [str.strip(i) for i in list(f)]
            f.closed

            edge_labels = [i.split(',') for i in edge_labels]
            e_labels = []
            for i in range(len(edge_labels)):
                if(edgeb_list[i]):
                    e_labels.append([int(j) for j in edge_labels[i]])
            
            i = 0
            for g in graph_db:
                for e in range(g.number_of_edges()):
                    g.edges[edge_list[i]]['labels'] = e_labels[i]
                    i += 1

        # Edge Attributes.
        if path.exists("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_edge_attributes.txt"):
            with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_edge_attributes.txt", "r") as f:
                edge_attributes = [str.strip(i) for i in list(f)]
            f.closed

            edge_attributes = [i.split(',') for i in edge_attributes]
            e_attributes = []
            for i in range(len(edge_attributes)):
                if(edgeb_list[i]):
                    e_attributes.append([float(j) for j in edge_attributes[i]])
            
            i = 0
            for g in graph_db:
                for e in range(g.number_of_edges()):
                    g.edges[edge_list[i]]['attributes'] = e_attributes[i]
                    i += 1

        # Classes.
        if path.exists("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_graph_labels.txt"):
            with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_graph_labels.txt", "r") as f:
                classes = [str.strip(i) for i in list(f)]
            f.closed
            classes = [i.split(',') for i in classes]
            cs = []
            for i in range(len(classes)):
                cs.append([int(j) for j in classes[i]])
            
            i = 0
            for g in graph_db:
                g.graph['classes'] = cs[i]
                i += 1

        # Targets.
        if path.exists("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_graph_attributes.txt"):
            with open("../../datasets/graph/"  + self.dataset_name + "/raw/" + self.dataset_name + "_graph_attributes.txt", "r") as f:
                targets = [str.strip(i) for i in list(f)]
            f.closed
            
            targets= [i.split(',') for i in targets]
            ts = []
            for i in range(len(targets)):
                ts.append([float(j) for j in targets[i]])
            
            i = 0
            for g in graph_db:
                g.graph['targets'] = ts[i]
                i += 1

        self.graph = graph_db
    def add_graph_features(self):
        # apply for each node
        for g in self.graph:
            features = []
            features_name = []

            betweenness_centrality = nx.betweenness_centrality(g,weight="weight")
            features.append(betweenness_centrality)
            features_name.append("betweenness_centrality")

            katz_centrality = nx.katz_centrality(g,weight="weight")
            features.append(katz_centrality)
            features_name.append("katz_centrality")

            closeness_centrality = nx.closeness_centrality(g,distance="weight",)
            features.append(closeness_centrality)
            features_name.append("closeness_centrality")

            eigenvector_centrality = nx.eigenvector_centrality(g,weight="weight",max_iter=100,tol=1e-3)
            features.append(eigenvector_centrality)
            features_name.append("eigenvector_centrality")

            harmonic_centrality = nx.harmonic_centrality(g,distance="weight",)
            features.append(harmonic_centrality)
            features_name.append("harmonic_centrality")

            load_centrality = nx.load_centrality(g,weight="weight")
            features.append(load_centrality)
            features_name.append("load_centrality")

            pagerank = nx.pagerank(g,weight='weight')
            features.append(pagerank)
            features_name.append("pagerank")

            for idx in range(len(features)):
                nx.set_node_attributes(g,features[idx],features_name[idx])


    def pre_process(self):
        self.unzip()
        self.tud_to_networkx()
        self.add_graph_features()
        self.write_to_file()

    def write_to_file(self):
        for i in range(len(self.graph)):
            nx.write_gml(self.graph[i], "../../datasets/graph/" + self.dataset_name + "/processed/" + self.dataset_name + "_" + str(i) + ".gml")
            print("Graph " + str(i) + " saved")

    def read_from_file(self):
        for i in range(len(self.graph)):
            self.graph[i] = nx.read_gml("../../datasets/graph" + self.dataset_name + "/processed/" + self.dataset_name + "_" + str(i) + ".gml")
            print("Graph " + str(i) + " read")



def pre_process(dataset_name):
    graph = Graph(dataset_name)
    graph.pre_process()

def load_graph(dataset_name):
    graph = Graph(dataset_name)
    graph.read_from_file()
    return graph.graph

pre_process("MUTAG")
pre_process("ENZYMES")
pre_process("NCI1")
pre_process("PROTEINS_full")
                    
        


