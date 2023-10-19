import os.path as path
import networkx as nx
import torch
from torch_geometric.data import Dataset as TGDataset, Data as TGData
import numpy as np
import zipfile

class Graph(TGData):

    def __init__(self,dataset_name):
        self.dataset_name = dataset_name
        self.graph = nx.Graph()

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
        # unzip the dataset
        with zipfile.ZipFile("../../datasets/graph" + self.dataset_name + ".zip", 'r') as zip_ref:
            zip_ref.extractall("../../datasets/graph")
        zip_ref.close()

    def tud_to_networkx(self):
        pre = self.dataset_name + "/"

        with open("../../datasets/graph" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_graph_indicator.txt", "r") as f:
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
        with open("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_A.txt", "r") as f:
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
        if path.exists("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_node_labels.txt"):
            with open("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_node_labels.txt", "r") as f:
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
        if path.exists("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_node_attributes.txt"):
            with open("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_node_attributes.txt", "r") as f:
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
        if path.exists("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_edge_labels.txt"):
            with open("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_edge_labels.txt", "r") as f:
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
        if path.exists("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_edge_attributes.txt"):
            with open("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_edge_attributes.txt", "r") as f:
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
        if path.exists("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_graph_labels.txt"):
            with open("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_graph_labels.txt", "r") as f:
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
        if path.exists("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_graph_attributes.txt"):
            with open("../../datasets/" + pre + self.dataset_name + "/raw/" + self.dataset_name + "_graph_attributes.txt", "r") as f:
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
            # apply this attribute for each node

            nx.set_node_attributes(g, nx.closeness_centrality(g), 'centrality')
            nx.set_node_attributes(g, nx.betweenness_centrality(g), 'betweenness')
            nx.set_node_attributes(g, nx.pagerank(g), 'pagerank')
            nx.set_node_attributes(g, nx.katz_centrality(g), 'katz')
            nx.set_node_attributes(g, nx.eigenvector_centrality(g), 'eigenvector')
            nx.set_node_attributes(g, nx.harmonic_centrality(g), 'harmonic')
            nx.set_node_attributes(g, nx.load_centrality(g), 'load')
            # normalize with mean max and linalg
            for nodes in nx.nodes(g):
                g.nodes[nodes]['centrality'] = (g.nodes[nodes]['centrality'] - np.mean(list(nx.get_node_attributes(g, 'centrality').values()))) / np.linalg.norm(list(nx.get_node_attributes(g, 'centrality').values()))
                g.nodes[nodes]['betweenness'] = (g.nodes[nodes]['betweenness'] - np.mean(list(nx.get_node_attributes(g, 'betweenness').values()))) / np.linalg.norm(list(nx.get_node_attributes(g, 'betweenness').values()))
                g.nodes[nodes]['pagerank'] = (g.nodes[nodes]['pagerank'] - np.mean(list(nx.get_node_attributes(g, 'pagerank').values()))) / np.linalg.norm(list(nx.get_node_attributes(g, 'pagerank').values()))
                g.nodes[nodes]['katz'] = (g.nodes[nodes]['katz'] - np.mean(list(nx.get_node_attributes(g, 'katz').values()))) / np.linalg.norm(list(nx.get_node_attributes(g, 'katz').values()))
                g.nodes[nodes]['eigenvector'] = (g.nodes[nodes]['eigenvector'] - np.mean(list(nx.get_node_attributes(g, 'eigenvector').values()))) / np.linalg.norm(list(nx.get_node_attributes(g, 'eigenvector').values()))
                g.nodes[nodes]['harmonic'] = (g.nodes[nodes]['harmonic'] - np.mean(list(nx.get_node_attributes(g, 'harmonic').values()))) / np.linalg.norm(list(nx.get_node_attributes(g, 'harmonic').values()))
                g.nodes[nodes]['load'] = (g.nodes[nodes]['load'] - np.mean(list(nx.get_node_attributes(g, 'load').values()))) / np.linalg.norm(list(nx.get_node_attributes(g, 'load').values()))

    def pre_process(self):
        self.unzip()
        self.tud_to_networkx()
        self.add_graph_features()
        self.write_to_file()

    def write_to_file(self):
        for i in range(len(self.graph)):
            nx.write_gpickle(self.graph[i], "./datasets/" + self.dataset_name + "/processed/" + self.dataset_name + "_" + str(i) + ".gpickle")
            print("Graph " + str(i) + " saved")

    def read_from_file(self):
        for i in range(len(self.graph)):
            self.graph[i] = nx.read_gpickle("./datasets/" + self.dataset_name + "/processed/" + self.dataset_name + "_" + str(i) + ".gpickle")
            print("Graph " + str(i) + " read")



def pre_process(dataset_name):
    graph = Graph(dataset_name)
    graph.pre_process()

def load_graph(dataset_name):
    graph = Graph(dataset_name)
    graph.read_from_file()
    return graph.graph

pre_process("MUTAG")

                    
        


