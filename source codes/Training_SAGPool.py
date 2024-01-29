from MainModel import *
from pre_process.GraphPreprocessing import *
from base_models.SelfAttentionGraphPooling import *

DATASET_NAME="NCI1"

# default values: pr=0.25, pd=0.25,batch_size=32,train /test= 0.8/0.2 ,weight_decay=0.0005, epoch=0.3,lr=0.01
MAINargs = {
    "SAGPoolNet_dataset_features":7,
    "out_channels":1,
    "is_hierarchical":True,
    "use_w_for_concat":True,
    "pooling_ratio":0.25,
    "p_dropout":0.25,
    "Conv":GATConv,
    "heads":6,
    "concat":False,
    "send_feature":False,
    "num_classes":2
}




model = SAGPoolNet(**MAINargs)

model = MainModel(model,dataset_name=DATASET_NAME,name="SAGPoolNet",save_address="self-attention-graph-pooling/graph_dataset")
model.load_data(loader_function=load_graph,batch_size=32,train_size=0.8,validation_size=0.2)
model.train(convert_function=ConvertBatchToGraph,lr=0.01,weight_decay=0.0005,epochs=30)

