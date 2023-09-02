import argparse

parser = argparse.ArgumentParser()

# Training model arguments
parser.add_argument('--model', help='Training model name between FeatureConcatModel, PointNetBasedGraphPoolingModel, '
                                    'PointNet [default: FeatureConcatModel]', default='FeatureConcatModel')
parser.add_argument('--learning_rate', help='Initial learning rate [default: 0.001]', type=float, default=0.001)
parser.add_argument('--decay', help='Weight decay during learning [default: 0.0005]', type=float, default=0.0005)
parser.add_argument('--epoch', help='Epoch to run [default: 60]', type=int, default=60)
parser.add_argument('--batch_size', help='Batch size during training default: 32]', type=int, default=32)
parser.add_argument('--optimizer', help='adam or momentum [default: adam]', default='adam')
parser.add_argument('--classes', help='Number of classes to predict [default: 10]', type=int, default=10)

# Dataset and preprocessing arguments
parser.add_argument('--preprocess_dir', help="Absolute address of a folder to preprocess")
parser.add_argument('--train_ratio', help="Training ratio to split the dataset [default: 0.8]", type=float, default=0.8)
parser.add_argument('--test_dataset_dir', help="Absolute address of test dataset [default: ??]", type=str, default="??")  # To Do
parser.add_argument('--train_dataset_dir', help="Absolute address of train dataset [default: ??]", type=str, default="??")  # To Do
parser.add_argument('--transform', help="Define possible transforms ??")  # To Do (GraphPreprocessing.py)
parser.add_argument('--pre_transform', help="Define possible pre_transforms ??")  # To Do (GraphPreprocessing.py)
parser.add_argument('--pre_filter', help="Define possible pre_filters ??")  # To Do (GraphPreprocessing.py)
parser.add_argument('--num_of_neighbor', help="Define possible number of nodes to consider as a same graph ??")  # To Do (GraphPreprocessing.py)
parser.add_argument('--alpha', help="Used in PointNetLoss ??")  # To Do (PointNet.py)

args = parser.parse_args()