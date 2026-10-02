# PointNet Meets Self-Attention Graph Pooling

This repository contains the code and experimental artifacts for **PointNet meets Self-Attention Graph Pooling: A Synergistic Approach to Point Cloud Classification**. The project studies whether PointNet's point-wise representations and Self-Attention Graph Pooling (SAGPool)'s graph-level representations can be combined to improve 3D point-cloud classification.

Read the full report: [PDF](<report/Report -PointNet meets Self-Attention Graph Pooling - A Synergistic Approach to Point Cloud Classification/report.pdf>).

## Overview

Point clouds are converted to weighted, directed 6-nearest-neighbor graphs. Each sampled point contributes its XYZ coordinates and can be enriched with seven graph-centrality values:

- Betweenness
- Closeness
- Katz
- PageRank
- Eigenvector
- Harmonic
- Load

The study evaluates two baselines and two proposed fusion models:

| Model | Description |
| --- | --- |
| PointNet | Point-wise baseline that learns permutation-invariant global shape features. |
| Hierarchical SAGPool | Graph baseline that uses attention-based graph pooling to retain informative nodes. |
| FeatureConcat | Concatenates PointNet and SAGPool graph-level features before classification. |
| PointNetBasedGraphPooling | Uses PointNet's per-point features as the input to SAGPool. |

## Key results

Experiments use 1,024 sampled points per object from ModelNet10 and ModelNet40. The best test accuracy for each model and dataset, as reported in the paper, is below.

| Model | Dataset | Best input features | Test accuracy | Model size |
| --- | --- | --- | ---: | ---: |
| PointNet | ModelNet10 | XYZ + 7 centralities | 92.95% | 13.259 MB |
| PointNet | ModelNet40 | XYZ | 87.44% | 13.259 MB |
| Hierarchical SAGPool | ModelNet10 | XYZ + harmonic | 87.89% | 0.3 MB |
| Hierarchical SAGPool | ModelNet40 | XYZ | 78.44% | 0.3 MB |
| FeatureConcat | ModelNet10 | XYZ | 94.34% | 11.418 MB |
| FeatureConcat | ModelNet40 | XYZ + harmonic | 88.20% | 11.418 MB |
| PointNetBasedGraphPooling | ModelNet10 | XYZ + 7 centralities | **95.80%** | 12.136 MB |
| PointNetBasedGraphPooling | ModelNet40 | XYZ + harmonic | 85.47% | 12.136 MB |

PointNetBasedGraphPooling achieved the best ModelNet10 result, improving on the PointNet baseline by 2.85 percentage points while using about 1.1 MB less model storage. FeatureConcat achieved the best ModelNet40 result. The results also show that the usefulness of centralities is dataset- and architecture-dependent: harmonic centrality is especially effective for the SAGPool and fusion configurations, while the all-centralities PointNetBasedGraphPooling configuration is strongest on ModelNet10.

## Method

1. Sample 1,024 surface points from each ModelNet mesh and normalize the point cloud.
2. Build a directed, weighted k-nearest-neighbor graph with `k=6`.
3. Compute the seven centrality measures and append them to XYZ when the selected experiment requires graph-derived features.
4. Train either a baseline or a fusion architecture. The experimental setup uses Adam, graph-pooling ratio 0.25, and six-head graph-attention convolutions.

The preprocessing implementation is in [`source codes/pre_process/CloudPointsPreprocessing.py`](<source codes/pre_process/CloudPointsPreprocessing.py>), and the model implementations are in [`source codes/base_models`](<source codes/base_models>).

## Repository layout

```text
source codes/
  base_models/       PointNet, SAGPool, and the two fusion architectures
  pre_process/       Point-cloud sampling, graph construction, and centralities
  visualization/     Training-result and point-cloud visualization utilities
  Training_*.py      Experiment entry points
outputs/             Saved training, validation, and test metrics
checkpoints/         Saved point-cloud checkpoints
results/             Plots and result figures
report/              Project report and supporting source files
```

## Setup

The original experiments used Python, PyTorch, PyTorch Geometric, NetworkX, NumPy, scikit-learn, and an NVIDIA RTX 3090 GPU. Install the pinned environment with:

```bash
python -m pip install -r requirements.txt
```

Place a ModelNet dataset in the location expected by the training script. The fusion scripts currently expect:

```text
datasets/pointcloud/raw/ModelNet40/<class>/{train,test}/*.off
```

The PointNet script is configured for `datasets/pointcloud/raw/modelnet-10/ModelNet10`. The `PointCloudData` loader creates and reuses precomputed point-cloud and graph-feature files beside the `.off` source files when `force_to_cal=True`.

## Running experiments

Run the scripts from `source codes` so their local imports and relative paths resolve:

```bash
cd "source codes"
python Training_PointNet.py
python Training_FeatureConcatModel.py
python Training_PointNetBasedGraphPoolingModel.py
```

The entry points contain research-specific dataset paths, model dimensions, hyperparameters, and output paths. Update those settings before a new run. The fusion-model forward passes explicitly target CUDA, so an NVIDIA CUDA environment is required unless the device handling is adapted for CPU or another accelerator.

## Citation

If you use this software, please cite the metadata in [CITATION.cff](CITATION.cff).

## License

This project is licensed under the [GNU GPL v3.0](LICENSE).

## Authors

- [Mohsen Ebadpour](https://github.com/MohsenEbadpour) - Amirkabir University of Technology
- [Mohammad Choupan](https://github.com/mohamadch91) - University of Bologna
- Mehdi Javanmardi - Amirkabir University of Technology

For questions or suggestions, please open an issue in the repository.
