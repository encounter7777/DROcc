# DRocc: Dual Residual Transformer for Indoor Occupancy Prediction

DRocc provides multi-view occupancy prediction on the EmbodiedScan dataset. It builds on the DenseFusion/ImVoxel occupancy pipeline with DRocc-specific heads and configs.

## Features
- Multi-view RGB-D occupancy prediction (semantic or binary) with Minkowski 3D backbone.
- DRocc variants (`mv-occ-drocc.py`) that fuse image/point volumes via cross-attention.
- Continuous and sparse settings supported via provided configs.

## Pipeline
- See the DRocc occupancy pipeline walkthrough in [pipeline.pdf](pipeline.pdf).

## 📚 Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 525.147.05
- CUDA 12.0
- Python 3.8.18
- PyTorch 1.11.0+cu113
- PyTorch3D 0.7.2

1. Clone this repository.

```bash
git clone https://github.com/encounter7777/DROcc
cd DROcc
```

2. Create an environment and install PyTorch.

```bash
conda create -n DROcc python=3.8 -y  # pytorch3d needs python>3.7
conda activate DROcc
# Install PyTorch, for example, install PyTorch 1.11.0 for CUDA 11.3
# For more information, please refer to https://pytorch.org/get-started/locally/
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
## Installation
Follow the project installer to set up dependencies (CUDA/PyTorch/MinkowskiEngine):
```bash
python install.py all
```

## Data Preparation
Organize data as in `data/README.md` (raw ScanNet, 3RScan, Matterport3D, ARKitScenes plus EmbodiedScan annotations):
```
data
├── scannet/...
├── 3rscan/...
├── matterport3d/...
├── arkitscenes/...
├── embodiedscan_occupancy
├── embodiedscan_infos_train.pkl
├── embodiedscan_infos_val.pkl
├── embodiedscan_infos_test.pkl
```
Then extract occupancy annotations:
```bash
python embodiedscan/converter/extract_occupancy_ann.py --src data/embodiedscan_occupancy --dst data
```

## Training
Run the DRocc multi-view occupancy model (RGB-D):
```bash
# Single GPU
python tools/train.py configs/occupancy/mv-occ-drocc.py --work-dir work_dirs/mv-occ-drocc
```


## Evaluation / Inference
Evaluate a checkpoint (adjust path as needed):
```bash
python tools/test.py configs/occupancy/mv-occ-drocc.py work_dirs/mv-occ-drocc/epoch_24.pth
```




