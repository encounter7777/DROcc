
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
git clone 
cd DROcc
```

2. Create an environment and install PyTorch.

```bash
conda create -n embodiedscan python=3.8 -y  # pytorch3d needs python>3.7
conda activate embodiedscan
# Install PyTorch, for example, install PyTorch 1.11.0 for CUDA 11.3
# For more information, please refer to https://pytorch.org/get-started/locally/
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Install EmbodiedScan.

```bash
# We plan to make EmbodiedScan easier to install by "pip install EmbodiedScan".
# Please stay tuned for the future official release.
# Make sure you are under ./EmbodiedScan/
# This script will install the dependencies and EmbodiedScan package automatically.
# use [python install.py run] to install only the execution dependencies
# use [python install.py visual] to install only the visualization dependencies
python install.py all  # install all the dependencies
```

**Note:** The automatic installation script make each step a subprocess and the related messages are only printed when the subprocess is finished or killed. Therefore, it is normal to seemingly hang when installing heavier packages, such as Mink Engine and PyTorch3D.

BTW, from our experience, it is easier to encounter problems when installing these two packages. Feel free to post your questions or suggestions during the installation procedure.

### Data Preparation

Please refer to the [guide](data/README.md) for downloading and organization.

### Tutorial

We provide a simple tutorial [here](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/embodiedscan/tutorial.ipynb) as a guideline for the basic analysis and visualization of our dataset. Welcome to try and post your suggestions!

### Demo Inference

We provide a demo for running EmbodiedScan's model on a sample scan. Please download the raw data from [Google Drive](https://drive.google.com/file/d/1nXIbH56TmIoEVv1AML7mZS0szTR5HgNC/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1GK9Z4M-VbRSMWErB39QGpg?pwd=v5w1) and refer to the [notebook](demo/demo.ipynb) for more details.

## 📦 Model and Benchmark

### Model Overview

<p align="center">
  <img src="assets/framework.png" align="center" width="100%">
</p>
Embodied Perceptron accepts RGB-D sequence with any number of views along with texts as multi-modal input. It uses classical encoders to extract features for each modality and adopts dense and isomorphic sparse fusion with corresponding decoders for different predictions. The 3D features integrated with the text feature can be further used for language-grounded understanding.

<!-- #### Pipeline Flow
<video src="assets/scannet_long_demo.mp4" controls>
</video>

#### Multi-objects Interaction
<video src="assets/multiobj_multistep_1.mp4" controls>
</video>
<video src="assets/multiobj_multistep_2.mp4" controls>
</video>

#### Diverse Interactions with the Same Object
<video src="assets/multistep_sit_demo.mp4" controls>
</video>
<video src="assets/multistep_bed_demo.mp4" controls>
</video>

#### ''Multi-agent'' Interaction Planned by LLMs
<video src="assets/scannet_two_bed_demo.mp4" controls>
</video> -->

### Training and Evaluation

We provide configs for different tasks [here](configs/) and you can run the train and test script in the [tools folder](tools/) for training and inference.
For example, to train a multi-view 3D detection model with pytorch, just run:

```bash
# Single GPU training
python tools/train.py configs/detection/mv-det3d_8xb4_embodiedscan-3d-284class-9dof.py --work-dir=work_dirs/mv-3ddet

# Multiple GPU training
python tools/train.py configs/detection/mv-det3d_8xb4_embodiedscan-3d-284class-9dof.py --work-dir=work_dirs/mv-3ddet --launcher="pytorch"
```

Or on the cluster with multiple machines, run the script with the slurm launcher following the sample script provided [here](tools/mv-grounding.sh).

NOTE: To run the multi-view 3D grounding experiments, please first download the 3D detection pretrained model to accelerate its training procedure. After downloading the detection checkpoint, please check the path used in the config, for example, the `load_from` [here](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof.py#L210), is correct.

To inference and evaluate the model (e.g., the checkpoint `work_dirs/mv-3ddet/epoch_12.pth`), just run the test script:

```bash
# Single GPU testing
python tools/test.py configs/detection/mv-det3d_8xb4_embodiedscan-3d-284class-9dof.py work_dirs/mv-3ddet/epoch_12.pth

# Multiple GPU testing
python tools/test.py configs/detection/mv-det3d_8xb4_embodiedscan-3d-284class-9dof.py work_dirs/mv-3ddet/epoch_12.pth --launcher="pytorch"
```

### Using Visualizer during inference

We provide EmbodiedScanBaseVisualizer to visualize the output of models during inference. Please refer to the [guide](embodiedscan/visualizer/README.md) for detail.

### Inference and Submit your Results

We preliminarily support format-only inference for multi-view 3D visual grounding. To achieve format-only inference during test, just set `format_only=True` in `test_evaluator` in the corresponding config like [here](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof.py#L183). Then just run the test script like:

```bash
python tools/test.py configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof.py work_dirs/mv-grounding/epoch_12.pth --launcher="pytorch"
```

The prediction file will be saved to `./test_results.json` in the current directory.
You can also set the `result_dir` in `test_evaluator` to specify the directory to save the result file.

Finally, to pack the prediction file into the submission format, please modify the script `tools/submit_results.py` according to your team information and saving paths, and run:

```bash
python tools/submit_results.py
```

Then you can submit the resulting pkl file to the test server and wait for the lottery :)

We also provide a sample script `tools/eval_script.py` for evaluating the submission file and you can check it by yourself to ensure your submitted file has the correct format.

### Benchmark

We preliminarily provide several baseline results here with their logs and pretrained models.

Note that the performance is a little different from the results provided in the paper because we re-split the training set as the released training and validation set while keeping the original validation set as the test set for the public benchmark.

#### Multi-View 3D Detection

| Method | Input | AP@0.25 | AR@0.25 | AP@0.5 | AR@0.5 | Download |
|:------:|:-----:|:-------:|:-------:|:------:|:------:|:------:|
| [Baseline](configs/detection/mv-det3d_8xb4_embodiedscan-3d-284class-9dof.py) | RGB-D | 15.22  | 52.23  | 8.13  | 26.66 | [Model](https://download.openmmlab.com/mim-example/embodiedscan/mv-3ddet.pth), [Log](https://download.openmmlab.com/mim-example/embodiedscan/mv-3ddet.log) |

#### Continuous 3D Detection

| Method | Input | AP@0.25 | AR@0.25 | AP@0.5 | AR@0.5 | Download |
|:------:|:-----:|:-------:|:-------:|:------:|:------:|:------:|
| [Baseline](configs/detection/cont-det3d_8xb1_embodiedscan-3d-284class-9dof.py) | RGB-D | 17.83  | 47.53  | 9.04  | 23.04 | [Model](https://download.openmmlab.com/mim-example/embodiedscan/cont-3ddet.pth), [Log](https://download.openmmlab.com/mim-example/embodiedscan/cont-3ddet.log) |

#### Multi-View 3D Visual Grounding

| Method |AP@0.25| AP@0.5| Download |
|:------:|:-----:|:-------:|:------:|
| [Baseline-Mini](configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof.py) | 33.59 | 14.40 | [Model](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding.pth), [Log](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding.log) |
| [Baseline-Mini (w/ FCAF box coder)](configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof_fcaf-coder.py) | - | - | - |
| [Baseline-Full](configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof-full.py) | 36.78 | 15.97 | [Model](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-full.pth), [Log](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-full.log) |

Note: As mentioned in the paper, due to much more instances annotated with our new tools and pipelines, we concatenate several simple prompts as more complex ones to ensure those prompts to be more accurate without potential ambiguity. The above table is the benchmark without complex prompts using the initial version of visual grounding data.

We found such data is much less than the main part though, it can boost the multi-modal model's performance a lot. Meanwhile, whether to include these data in the validation set is not much important. We provide the updated benchmark as below and update a version of visual grounding data via emails to the community.

| Method | train | val | AP@0.25| AP@0.5| Download |
|:------:|:-----:|:---:|:------:|:-----:|:--------:|
| [Baseline-Full](configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof-full.py) | w/o complex | w/o complex | 36.78 | 15.97 | [Model](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-full.pth), [Log](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-full.log) |
| [Baseline-Full](configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof-full.py) | w/ complex | w/o complex | 39.26 | 18.86 |[Model](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-complex.pth), [Log](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-complex.log) |
| [Baseline-Full](configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof-full.py) | w/ complex | w/ complex | 39.21 | 18.84 |[Model](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-complex.pth), [Log](https://download.openmmlab.com/mim-example/embodiedscan/mv-grounding-complex.log) |

#### Multi-View Occupancy Prediction

| Method | Input | mIoU | Download |
|:------:|:-----:|:----:|:--------:|
| [Baseline](configs/occupancy/mv-occ_8xb1_embodiedscan-occ-80class.py) | RGB-D | 21.28 | [Log](https://download.openmmlab.com/mim-example/embodiedscan/mv-occ.log) |

#### Continuous Occupancy Prediction

| Method | Input | mIoU | Download |
|:------:|:-----:|:----:|:--------:|
| [Baseline](configs/occupancy/cont-occ_8xb1_embodiedscan-occ-80class.py) | RGB-D | 22.92 | [Log](https://download.openmmlab.com/mim-example/embodiedscan/cont-occ.log) |

Because the occupancy prediction models are a little large, we save them via OpenXLab and do not provide direct download links here. To download these checkpoints on OpenXLab, please run the following commands:

```bash
# If you did not install LFS before
git lfs install
# git clone EmbodiedScan model repo via
git clone https://code.openxlab.org.cn/wangtai/EmbodiedScan.git
# Then you can cd EmbodiedScan to get all the pretrained models
```

Please see the [paper](./assets/EmbodiedScan.pdf) for more details of our benchmarks. This dataset is still scaling up and the benchmark is being polished and extended. Please stay tuned for our recent updates.

## 📝 TODO List

- \[x\] Release the paper and partial codes for datasets.
- \[x\] Release EmbodiedScan annotation files.
- \[x\] Release partial codes for models and evaluation.
- \[ \] Polish dataset APIs and related codes.
- \[x\] Release Embodied Perceptron pretrained models.
- \[x\] Release multi-modal datasets and codes.
- \[x\] Release codes for our baselines and benchmarks.
- \[ \] Release codes for all the other methods.
- \[ \] Full release and further updates.
- \[ \] Release MMScan data and codes.

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{embodiedscan,
    title={EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI},
    author={Wang, Tai and Mao, Xiaohan and Zhu, Chenming and Xu, Runsen and Lyu, Ruiyuan and Li, Peisen and Chen, Xiao and Zhang, Wenwei and Chen, Kai and Xue, Tianfan and Liu, Xihui and Lu, Cewu and Lin, Dahua and Pang, Jiangmiao},
    year={2024},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
@inproceedings{mmscan,
    title={MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations},
    author={Lyu, Ruiyuan and Wang, Tai and Lin, Jingli and Yang, Shuai and Mao, Xiaohan and Chen, Yilun and Xu, Runsen and Huang, Haifeng and Zhu, Chenming and Lin, Dahua and Pang, Jiangmiao},
    year={2024},
    booktitle={arXiv},
}
```

If you use our dataset and benchmark, please kindly cite the original datasets involved in our work. BibTex entries are provided below.

<details><summary>Dataset BibTex</summary>

```BibTex
@inproceedings{dai2017scannet,
  title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
  author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
  booktitle = {Proceedings IEEE Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}
```

```BibTex
@inproceedings{Wald2019RIO,
  title={RIO: 3D Object Instance Re-Localization in Changing Indoor Environments},
  author={Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, Matthias Niessner},
  booktitle={Proceedings IEEE International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```

```BibTex
@article{Matterport3D,
  title={{Matterport3D}: Learning from {RGB-D} Data in Indoor Environments},
  author={Chang, Angel and Dai, Angela and Funkhouser, Thomas and Halber, Maciej and Niessner, Matthias and Savva, Manolis and Song, Shuran and Zeng, Andy and Zhang, Yinda},
  journal={International Conference on 3D Vision (3DV)},
  year={2017}
}
```

</details>

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements

- [OpenMMLab](https://github.com/open-mmlab): Our dataset code uses [MMEngine](https://github.com/open-mmlab/mmengine) and our model is built upon [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d): We use some functions supported in PyTorch3D for efficient computations on fundamental 3D data structures.
- [ScanNet](https://github.com/ScanNet/ScanNet), [3RScan](https://github.com/WaldJohannaU/3RScan), [Matterport3D](https://github.com/niessner/Matterport): Our dataset uses the raw data from these datasets.
- [ReferIt3D](https://github.com/referit3d/referit3d): We refer to the SR3D's approach to obtaining the language prompt annotations.
- [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS): Our annotation tool is developed based on the open-source framework used by SUSTechPOINTS.
