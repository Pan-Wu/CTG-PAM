# README

## Project Overview

This repository contains code and demo data for the paper "Wu, R., Chen, Z., Yu, J. et al. A graph-learning based model for automatic diagnosis of Sjögren’s syndrome on digital pathological images: a multicentre cohort study. J Transl Med 22, 748 (2024). https://doi.org/10.1186/s12967-024-05550-8".

## Directory Structure

```
├── cell_tissue_graph.py
├── classification.py  
├── detector.py
├── utils.py
├── model.py
├── models
├── dataset
│   ├── sample
│   │   ├── cc_adjust_feature
│   │   │   └── cc_adjust_feature_cth_50_n_20.npy
│   │   ├── cc_dist.npy
│   │   ├── cell_feature.npy
│   │   ├── cell_groups.png
│   │   ├── cell_label.npy
│   │   ├── cell_map.npy
│   │   ├── cell_predict.png
│   │   ├── ct_adjust_feature
│   │   │   └── ct_adjust_feature_tth_5_n_2.npy
│   │   ├── ct_dist.npy
│   │   ├── gla_mask.png
│   │   ├── img.png
│   │   ├── lym_mask.png
│   │   ├── tissue_feature.npy
│   └── └── tissue_map.npy
└── train_dataset_demo
```

## Main Components

### Code

- **cell_tissue_graph.py**: Main code for CTG-PAM. This script is used to perform the diagnosis of SS by CTG-PAM, executing different steps based on the parameter settings: `pre`, `single`, `ctg`, `detect`, and `all`. The `all` parameter executes all steps in sequence, from preprocessing (detection of cells and tissue structures) to the final diagnosis of SS. For example, to perform lymphocyte recognition and SS diagnosis, use the following command:
  ```bash
  python cell_tissue_graph.py detect
  ```

- **classification.py**: Used for training and testing the neural network for cell classification in CTG-PAM. An example of a training command:
  ```bash
  python classification.py train 50 50 50 0.001 1 20 2 50 5 ctg
  ```
  Here, the parameters after `train` specify the network width, batch size, number of epochs, learning rate, GPU index, and feature evolution hyperparameters. The last parameter, `ctg`, indicates the use of CTG features. Changing `ctg` to `ccg` uses only CCG features. An example of a testing command:
  ```bash
  python classification.py test 50 1 20 2 50 5 ctg
  ```
  The parameters after `test` specify the network width, GPU index, and other feature evolution hyperparameters.

- Other scripts (`detector.py`, `utils.py`, `model.py`) are auxiliary and not intended for direct execution.

### Dataset

The repository contains two data folders: `dataset` and `train_dataset_demo`.

In the `dataset` folder, we provide several pathological image samples for testing the open-source code. Samples named with `sample_with_cd45`, includes IHC results for comparison, while other samples provide partial cell type annotations for performance evaluation. 

- `img.png` is the pathological slice image.
- `gla_mask.png` and `lym_mask.png` are partial annotations for gland cells and lymphocytes, respectively.
- Ensure these data files exist before running the code.
- In `sample_with_cd45`, `cd45.png` provides IHC results. This sample's original image and results are also displayed in the paper.

In addition to images and annotations, we provide intermediate data and visualization results obtained after running the code, allowing you to review results at any stage without executing the code.

- `cc_adjust_feature`: Folder for cell-cell feature evolution results with different hyperparameters. We provide an example feature with `$n_{CCG}=20, d_{CCG}=50$`, named `cc_adjust_feature_cth_50_n_20.npy`.
- `ct_adjust_feature`: Folder for cell-tissue feature evolution results with different hyperparameters. We provide an example feature with `$n_{CTG}=2, d_{CTG}=5$`, named `ct_adjust_feature_tth_5_n_2.npy`.
- `cell_map.npy` and `tissue_map.npy`: Intermediate data storing results of cell and tissue structure detection.
- `cell_feature.npy` and `tissue_feature.npy`: Intermediate data storing single-cell and single-tissue features.
- `cc_dist.npy` and `ct_dist.npy`: Intermediate data storing distances between cells and between cells and tissue structures.
- `cell_label.npy`: Intermediate data storing cell labels generated from annotations.
- `cell_predict.png`: Visualization of cell classification results, with lymphocytes marked in red.
- `cell_groups.png`: Visualization of lymphocyte group detection results, marking lymphocyte groups with more than 50 cells in red. If no such groups are detected, the sample is diagnosed as not having SS.

### Training Data

- **train_dataset_demo**: Example training dataset for demonstration purposes. You can use this dataset to test the neural network training process in `classification.py`. The trained models will be saved in the `models` folder.

### Models

We provide a pre-trained CTG-PAM model file, `models/ctg_pam.pth`. Running `cell_tissue_graph.py` by default uses this pre-trained model. Additionally, we provide a model file that does not use cell-tissue features, `models/ccg.pth`. You can use this model by modifying `model_path` in `cell_tissue_graph.py` and changing the `feature_type` parameter in the `cell_classification` function from `ctg` to `ccg`.

## Libraries requirements

```
scikit-image==0.24.0
torch==2.3.1+cu118
segment-anything==1.0
stardist==0.9.1
tensorflow==2.13.1
networkx==3.0
```

