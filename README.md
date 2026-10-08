# Neural networks meet light transport physics for passive non-line-of-sight imaging enhancement

## Introduction

This repository provides the official PyTorch implementation of **"Neural networks meet light transport physics for passive non-line-of-sight imaging enhancement"**. The datasets are available on [Zenodo](https://doi.org/10.5281/zenodo.23050462). If you find this work useful, please give this repository a star ⭐ and consider citing our paper. Thank you!

## Datasets

**Download:** [Zenodo dataset release](https://doi.org/10.5281/zenodo.23050462)  
**DOI:** `10.5281/zenodo.23050462`

The dataset collection contains six datasets: **Anime**, **NIST**, **Quickdraw**, **SHAPES**, **STL10**, and **SuperModel**. Each dataset contains paired images stored in `raw/` and `label/`. All images are provided in BMP (`.bmp`) format.

### Dataset Sizes

Counts below refer to image pairs. Each pair consists of one image in `raw/` and its corresponding image in `label/`.

| Dataset | Training pairs | Validation pairs | Test pairs | Additional test pairs |
| --- | --- | --- | --- | --- |
| Anime | 36,000 | 2,000 | 2,000 | — |
| NIST | 600 / 1,200 / 2,400 / 4,800 / 7,200 / 9,600 / 12,000 | 2,000 | 2,000 | 2,600 (`test2`) |
| Quickdraw | 1,080 / 2,160 / 4,320 / 8,640 / 12,960 / 17,280 / 21,600 | 3,600 | 3,600 | — |
| SHAPES | 320 / 640 / 1,280 / 2,560 / 3,840 / 5,120 / 6,400 | 1,024 | 1,024 | — |
| STL10 | 36,000 | 2,000 | 2,000 | — |
| SuperModel | 8,000 | 1,000 | 1,000 | — |

NIST, Quickdraw, and SHAPES provide multiple training set sizes. Each size corresponds to a separate `train_N/` directory and its matching `train_label_N/` directory.

### Directory Structure

`<dataset_root>` denotes the parent directory containing the datasets. Comments indicate the number of BMP images in each directory.

<details>
<summary>Click to expand the complete directory structure</summary>

```text
<dataset_root>/
├── Anime_img/
│   ├── raw/
│   │   ├── train_36000/            # 36,000 images
│   │   ├── val/                    # 2,000 images
│   │   └── test/                   # 2,000 images
│   └── label/
│       ├── train_label_36000/       # 36,000 images
│       ├── val_label/              # 2,000 images
│       └── test_label/             # 2,000 images
├── NIST_img/
│   ├── raw/
│   │   ├── train_600/              # 600 images (MNIST)
│   │   ├── train_1200/             # 1,200 images (MNIST)
│   │   ├── train_2400/             # 2,400 images (MNIST)
│   │   ├── train_4800/             # 4,800 images (MNIST)
│   │   ├── train_7200/             # 7,200 images (MNIST)
│   │   ├── train_9600/             # 9,600 images (MNIST)
│   │   ├── train_12000/            # 12,000 images (MNIST)
│   │   ├── val/                    # 2,000 images (MNIST)
│   │   ├── test/                   # 2,000 images (MNIST)
│   │   └── test2/                  # 2,600 images (EMNIST)
│   └── label/
│       ├── train_label_600/         # 600 images (MNIST)
│       ├── train_label_1200/        # 1,200 images (MNIST)
│       ├── train_label_2400/        # 2,400 images (MNIST)
│       ├── train_label_4800/        # 4,800 images (MNIST)
│       ├── train_label_7200/        # 7,200 images (MNIST)
│       ├── train_label_9600/        # 9,600 images (MNIST)
│       ├── train_label_12000/       # 12,000 images (MNIST)
│       ├── val_label/              # 2,000 images (MNIST)
│       ├── test_label/             # 2,000 images (MNIST)
│       └── test2_label/            # 2,600 images (EMNIST)
├── Quickdraw_img/
│   ├── raw/
│   │   ├── train_1080/             # 1,080 images
│   │   ├── train_2160/             # 2,160 images
│   │   ├── train_4320/             # 4,320 images
│   │   ├── train_8640/             # 8,640 images
│   │   ├── train_12960/            # 12,960 images
│   │   ├── train_17280/            # 17,280 images
│   │   ├── train_21600/            # 21,600 images
│   │   ├── val/                    # 3,600 images
│   │   └── test/                   # 3,600 images
│   └── label/
│       ├── train_label_1080/        # 1,080 images
│       ├── train_label_2160/        # 2,160 images
│       ├── train_label_4320/        # 4,320 images
│       ├── train_label_8640/        # 8,640 images
│       ├── train_label_12960/       # 12,960 images
│       ├── train_label_17280/       # 17,280 images
│       ├── train_label_21600/       # 21,600 images
│       ├── val_label/              # 3,600 images
│       └── test_label/             # 3,600 images
├── SHAPES_img/
│   ├── raw/
│   │   ├── train_320/              # 320 images
│   │   ├── train_640/              # 640 images
│   │   ├── train_1280/             # 1,280 images
│   │   ├── train_2560/             # 2,560 images
│   │   ├── train_3840/             # 3,840 images
│   │   ├── train_5120/             # 5,120 images
│   │   ├── train_6400/             # 6,400 images
│   │   ├── val/                    # 1,024 images
│   │   └── test/                   # 1,024 images
│   └── label/
│       ├── train_label_320/         # 320 images
│       ├── train_label_640/         # 640 images
│       ├── train_label_1280/        # 1,280 images
│       ├── train_label_2560/        # 2,560 images
│       ├── train_label_3840/        # 3,840 images
│       ├── train_label_5120/        # 5,120 images
│       ├── train_label_6400/        # 6,400 images
│       ├── val_label/              # 1,024 images
│       └── test_label/             # 1,024 images
├── STL10_img/
│   ├── raw/
│   │   ├── train_36000/            # 36,000 images
│   │   ├── val/                    # 2,000 images
│   │   └── test/                   # 2,000 images
│   └── label/
│       ├── train_label_36000/       # 36,000 images
│       ├── val_label/              # 2,000 images
│       └── test_label/             # 2,000 images
├── SuperModel_img/
│   ├── raw/
│   │   ├── train_8000/             # 8,000 images
│   │   ├── val/                    # 1,000 images
│   │   └── test/                   # 1,000 images
│   └── label/
│       ├── train_label_8000/        # 8,000 images
│       ├── val_label/              # 1,000 images
│       └── test_label/             # 1,000 images
```

</details>

### Image Pairing

- `raw/train_N/` and `label/train_label_N/`: training images and their corresponding labels.
- `raw/val/` and `label/val_label/`: validation images and their corresponding labels.
- `raw/test/` and `label/test_label/`: test images and their corresponding labels.
- `raw/test2/` and `label/test2_label/`: the additional EMNIST dataset for out-of-distribution (OOD) generalization testing.

Images in each directory use consecutive five-digit filenames starting from `00000.bmp`. Images and labels are matched by filename within the corresponding split. For example:

```text
Anime_img/raw/train_36000/00000.bmp
Anime_img/label/train_label_36000/00000.bmp
```

## Code

HPDI combines a light-transport-based reconstruction path with an implicit reconstruction path. FISTA first produces coarse reconstructions from the RAW measurements. The two reconstruction networks are trained separately, then jointly optimized with the fusion network.

| Paper module | Python class | Input and role |
| --- | --- | --- |
| Refinement Network (RN) | `RefinementNet` | Refines the coarse images reconstructed by FISTA |
| Implicit Reconstruction Network (IRN) | `ImplicitReconstructionNet` | Learns reconstruction directly from RAW measurements |
| Fusion Network (FN) | `FusionNet` | Fuses four scales of IRN and RN features to predict the final reconstruction |

RN and IRN share the same U-shaped architecture but have independent weights. The fusion blocks are implemented by `AdaptiveFeatureSelection` and `CrossAttentionFusion`.

The complete reproduction workflow is:

**FISTA preprocessing → separate IRN and RN training → joint IRN/RN/FN training → evaluation.**

### Repository Structure

```text
HPDI_for_Passive_NLOS/
├── HPDI_fista.py                    # Generate coarse reconstruction (CTF) images
├── HPDI_train.py                    # Train IRN or RN separately
├── HPDI_Fusion_train.py             # Jointly train IRN, RN, and FN
├── HPDI_test.py                     # Evaluate one separately trained branch
├── HPDI_Fusion_test.py              # Evaluate the jointly trained system
├── hpdi/
│   ├── algorithms/fista.py          # Reusable FISTA solver
│   ├── transport_matrix/
│   │   └── real_trans_matrix.pt     # Supplied light transport matrix
│   ├── models/
│   │   ├── reconstruction.py        # RN and IRN architectures
│   │   └── fusion.py                # FN and its fusion components
│   ├── data/                       # BMP datasets and data loaders
│   ├── options/                    # Shared argparse configuration
│   ├── reconstruction.py           # FISTA preprocessing workflow
│   ├── training.py                 # Separate and joint training workflows
│   ├── evaluation.py               # Inference, image saving, and PSNR
│   ├── runtime.py                  # Device setup and experiment records
│   └── metrics.py                  # Image quality metrics
├── requirements.txt
├── pyproject.toml
├── MANIFEST.in
├── CITATION.cff
└── README.md
```

### Installation

The package supports Python 3.9–3.12. Install a matching PyTorch and torchvision build for your Python version and CPU/CUDA environment, then install the remaining dependencies.

```bash
git clone https://github.com/RL-NLOS/HPDI_for_Passive_NLOS.git
cd HPDI_for_Passive_NLOS
python -m pip install -r requirements.txt
```

To use the package and its command-line entry points, install it in editable mode:

```bash
python -m pip install -e .
```

The installed commands `hpdi-fista`, `hpdi-train`, `hpdi-fusion-train`, `hpdi-test`, and `hpdi-fusion-test` accept the same arguments as their corresponding Python scripts. The transport matrix is included in the package; its default path is resolved independently of the working directory.

Run the commands below from the repository root. They use `--device auto`, which selects CUDA when available and otherwise uses the CPU. Use `--device cuda:0`, `--device cuda:1`, or `--device cpu` to select a device explicitly.

### Preparing a Training Subset

The downloaded datasets use `train_N/` and `train_label_N/` to distinguish training set sizes. The training code expects a selected subset under `train/` and `train_label/`.

Choose one training size and create the following working layout. For example, to use the 6,400-pair SHAPES subset, copy `SHAPES_img/raw/train_6400/` to `data/HPDI/SHAPES_img/raw/train/`, and copy `SHAPES_img/label/train_label_6400/` to `data/HPDI/SHAPES_img/label/train_label/`. Copy the corresponding validation and test folders to the locations below. Keep the downloaded collection if you plan to compare multiple training sizes.

```text
data/HPDI/SHAPES_img/
├── raw/
│   ├── train/*.bmp
│   ├── val/*.bmp
│   └── test/*.bmp
└── label/
    ├── train_label/*.bmp
    ├── val_label/*.bmp
    └── test_label/*.bmp
```

`--data-root` points to the parent of the dataset folders, such as `data/HPDI`, rather than to `SHAPES_img` itself. Use the same dataset name and working subset in every stage. Select one `train_N/` subset per experiment to retain the chosen training-set size.

Images and labels are paired by independently sorted filenames. Keep the same sample order and matching image counts; matching filenames are recommended. The loaders preserve the BMP image mode and use `ToTensor` without resizing or data augmentation.

For the supplied matrix, use 32×32 measurements and reconstruction targets. `Anime_img`, `SHAPES_img`, `STL10_img`, and `SuperModel_img` default to three channels; other dataset names default to one channel. This automatic choice is based on the folder name. Set `--channels 1` or `--channels 3` explicitly when the actual image mode differs from the default, and use the same value for separate training, joint training, and evaluation.

The additional NIST `test2/` split is not selected by the default evaluation commands. To evaluate it, create a separate working dataset whose `raw/test/` and `label/test_label/` contain the `test2/` measurements and `test2_label/` labels, generate its `ctf/test/` with `--split test`, and use the checkpoints trained on the original NIST training subset.

### Light Transport Matrix

The supplied file `hpdi/transport_matrix/real_trans_matrix.pt` contains a `1024 × 1024` float64 tensor. Its shape follows `y = T x`: rows index measurement pixels and columns index reconstruction pixels. FISTA uses the original float32 conversion and a default matrix scale of 1. Each channel is flattened independently in PyTorch's row-major order.

The matrix must match the imaging geometry and measurement preprocessing. For another experimental configuration, specify its matrix with `--matrix-path` and the corresponding reconstruction size with `--height` and `--width`. The reconstruction pixel count must equal the number of matrix columns, and each measurement channel must have as many pixels as there are matrix rows. Network input and target sizes must be compatible with the architecture, including spatial dimensions divisible by 8.

### FISTA Preprocessing

Generate coarse reconstructions for the training, validation, and test splits:

```bash
python HPDI_fista.py --data-root data/HPDI --dataset-name SHAPES_img --device auto
```

This command uses the supplied matrix and defaults to `--split all`. It creates:

```text
data/HPDI/SHAPES_img/ctf/
├── train/*.bmp
├── val/*.bmp
└── test/*.bmp
```

The CTF images retain the RAW filenames and channel count, so grayscale measurements produce single-channel CTF images. This step needs RAW measurements and the transport matrix; it does not use labels.

| FISTA parameter | Default |
| --- | --- |
| `--lambda` | `1e-4` |
| `--max-iter` | `200` |
| `--tol` | `1e-6` |
| `--matrix-scale` | `1.0` |
| `--height`, `--width` | `32`, `32` |
| `--stopping-rule` | `legacy` |

The matrix transpose and exact Lipschitz constant are computed once and reused across all splits, samples, and channels. The `legacy` stopping rule preserves the original early-return behavior; `latest` explicitly selects the new iterate on convergence.

For one split, use `--split train`, `--split val`, or `--split test`. For custom folders, `--input-dir` and `--output-dir` refer to image directories for a single split, or to their parent directories for `--split all`.

### Separate Training

Train the IRN on RAW measurements and the RN on FISTA reconstructions:

```bash
python HPDI_train.py --data-root data/HPDI --dataset-name SHAPES_img --input-type raw --device auto
python HPDI_train.py --data-root data/HPDI --dataset-name SHAPES_img --input-type ctf --device auto
```

The first command uses `raw/train` and `label/train_label`; the second uses `ctf/train` and the same labels. Validation uses each branch's `val/` images and `label/val_label/`.

The network role is inferred from the input folder: `raw` selects IRN and `ctf` selects RN. For a custom coarse-image folder, specify the role explicitly, for example `--input-type coarse --network rn`.

The best checkpoint for each branch is selected by validation BCE loss and saved under `data/HPDI/SHAPES_img/model/`:

| Checkpoint | Network |
| --- | --- |
| `raw_best.pkl` | Separately trained IRN |
| `ctf_best.pkl` | Separately trained RN |

### Joint Training

After both branch checkpoints are available, jointly train IRN, RN, and FN:

```bash
python HPDI_Fusion_train.py --data-root data/HPDI --dataset-name SHAPES_img --device auto
```

The command loads `raw_best.pkl` and `ctf_best.pkl`. FN starts from random initialization and combines the multi-scale features in IRN-then-RN order. The final reconstruction is supervised by the ground truth, and gradients update all three networks.

The three checkpoints from the same best validation epoch are saved together:

```text
data/HPDI/SHAPES_img/model/
├── joint_raw_best.pkl
├── joint_ctf_best.pkl
└── joint_fusion_best.pkl
```

Use `--raw-checkpoint` and `--ctf-checkpoint` to select branch weights from another location. `--model-dir` changes the checkpoint directory; `--name` changes the save prefix. When using custom names, provide the corresponding checkpoint paths during evaluation.

## Citation

If you use the code or datasets in your research, please cite the paper:

```bibtex
@article{liang2026hpdi,
  author  = {Liang, Rui and Xu, Zhenjun and Tong, Xi and Yang, Jiangxin and Li, Xin and Cao, Yanpeng},
  title   = {Neural Networks Meet Light Transport Physics for Passive Non-Line-of-Sight Imaging Enhancement},
  journal = {IEEE Transactions on Computational Imaging},
  volume  = {12},
  pages   = {282--296},
  year    = {2026},
  doi     = {10.1109/TCI.2026.3653304}
}
```

The repository also provides `CITATION.cff` with this paper as the preferred citation.
