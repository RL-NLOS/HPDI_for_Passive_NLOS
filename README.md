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
