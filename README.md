# VA-UMC
Visual Analytics tool for a collaboration with the UMC (Netherlands).

## Installation

##### Required python packages
```
pip install keras
pip install napari
pip install numpy
pip install scikit-image
pip install tensorflow # or install the gpu version
pip install tensorflow-gpu
```

To install TensorFlow FPU on Ubuntu 20.04: https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d


## Datasets
Datasets used for testing and evaluation: folder `datasets`.

#### 1. CamCan_axial
Set of control 2D axial slices from the [CamCan428 dataset](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/). CamCan provides a pair of MRI-T1 and -T2 volumetric scans for 653 control subjects.

We considered the following steps to generate the 2D axial slices:
- Noise filtering (median filter)
- Bias Field Correction by N4
- Affine registration with the MNI Template
- Histogram matching with the MNI template
- Extraction of the mid-axial slice of each image.
- Intensity scale to uint8 ([0, 255])

##### Training and Testing sets
- `./datasets/CamCan_axial/train_T1.csv`: Training set with the first 90% T1 images (pathnames)
- `./datasets/CamCan_axial/test_T1.csv`: Test set with the remaining 10% T1 images (pathnames)
- `./datasets/CamCan_axial/train_T2.csv`: Training set with the first 90% T2 images (pathnames)
- `./datasets/CamCan_axial/test_T2.csv`: Test set with the remaining 10% T2 images (pathnames)

## Unsupervised Neural Networks
#### Training an Autoencoder
###### Usage: `python unsup-neural-net/generic-autoencoder/train_autoencoder.py -h`
###### Example:
`python unsup-neural-net/generic-autoencoder/train_autoencoder.py datasets/CamCan_axial/T1 datasets/CamCan_axial/T2 models/autoencoder_t1_to_t2.h5 -b 32 -e 100`

#### Reconstruct an Image
###### Usage: `python unsup-neural-net/generic-autoencoder/reconstruct_image.py -h`
###### Example:
`python unsup-neural-net/generic-autoencoder/reconstruct_image.py datasets/CamCan_axial/T1/000600_000001.png models/autoencoder_t1_to_t2.h5 out/000600_000001.png`

#### Reconstruct an Image Set
###### Usage: `python unsup-neural-net/generic-autoencoder/reconstruct_image_set.py -h`
###### Example:
`python unsup-neural-net/generic-autoencoder/reconstruct_image_set.py datasets/CamCan_axial/test_T1.csv models/autoencoder_t1_to_t2.h5 out`


## Available Pretrained Models
- `models/autoencoder_t1_to_t2.h5`
    - Model to reconstruct T2 images (axial slices) from T1 ones;
    - Trained with the sets:
        - `datasets/CamCan_axial/T1`
        - `datasets/CamCan_axial/T2`
    - **Epochs**: 100, **batch size**: 32
    - **Final loss (MAE)**: 0.0068


## visual analytics tool

