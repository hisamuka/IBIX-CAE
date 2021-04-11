# VA-UMC
Visual Analytics tool for a collaboration with the UMC (Netherlands).

## installation


## datasets
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


## generative neural networks


## visual analytics tool

