import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def read_image_entry(img_entry):
    img_list = []
    if os.path.isdir(img_entry):
        img_list = [os.path.join(img_entry, img_path) for img_path in
                    sorted(os.listdir(img_entry))]
    elif img_entry.endswith(".csv"):
        with open(img_entry) as f:
            img_list = f.read().splitlines()

    return img_list


def read_image_list(img_list):
    X = []

    for img_path in img_list:
        img = io.imread(img_path)
        X.append(img)

    X = np.array(X)
    if len(X.shape) == 3: # gray images (n_imgs, ysize, xsize)
        new_shape = tuple(list(X.shape) + [1])
        X = np.reshape(X, new_shape)

    return X

def normalization_value(img):
    # Given that the image formats are only 8, 12, 16, 32, and 64 bits, we must impose this constraint here.
    n_bits = math.floor(math.log2(img.max()))
    
    if n_bits > 1 and n_bits < 8:
        n_bits = 8
    elif n_bits > 8 and n_bits < 12:
        n_bits = 12
    elif n_bits > 12 and n_bits < 16:
        n_bits = 16
    elif n_bits > 16 and n_bits < 32:
        n_bits = 32
    elif n_bits > 32 and n_bits < 64:
        n_bits = 64
    elif n_bits > 64:
        sys.exit("Error: Number of Bits %d not supported. Try <= 64" % n_bits)

    return math.pow(2, n_bits) - 1


def normalize_image_set(X):
    # convert the int images to float and normalize them to [0, 1],
    # acoording to its normalization value = (2^b) - 1, where b is the image depth (bits)
    norm_val = normalization_value(X[0])
    return X.astype('float32') / norm_val


def crop(X, shape):
    if X.ndim != 4:
        raise Exception(f'Invalid number of dimensions {X.ndim} != 4 ' \
               '(n_imgs, ysize, xsize, n_channels)')
    
    ysize, xsize = shape

    return X[:, :ysize, :xsize, :]


def get_colors(cmap_name, n_colors):
    list(plt.get_cmap(cmap_name, n_colors).colors)


def mix_image_heatmap(img, heatmap, cmap_name):
    max_range = normalization_value(img)  # e.g., 255
    n_colors = max_range + 1  # e.g., 255 + 1

    print(heatmap)
    print(max_range)
