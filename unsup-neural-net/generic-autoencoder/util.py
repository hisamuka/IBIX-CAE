import csv
import math
import os
import pdb
import sys

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



def pad_by_downsampling_factors(X, downsampling_factors=(8, 8)):
    # padding zeros around a numpy array X (n_imgs, ysize, xsize, n_channels)
    # according to the x- and y-downsampling factors of the network
    # downsampling_factors = (on y, on x)
    
    if X.ndim != 4:
        raise Exception(f'Invalid number of dimensions {X.ndim} != 4 ' \
               '(n_imgs, ysize, xsize, n_channels)')
    
    n_imgs, ysize, xsize, n_channels = X.shape
    
    new_ysize = math.ceil(ysize / downsampling_factors[0]) * downsampling_factors[0]
    new_xsize = math.ceil(xsize / downsampling_factors[1]) * downsampling_factors[1]

    if (ysize, xsize) == (new_ysize, new_xsize):
        return X
    else:
        offset_y = new_ysize - ysize
        offset_x = new_xsize - xsize

        Xnew = np.pad(X, [(0, 0), (0, offset_y), (0, offset_x), (0, 0)])
        
        return Xnew


def pad_by_autoencoder_input(X, autoencoder):
    # X.shape ==> (n_imgs, ysize, xsize, n_channels)
    if X.ndim != 4:
        raise Exception(f'Invalid number of dimensions {X.ndim} != 4 ' \
               '(n_imgs, ysize, xsize, n_channels)')
    
    n_imgs, ysize, xsize, n_channels = X.shape

    # it returns something like: (None, ysize, xsize, n_channels)
    input_shape = autoencoder.layers[0].get_input_shape_at(0)
    
    if len(input_shape) != 4:
        raise Exception(f'Invalid number of dimensions for autoencoder input: ' \
                        f'{len(input_shape)} != 4 (n_imgs, ysize, xsize, n_channels)')


    
    if (ysize, xsize, n_channels) == input_shape[1:]:
        return X
    else:
        new_ysize, new_xsize = input_shape[1], input_shape[2]

        offset_y = new_ysize - ysize
        offset_x = new_xsize - xsize

        Xnew = np.pad(X, [(0, 0), (0, offset_y), (0, offset_x), (0, 0)])
        
        return Xnew

        # Xnew = np.zeros((n_imgs, new_ysize, new_xsize, n_channels),
        #                 dtype=X.dtype)
        # Xnew[:, :ysize, :xsize, :] = X

        # return Xnew


def pad_by_autoencoder_input(X, autoencoder):
    # X.shape ==> (n_imgs, ysize, xsize, n_channels)
    if X.ndim != 4:
        raise Exception(f'Invalid number of dimensions {X.ndim} != 4 ' \
               '(n_imgs, ysize, xsize, n_channels)')
    
    n_imgs, ysize, xsize, n_channels = X.shape

    # it returns something like: (None, ysize, xsize, n_channels)
    input_shape = autoencoder.layers[0].get_input_shape_at(0)
    
    if len(input_shape) != 4:
        raise Exception(f'Invalid number of dimensions for autoencoder input: ' \
                        f'{len(input_shape)} != 4 (n_imgs, ysize, xsize, n_channels)')


    
    if (ysize, xsize, n_channels) == input_shape[1:]:
        return X
    else:
        new_ysize, new_xsize = input_shape[1], input_shape[2]

        Xnew = np.zeros((n_imgs, new_ysize, new_xsize, n_channels),
                        dtype=X.dtype)
        Xnew[:, :ysize, :xsize, :] = X

        return Xnew



def crop(X, shape):
    if X.ndim != 4:
        raise Exception(f'Invalid number of dimensions {X.ndim} != 4 ' \
               '(n_imgs, ysize, xsize, n_channels)')
    
    ysize, xsize = shape

    return X[:, :ysize, :xsize, :]

