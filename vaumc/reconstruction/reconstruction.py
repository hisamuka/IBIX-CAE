import math

import numpy as np
from . import util


def reconstruct_image(img, autoencoder):
    if img.ndim not in [2, 3]:
        raise Exception(f'Invalid number of image dimensions: {img.ndim}. ' \
                        'Expected: 2 or 3')

    ysize, xsize = img.shape[0], img.shape[1]
    n_channels = img.shape[2] if img.ndim == 3 else 1
    norm_value = util.normalization_value(img)  # e.g., 255 for 8-bit images

    X = img.reshape(1, ysize, xsize, n_channels)
    X = util.normalize_image_set(X)
    X = pad_by_autoencoder_input(X, autoencoder)

    Xout = autoencoder.predict(X)  # shape (1, ysize, xsize, n_channels)
    Xout = util.crop(Xout, shape=(ysize, xsize))

    if n_channels == 1:
        img_out = Xout.reshape((ysize, xsize))
    else:
        img_out = Xout.reshape((ysize, xsize, n_channels))

    img_out *= norm_value
    img_out = img_out.astype(np.int32)

    return img_out


def reconstruct_image_set(X, autoencoder):
    # X.shape = (n_imgs, ysize, xsize) or (n_imgs, ysize, xsize, n_channels)

    if X.ndim not in [3, 4]:
        raise Exception(f'Invalid number of image array dimensions: {X.ndim}. ' \
                         'Expected: 3 (n_imgs, ysize, xsize) or 4 (n_imgs, ysize, xsize, n_channels)')

    if X.ndim == 3:
        new_shape = tuple(list(X.shape) + [1])
        X = np.reshape(X, new_shape)  # (n_imgs, ysize, xsize, 1)

    n_imgs, ysize, xsize, n_channels = X.shape

    norm_value = util.normalization_value(X)  # e.g., 255 for 8-bit images
    X = util.normalize_image_set(X)
    X = pad_by_autoencoder_input(X, autoencoder)

    Xout = autoencoder.predict(X)  # shape (1, ysize, xsize, n_channels)
    Xout = util.crop(Xout, shape=(ysize, xsize))

    if n_channels == 1:
        Xout = Xout.reshape((n_imgs, ysize, xsize))
    else:
        Xout = Xout.reshape((n_imgs, ysize, xsize, n_channels))

    Xout *= norm_value
    Xout = Xout.astype(np.int32)

    return Xout


def pad_by_downsampling_factors(X, downsampling_factors=(8, 8)):
    '''
    Adjust the image shape from a image data set (numpy array) with the
    downsampling factors of the encoding layer from an autoencoder:
    e.g., MaxPooling * MaxPooling * ...

    The function pads zero in the right and bottom of the images.

    Parameters
    ----------
    X: numpy array (n_imgs, ysize, ysize, n_channels)
        A numpy array representing a 2D image set.

    downsampling_factors: tuple (downsampling_at_ysize, downsampling_at_xsize)
        Tuple with the downsampling factors of the autoencoders.


    Returns
    -------
    numpy array (n_imgs, new_ysize, new_ysize, n_channels)
        A new numpy array after padding zeros.

    Raises
    ------
    Exception
        If the input numpy array does not have 4 dimensions:
        (n_imgs, new_ysize, new_ysize, n_channels).
    '''


    # padding zeros around a numpy array X (n_imgs, ysize, xsize, n_channels)
    # according to the x- and y-downsampling factors of the network
    # downsampling_factors = (on y, on x)

    if X.ndim != 4:
        raise Exception(f'Invalid number of dimensions {X.ndim} != 4 ' \
               '(n_imgs, ysize, xsize, n_channels)')

    _, ysize, xsize, _ = X.shape

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
    '''
    Adjust the image shape from a image data set (numpy array) with the input
    layer of an autoencoder by padding zeros.

    The function pads zero in the right and bottom of the images.

    Parameters
    ----------
    X: numpy array (n_imgs, ysize, ysize, n_channels)
        A numpy array representing a 2D image set.

    autoencoder
        An autoencoder model.


    Returns
    -------
    numpy array (n_imgs, new_ysize, new_ysize, n_channels)
        A new numpy array after padding zeros.

    Raises
    ------
    Exception
        If the input numpy array does not have 4 dimensions:
        (n_imgs, new_ysize, new_ysize, n_channels).

    Exception
        If the autoencoder's input layer does not have 4 dimensions.
    '''
    if X.ndim != 4:
        raise Exception(f'Invalid number of dimensions {X.ndim} != 4 ' \
               '(n_imgs, ysize, xsize, n_channels)')

    _, ysize, xsize, n_channels = X.shape

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
