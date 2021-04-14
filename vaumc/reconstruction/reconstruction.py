import numpy as np
from . import util


def reconstruct_image(img, autoencoder):
    print('-------------------------------')
    if img.ndim not in [2, 3]:
        raise Exception(f'Invalid number of image dimensions: {img.ndim}. ' \
                        'Expected: 2 or 3')

    print('[1]')
    ysize, xsize = img.shape[0], img.shape[1]
    print('[2]')
    n_channels = img.shape[2] if img.ndim == 3 else 1
    print('[3]')
    norm_value = util.normalization_value(img)  # e.g., 255 for 8-bit images

    print('[4]')
    X = img.reshape(1, ysize, xsize, n_channels)
    print('[5]')
    X = util.normalize_image_set(X)
    print('[6]')
    X = util.pad_by_autoencoder_input(X, autoencoder)
    print('[7]')

    Xout = autoencoder.predict(X)  # shape (1, ysize, xsize, n_channels)
    print('[8]')
    Xout = util.crop(Xout, shape=(ysize, xsize))
    print('[9]')

    if n_channels == 1:
        img_out = Xout.reshape((ysize, xsize))
    else:
        img_out = Xout.reshape((ysize, xsize, n_channels))
    print('[10]')

    img_out *= norm_value
    print('[11]')
    img_out = img_out.astype(np.int32)
    print('[12]')

    return img_out


def reconstruct_image_set(X, autoencoder):
    # X.shape = (n_imgs, ysize, xsize) or (n_imgs, ysize, xsize, n_channels)

    if X.ndim not in [3, 4]:
        raise Exception(f'Invalid number of image array dimensions: {X.ndim}. ' \
                         'Expected: 3 (n_imgs, ysize, xsize) or 4 (n_imgs, ysize, xsize, n_channels)')

    if X.ndim == 3:
        X = np.reshape(tuple(X.shape) + [1])  # (n_imgs, ysize, xsize, 1)

    n_imgs, ysize, xsize, n_channels = X.shape

    norm_value = util.normalization_value(X)  # e.g., 255 for 8-bit images
    X = util.normalize_image_set(X)
    X = util.pad_by_autoencoder_input(X, autoencoder)

    Xout = autoencoder.predict(X)  # shape (1, ysize, xsize, n_channels)
    Xout = util.crop(Xout, shape=(ysize, xsize))

    if n_channels == 1:
        Xout = Xout.reshape((n_imgs, ysize, xsize))
    else:
        Xout = Xout.reshape((n_imgs, ysize, xsize, n_channels))

    Xout *= norm_value
    Xout = Xout.astype(np.int32)

    return Xout

