import argparse
import os
import pdb
import sys

from keras.models import load_model
import numpy as np
from skimage import io

import util


def build_argparse():
    prog_desc = \
'''
Reconstruct an image by a trained autoencoder.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('image_path', type=str, help='Input Image.')
    parser.add_argument('autoencoder', type=str, help='AutoEncoder2D (*.h5).')
    parser.add_argument('output_image_path', type=str,
                        help='Output Image Path.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Input Image: %s' % args.image_path)
    print('- AutoEncoder2D: %s' % args.autoencoder)
    print('- Output Image Path: %s' % args.output_image_path)
    print('--------------------------------------------\n')


def validate_args(args):
    if not args.autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder2D extension: %s\nTry *.h5' %
                 args.autoencoder)

    parent_dir = os.path.dirname(args.output_image_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    print('- Loading Input Image')
    img_paths = [args.image_path]
    X = util.read_image_list(img_paths)  # (1, ysize, xsize, n_channels)
    norm_value = util.normalization_value(X[0])  # e.g., 255 for 8-bit images
    X = util.normalize_image_set(X)

    print('- Loading AutoEncoder2D')
    autoencoder = load_model(args.autoencoder)

    print('- Padding according to Models\'s Input Shape')
    ysize, xsize, n_channels = X.shape[1:]
    X = util.pad_by_autoencoder_input(X, autoencoder)

    print('- Reconstructing Image')
    Xout = autoencoder.predict(X) # shape (1, ysize, xsize, n_channels)
    Xout = util.crop(Xout, shape=(ysize, xsize))

    if n_channels == 1:
        img = Xout.reshape((ysize, xsize))
    else:
        img = Xout.reshape((ysize, xsize, n_channels))
    
    print('- Normalizing Image: float to int')
    img *= norm_value
    img = img.astype(np.int32)

    print('- Saving Reconstruct Image')
    io.imsave(args.output_image_path, img)

    print('\n- Done...')



if __name__ == '__main__':
    main()
