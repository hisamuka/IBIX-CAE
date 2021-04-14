import argparse
import os
import pdb
import sys

from keras.models import load_model
import numpy as np
from skimage import io

from . import util


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



def reconstruct_image(img, autoencoder, verbose=False):
    if img.ndim not in [2, 3]:
        raise Exception(f'Invalid number of imagem dimensions: {img.ndim}. ' \
                         'Expected: 2 or 3')
    
    ysize, xsize = img.shape[0], img.shape[1]
    n_channels = img.shape[2] if img.ndim == 3 else 1
    norm_value = util.normalization_value(img)  # e.g., 255 for 8-bit images

    X = img.reshape(1, ysize, xsize, n_channels)
    X = util.normalize_image_set(X)
    X = util.pad_by_autoencoder_input(X, autoencoder)

    Xout = autoencoder.predict(X) # shape (1, ysize, xsize, n_channels)
    Xout = util.crop(Xout, shape=(ysize, xsize))

    if n_channels == 1:
        img_out = Xout.reshape((ysize, xsize))
    else:
        img_out = Xout.reshape((ysize, xsize, n_channels))
    
    img_out *= norm_value
    img_out = img_out.astype(np.int32)

    return img_out


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)


    print('- Loading Input Image')
    img = io.imread(args.image_path)

    print('- Loading AutoEncoder2D')
    autoencoder = load_model(args.autoencoder)

    print('- Reconstructing Image')
    img_out = reconstruct_image(img, autoencoder)

    print('- Saving Reconstruct Image')
    io.imsave(args.output_image_path, img_out)

    print('\n- Done...')



if __name__ == '__main__':
    main()
