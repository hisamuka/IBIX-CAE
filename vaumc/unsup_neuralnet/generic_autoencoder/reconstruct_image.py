import argparse
import os
import sys

from keras.models import load_model
from skimage import io

from reconstruction import reconstruct_image


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
