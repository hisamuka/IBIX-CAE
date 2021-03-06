import argparse
import os
import sys

from keras.models import load_model
from skimage import io

from reconstruction.reconstruction import reconstruct_image_set
from reconstruction import util


def build_argparse():
    prog_desc = \
'''
Reconstruct a set of images by a trained autoencoder.

The output reconstructed images will be saved into the output directory with the same filename of their
original images (before feature extraction).
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('image_entry', type=str,
                        help='Directory or CSV file with image pathnames.')
    parser.add_argument('autoencoder', type=str, help='AutoEncoder2D (*.h5).')
    parser.add_argument('output_dir', type=str,
                        help='Output Directory where the reconstructed images will be saved.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Image entry: %s' % args.image_entry)    
    print('- AutoEncoder2D: %s' % args.autoencoder)
    print('- Output Directory: %s' % args.output_dir)
    print('--------------------------------------------\n')


def validate_args(args):
    if not os.path.isdir(args.image_entry) and not args.image_entry.endswith('.csv'):
        sys.exit('Invalid image entry: %s\nTry a CSV or a Directory' % args.image_entry)

    if not args.autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder2D extension: %s\nTry *.h5' %
                 args.autoencoder)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    print('- Loading Image set')
    img_paths = util.read_image_entry(args.image_entry)
    X = util.read_image_list(img_paths)  # (n_imgs, ysize, xsize, n_channels)

    print('- Loading AutoEncoder2D')
    autoencoder = load_model(args.autoencoder)

    print('- Reconstructing Image')
    Xout = reconstruct_image_set(X, autoencoder)

    print('- Saving Reconstruct Images')
    for i, img_path in enumerate(img_paths):
        out_img_path = os.path.join(args.output_dir, os.path.basename(img_path))
        io.imsave(out_img_path, Xout[i])

    print('\n- Done...')



if __name__ == '__main__':
    main()
