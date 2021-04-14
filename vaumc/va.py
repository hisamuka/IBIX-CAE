import argparse
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
from magicgui import magicgui
import napari
from napari.types import LayerDataTuple, ImageData
import numpy as np
from pathlib import Path
from skimage import io

from keras.models import load_model
from unsup_neuralnet.generic_autoencoder.reconstruct_image import reconstruct_image


class MappingDirection(Enum):
    INPUT_2_RECONSTRUCTION = 'input --> reconstruction'
    RECONSTRUCTION_2_INPUT = 'reconstruction --> input'

class LayerKey(Enum):
    INPUT_IMAGE = 'input image'
    MARKERS = 'markers'
    RECONSTRUCTION = 'reconstruction'




model = None


def build_argparse():
    prog_desc = \
'''
Visual Analytics Tool for Exploratory Analysis on Unsup. Neural Nets.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input-image', type=str, required=True,
                        help='Input image to reconstruct. Default: None')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Trained unsup. neural network (Keras model *.h5). Default: None')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Input image: %s' % args.input_image)
    print('- Trained unsup. neural network: %s' % args.model)
    print('--------------------------------------------\n')


def validate_args(args):
    if not args.model.endswith('.h5'):
        raise Exception(f'Invalid AutoEncoder2D extension: {args.model}\nTry *.h5')


def build_colors_from_colormap(cmap_name='Set3'):
    # list of RGB tuples (scale 0..1)
    colors = list(plt.get_cmap(cmap_name).colors)
    napari_colors = {}

    for i, RGB in enumerate(colors):
        napari_colors[i + 1] = tuple(list(RGB) + [1.0])  # RGBA

    return napari_colors



@magicgui(call_button='Load Input Image', filename={"filter": "Images (*.jpg *.jpeg *.png)"},
          clear_markers={'label': 'Clear the markers?'})
def image_filepicker(filename=Path(), clear_markers=True) -> List[napari.types.LayerDataTuple]:
    print("****************************")
    img = io.imread(filename)
    out_layers = [(img, {'name': LayerKey.INPUT_IMAGE.value})]

    rec_img_layer = reconstruct(img)
    out_layers.append(rec_img_layer)

    if clear_markers:
        blank = np.zeros(img.shape, dtype=np.int)
        out_layers.append((blank, {'name': LayerKey.MARKERS.value}))

    return out_layers


@magicgui(call_button='Reconstruct')
def reconstruct(img: ImageData) -> napari.types.LayerDataTuple:
    global model
    print(model)

    rec_img = reconstruct_image(img, model)

    return (rec_img, {'name': LayerKey.RECONSTRUCTION.value}, 'image')


# If we use only the Enum as the type (without changing), a dropdown menu is also created but
# the options are the enum names instead of their values: INPUT_2_RECONSTRUCTION and INPUT_2_RECONSTRUCTION
# Although it works that way, I preferred to use their option values.
@magicgui(call_button='Mapping',
          direction={'choices': [MappingDirection.INPUT_2_RECONSTRUCTION.value,
                                MappingDirection.RECONSTRUCTION_2_INPUT.value]})
def mapping(viewer: napari.Viewer, direction=MappingDirection.INPUT_2_RECONSTRUCTION.value):
    img = viewer.layers[LayerKey.INPUT_IMAGE.value]
    markers = viewer.layers[LayerKey.MARKERS.value]
    rec_img = viewer.layers[LayerKey.RECONSTRUCTION.value]

    if direction == MappingDirection.INPUT_2_RECONSTRUCTION.value:
        print('Forward Mapping')
    else:
        print('Reverse Mapping')




if __name__ == '__main__':
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    with napari.gui_qt():
        viewer = napari.Viewer()

        input_image = io.imread(args.input_image)
        viewer.add_image(input_image, name='input image')

        model = load_model(args.model)

        viewer.window.add_dock_widget(image_filepicker, area='left')
        viewer.window.add_dock_widget(reconstruct, area='left')
        viewer.window.add_dock_widget(mapping, area='left')

        reconstruct(viewer.layers[LayerKey.INPUT_IMAGE.value].data)

        blank = np.zeros(input_image.shape, dtype=np.int)
        napari_colors = build_colors_from_colormap(cmap_name='Set1')
        viewer.add_labels(blank, name=LayerKey.MARKERS.value, color=napari_colors)

