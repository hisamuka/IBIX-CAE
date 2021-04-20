import argparse
from enum import Enum
from typing import List

from keras.models import load_model
import matplotlib.pyplot as plt
from magicgui import magicgui
import napari
from napari.types import LayerDataTuple, ImageData
import numpy as np
from pathlib import Path
from skimage import io

from qtpy import QtWidgets

from reconstruction.mapping import forward_mapping
from reconstruction.reconstruction import reconstruct_image


class MappingDirection(Enum):
    INPUT_2_RECONSTRUCTION = 'input --> reconstruction'
    RECONSTRUCTION_2_INPUT = 'reconstruction --> input'

class LayerName(Enum):
    INPUT_IMAGE = 'input image'
    INPUT_MARKERS = 'input markers'
    RECONSTRUCTION = 'reconstruction'
    FWD_INFLUENCE = 'forwarding influence'
    REV_INFLUENCY = 'reverse influence'




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



@magicgui(call_button='Load Input Image', filename={"filter": "Images (*.jpg *.jpeg *.png)"})
def image_filepicker(viewer: napari.Viewer, filename=Path()):
    global model

    viewer.layers.clear()

    img = io.imread(filename)
    rec_img = reconstruct_image(img, model)
    blank = np.zeros(img.shape, dtype=np.int)

    viewer.add_image(img, name=LayerName.INPUT_IMAGE.value)
    viewer.add_image(rec_img, name=LayerName.RECONSTRUCTION.value)
    viewer.add_labels(blank, name=LayerName.INPUT_MARKERS.value, color=napari_colors)


@magicgui(call_button='Reconstruct')
def reconstruct(img: ImageData) -> napari.types.LayerDataTuple:
    global model
    print(model)

    rec_img = reconstruct_image(img, model)

    return (rec_img, {'name': LayerName.RECONSTRUCTION.value}, 'image')


# If we use only the Enum as the type (without changing), a dropdown menu is also created but
# the options are the enum names instead of their values: INPUT_2_RECONSTRUCTION and INPUT_2_RECONSTRUCTION
# Although it works that way, I preferred to use their option values.
@magicgui(call_button='Forwarding Mapping',
          n_perturbations={'label': 'num. perturbations'},
          save_aux_images={'label': 'save aux images', 'tooltip': 'Save auxiliary image into folder \'./out\''})
def mapping(viewer: napari.Viewer, n_perturbations=100, save_aux_images=False) -> napari.types.LayerDataTuple:
    global model

    img = viewer.layers[LayerName.INPUT_IMAGE.value].data
    rec_img = viewer.layers[LayerName.RECONSTRUCTION.value].data
    markers = viewer.layers[LayerName.INPUT_MARKERS.value].data

    print('***** Forward Mapping *****')
    mean_influence = forward_mapping(img, rec_img, markers, n_perturbations, model, save_aux_images)

    return (mean_influence, {'name': LayerName.FWD_INFLUENCE.value,
                             'colormap': 'magma', 'blending': 'additive'}, 'image')



def reorganize_layer_list(viewer: napari.Viewer):
    print('oioi')


if __name__ == '__main__':
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    with napari.gui_qt():
        viewer = napari.Viewer()

        input_image = io.imread(args.input_image)
        viewer.add_image(input_image, name='input image')

        blank = np.zeros(input_image.shape, dtype=np.int)
        napari_colors = build_colors_from_colormap(cmap_name='Set1')
        viewer.add_labels(blank, name=LayerName.INPUT_MARKERS.value, color=napari_colors)

        model = load_model(args.model)

        viewer.window.add_dock_widget(image_filepicker, area='left')
        viewer.window.add_dock_widget(reconstruct, area='left')
        viewer.window.add_dock_widget([QtWidgets.QLabel('Mapping'), mapping.native], area='left')

        reconstruct(viewer.layers[LayerName.INPUT_IMAGE.value].data)

