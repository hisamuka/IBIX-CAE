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

from reconstruction import mapping
from reconstruction.reconstruction import reconstruct_image
from reconstruction import util


class MappingDirection(Enum):
    INPUT_2_RECONSTRUCTION = 'input --> reconstruction'
    RECONSTRUCTION_2_INPUT = 'reconstruction --> input'

class LayerName(Enum):
    INPUT_IMAGE = 'input image'
    INPUT_MARKERS = 'input markers'
    OUTPUT_MARKERS = 'output markers'
    RECONSTRUCTION = 'reconstruction'
    FWD_INFLUENCE = 'forward influence'
    BWD_INFLUENCE = 'backward influence'
    INPUT_SUPERPIXELS = 'input superpixels'




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


@magicgui(call_button='Forward Mapping',
          n_perturbations={'label': 'num. perturbations'},
          save_aux_images={'label': 'save aux images', 'tooltip': 'Save auxiliary image into folder \'./out\''})
def forward_mapping(viewer: napari.Viewer, n_perturbations=100, save_aux_images=False) -> napari.types.LayerDataTuple:
    global model

    img = viewer.layers[LayerName.INPUT_IMAGE.value].data
    rec_img = viewer.layers[LayerName.RECONSTRUCTION.value].data
    markers = viewer.layers[LayerName.INPUT_MARKERS.value].data

    print('***** Forward Mapping *****')
    influence_map = mapping.forward_mapping(img, rec_img, markers, n_perturbations, model, save_aux_images)

    # util.mix_image_heatmap(img, influence_map, 'magma')

    return (influence_map, {'name': LayerName.FWD_INFLUENCE.value,
                             'colormap': 'magma', 'blending': 'translucent'}, 'image')


@magicgui(call_button='Backward Mapping',
          n_perturbations={'label': 'num. perturbations'},
          window_size={'label': 'window size'})
def backward_mapping_by_window_sliding(viewer: napari.Viewer, window_size=10, stride=5, n_perturbations=100) -> napari.types.LayerDataTuple:
    global model

    img = viewer.layers[LayerName.INPUT_IMAGE.value].data
    rec_img = viewer.layers[LayerName.RECONSTRUCTION.value].data
    markers = viewer.layers[LayerName.OUTPUT_MARKERS.value].data

    print('***** Inverse Mapping *****')
    influence_map = mapping.backward_mapping_by_window_sliding(img, rec_img, markers, window_size, stride, n_perturbations, model)

    return (influence_map, {'name': LayerName.BWD_INFLUENCE.value,
                             'colormap': 'magma'}, 'image')


@magicgui(call_button='Backward Mapping',
          n_superpixels={'label': 'num. superpixels'},
          compactness={'label': 'compactness'},
          n_perturbations={'label': 'num. perturbations'})
def backward_mapping(viewer: napari.Viewer, n_superpixels=100, compactness=0.1, n_perturbations=100,
                                 multi_scale_optimization=True) -> List[napari.types.LayerDataTuple]:
    global model

    img = viewer.layers[LayerName.INPUT_IMAGE.value].data
    rec_img = viewer.layers[LayerName.RECONSTRUCTION.value].data
    markers = viewer.layers[LayerName.OUTPUT_MARKERS.value].data

    print('***** Inverse Mapping by Superpixels *****')
    influence_map, superpixels = mapping.backward_mapping(img, rec_img, markers, n_superpixels, compactness,
                                                          n_perturbations, model, multi_scale_optimization)

    layers = [
        (superpixels, {'name': LayerName.INPUT_SUPERPIXELS.value}, 'labels'),
        (influence_map, {'name': LayerName.BWD_INFLUENCE.value, 'colormap': 'magma'}, 'image')
    ]

    return layers


def set_layer_contour(viewer, label_name, contour):
    viewer.layers[label_name].contour = contour


if __name__ == '__main__':
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    with napari.gui_qt():
        viewer = napari.Viewer()

        input_image = io.imread(args.input_image)


        viewer.add_image(input_image, name=LayerName.INPUT_IMAGE.value)

        blank = np.zeros(input_image.shape, dtype=np.int)
        napari_colors = build_colors_from_colormap(cmap_name='Set1')
        viewer.add_labels(blank.copy(), name=LayerName.INPUT_MARKERS.value, color=napari_colors)

        model = load_model(args.model)
        rec_img = reconstruct_image(input_image, model)
        viewer.add_image(rec_img, name=LayerName.RECONSTRUCTION.value)

        viewer.window.add_dock_widget(image_filepicker, area='left')
        # viewer.window.add_dock_widget(reconstruct, area='left')
        viewer.window.add_dock_widget([QtWidgets.QLabel('Forward Mapping'), forward_mapping.native], area='left')
        viewer.window.add_dock_widget([QtWidgets.QLabel('Backward Mapping by Window Sliding'), backward_mapping_by_window_sliding.native], area='left')
        viewer.window.add_dock_widget([QtWidgets.QLabel('Backward Mapping'), backward_mapping.native], area='left')

        reconstruct(viewer.layers[LayerName.INPUT_IMAGE.value].data)

        viewer.add_labels(blank.copy(), name=LayerName.OUTPUT_MARKERS.value, color=napari_colors)

        # I couldn't set the label contour from the own LayerType
        backward_mapping.called.connect(lambda event: set_layer_contour(viewer, LayerName.INPUT_SUPERPIXELS.value, 1))
