import numpy as np
from skimage import io
from skimage.segmentation import slic

from .reconstruction import reconstruct_image_set
from .util import normalization_value


def forward_mapping(input_img, rec_img, markers, n_perturbations, model, save_aux_images=False):
    roi = markers != 0
    print(roi.shape)

    max_range = int(normalization_value(input_img))  # e.g., 255 in 8-bit image

    pertubations = np.arange(-n_perturbations, n_perturbations + 1, 2)[:n_perturbations]
    shape = tuple([n_perturbations] + list(input_img.shape))  # (n_perturbations, ysize, xsize)

    # creates a numpy array with `n_pertubation` repetitions of the input_image
    # X.shape ==> (n_pertubation, input_image.shape)
    Xpert = np.zeros(shape, dtype='int')

    for i, pert in enumerate(pertubations):
        noise_img = np.array(input_img).astype('float')
        noise_img[roi] += pert
        noise_img[noise_img < 0] = 0
        noise_img[noise_img > max_range] = max_range

        Xpert[i] = noise_img.astype('int')

    Xpert_rec = reconstruct_image_set(Xpert, model)


    reps = tuple([n_perturbations] + [1] * input_img.ndim)
    Xinput = np.tile(input_img, reps=reps)
    Xrec = np.tile(rec_img, reps=reps)

    numerator = np.abs(Xpert_rec - Xrec)
    denominator = np.abs(Xpert - Xinput)
    denominator[denominator == 0] = 1  # to avoid zero-division
    influences = numerator / denominator
    influences = influences.astype('int')

    influence_map = np.mean(numerator, axis=0).astype('int')


    if save_aux_images:
        import os
        if not os.path.exists('./out'):
            os.makedirs('./out')

        for i in range(n_perturbations):
            print(pertubations[i], denominator[i].max())
            io.imsave(f'out/{i}.png', Xpert[i].astype('uint8'))  # debugging
            io.imsave(f'out/{i}_rec.png', Xpert_rec[i].astype('uint8'))  # debugging
            io.imsave(f'out/{i}_numerator.png', numerator[i].astype('uint8'))  # debugging
            io.imsave(f'out/{i}_denominator.png', denominator[i].astype('uint8'))  # debugging
            io.imsave(f'out/{i}_influence.png', influences[i].astype('uint8'))  # debugging

    return influence_map


def backward_mapping(input_img, rec_img, markers, window_size, stride, n_perturbations, model):
    markers_bin = markers != 0
    influence_map = np.zeros(input_img.shape)
    ysize, xsize = input_img.shape[:2]

    n_windows = 0

    for y in range(0, ysize, stride):
        for x in range(0, xsize, stride):
            n_windows += 1
            print(f'** window = {n_windows}')
            y0, y1 = y, min(y + window_size, ysize)
            x0, x1 = x, min(x + window_size, xsize)

            window_mask = np.zeros(input_img.shape, dtype=np.int)
            window_mask[y0:y1, x0:x1] = 1

            direct_influence_map = forward_mapping(input_img, rec_img, window_mask, n_perturbations, model, save_aux_images=False)
            mean_value = np.mean(direct_influence_map[markers_bin])
            influence_map[y0:y1+1, x0:x1+1] += mean_value
            print(y0, y1, x0, x1, mean_value)

    influence_map = influence_map.astype(np.int)

    return influence_map


def backward_mapping_by_superpixels(input_img, rec_img, markers, n_superpixels, compactness, n_perturbations, model):
    markers_bin = markers != 0

    superpixels = slic(input_img, n_segments=n_superpixels, compactness=compactness)
    influence_map = np.zeros(input_img.shape)

    n_superpixels = superpixels.max()
    print(f'n_superpixels = {n_superpixels}')

    for label in range(1, n_superpixels + 1):
            print(f'** label = {label}')

            mask_bool = superpixels == label
            mask = mask_bool.astype(np.int)

            direct_influence_map = forward_mapping(input_img, rec_img, mask, n_perturbations, model, save_aux_images=False)
            mean_value = np.mean(direct_influence_map[markers_bin])
            influence_map[mask_bool] += mean_value
            print(label, mean_value)

    influence_map = influence_map.astype(np.int)

    return influence_map, superpixels
