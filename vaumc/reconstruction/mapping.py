import numpy as np
from joblib import Parallel, delayed
from skimage import io
from skimage.segmentation import slic

from pytorch_model import reconstruct_image_set


def forward_mapping(input_img, rec_img, markers, n_perturbations, model, save_aux_images=False):
    roi = markers != 0

    # max_range = int(normalization_value(input_img))  # e.g., 255 in 8-bit image

    # Our model ranges from -1 to 1, we need to divide that range into the required steps
    perturbation_step = 2 / n_perturbations

    perturbations = np.arange(-1.0, 1.0, perturbation_step)[:n_perturbations]
    shape = tuple([n_perturbations] + list(input_img.shape))  # (n_perturbations, ysize, xsize)

    # creates a numpy array with `n_pertubation` repetitions of the input_image
    # X.shape ==> (n_pertubation, input_image.shape)
    Xpert = np.zeros(shape, dtype=input_img.dtype)

    for i, pert in enumerate(perturbations):
        pert = perturbations[i]
        noise_img = np.array(input_img).astype('float')
        noise_img[roi] += pert
        noise_img[noise_img < -1.0] = -1.0
        noise_img[noise_img > 1.0] = 1.0

        Xpert[i] = noise_img

    print("Reconstructing...")
    try:
        Xpert_rec = reconstruct_image_set(Xpert, model)
    except Exception as err:
        print(err)
        raise err

    print("Reps")
    reps = tuple([n_perturbations] + [1] * input_img.ndim)
    # Xinput = np.tile(input_img, reps=reps)
    Xrec = np.tile(rec_img, reps=reps)
    Xrec = np.squeeze(Xrec)

    print("Absolute Error")
    numerator = np.abs(Xpert_rec - Xrec)
    # Dirty hack to make this work with 3-channel images
    # numerator_3ch = np.repeat(numerator[:, :, :, np.newaxis], 3, axis=-1)
    # denominator = np.abs(Xpert - Xinput)
    # denominator[denominator == 0] = 1  # to avoid zero-division
    # influences = numerator_3ch / denominator
    # influences = influences.astype('int')

    print("Influence map")
    influence_map = np.mean(numerator, axis=0)

    # if save_aux_images:
    #     import os
    #     if not os.path.exists('./out'):
    #         os.makedirs('./out')
    #
    #     for i in range(n_perturbations):
    #         print(perturbations[i], denominator[i].max())
    #         io.imsave(f'out/{i}.png', Xpert[i].astype('uint8'))  # debugging
    #         io.imsave(f'out/{i}_rec.png', Xpert_rec[i].astype('uint8'))  # debugging
    #         io.imsave(f'out/{i}_numerator.png', numerator[i].astype('uint8'))  # debugging
    #         io.imsave(f'out/{i}_denominator.png', denominator[i].astype('uint8'))  # debugging
    #         io.imsave(f'out/{i}_influence.png', influences[i].astype('uint8'))  # debugging

    print("Returning")
    return influence_map


def backward_mapping_by_window_sliding(input_img, rec_img, markers, window_size, stride, n_perturbations, model):
    markers_bool = markers != 0
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
            mean_value = np.mean(direct_influence_map[markers_bool])
            influence_map[y0:y1+1, x0:x1+1] += mean_value
            print(y0, y1, x0, x1, mean_value)

    influence_map = influence_map.astype(np.int)

    return influence_map


def _process_backward_mapping_for_single_superpixel(influence_maps, superpixels, label, input_img, rec_img,
                                                    n_perturbations, model, markers_bool):
    print(f'processing backward mapping - superpixel {label}')
    mask_bool = superpixels == label
    mask = mask_bool.astype(np.int)

    direct_influence_map = forward_mapping(input_img, rec_img, mask, n_perturbations, model, save_aux_images=False)
    mean_value = np.mean(direct_influence_map[markers_bool])

    # [i] ==> influence map for the label i + 1
    influence_map_label_ref = influence_maps[label - 1]
    influence_map_label_ref[mask_bool] = mean_value


def backward_mapping_by_superpixels(input_img, rec_img, markers, n_superpixels, compactness, n_perturbations, model,
                                    mask_for_superpixels=None):
    print('===> backward_mapping_by_superpixels')

    ### strategy that extracts superpixels inside a mask (if passed)
    if mask_for_superpixels is None:
        superpixels = slic(input_img[:, :, 2], n_segments=n_superpixels, compactness=compactness)
    else:
        superpixels = slic(input_img[:, :, 2], n_segments=n_superpixels, compactness=compactness,
                           mask=mask_for_superpixels)

    ### strategy that crops the original superpixels map acording to a mask (if passed)
    # superpixels = slic(input_img, n_segments=n_superpixels, compactness=compactness)
    # if mask_for_superpixels is not None:
    #     superpixels[~mask_for_superpixels] = 0
    #
    #     from skimage.segmentation import relabel_sequential
    #     superpixels, _, _ = relabel_sequential(superpixels)

    markers_bool = markers != 0
    found_superpixels = superpixels.max()
    print(f"Number of Superpixels: {found_superpixels} (We wanted: {n_superpixels})")
    influence_maps = np.zeros(tuple([found_superpixels] + list(input_img.shape[:-1])))
    print(influence_maps.shape)

    Parallel(n_jobs=-1, require='sharedmem')(delayed(_process_backward_mapping_for_single_superpixel)
                                             (influence_maps, superpixels, label, input_img, rec_img,
                                              n_perturbations, model, markers_bool)
                                             for label in range(1, found_superpixels + 1))

    influence_map = influence_maps.sum(axis=0)
    print(f'FINISHED - backward_mapping_by_superpixels')

    return influence_map, superpixels


def backward_mapping(input_img, rec_img, markers, n_superpixels, compactness, n_perturbations, model, first_ratio,
                     multiscale=False):
    if multiscale:
        print("===> FIRST SCALE")
        n_superpixels_large_scale = int(max(10, n_superpixels * first_ratio))
        print(n_superpixels_large_scale)
        influence_map_large_scale, superpixels_large_scale = backward_mapping_by_superpixels(input_img, rec_img,
                                                                                             markers,
                                                                                             n_superpixels_large_scale,
                                                                                             compactness,
                                                                                             n_perturbations, model)
        # return influence_map_large_scale, superpixels_large_scale
        print("===> SECOND SCALE")

        mask_for_superpixels = influence_map_large_scale != 0
        io.imsave("~/superpixels_mask.png", mask_for_superpixels.astype(np.uint8) * 255)
        found_superpixels = superpixels_large_scale.max()
        scaling_factor = 255 / found_superpixels
        io.imsave("~/superpixels.png", (superpixels_large_scale * scaling_factor).astype(np.uint8))
    else:
        mask_for_superpixels = None

    return backward_mapping_by_superpixels(input_img, rec_img, markers, n_superpixels, compactness,
                                           n_perturbations, model, mask_for_superpixels)
