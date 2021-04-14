import numpy as np
from skimage import io

from .reconstruction import reconstruct_image_set
from .util import normalization_value


def forward_mapping(input_img, rec_img, markers, n_perturbations, model, save_aux_images=False):
    roi = markers != 0
    print(roi.shape)

    max_range = int(normalization_value(input_img))  # e.g., 255 in 8-bit image

    # step = (2 * max_range) / n_perturbations
    # # the step may generate an extra value, so we only get the first `n_perturbations` numbers (float)
    # pertubations = np.arange(-max_range, max_range + 1, step)[:n_perturbations]

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

    # mean_influence = np.mean(influences, axis=0).astype('int')  # derivative idea
    mean_influence = np.mean(numerator, axis=0).astype('int')


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

    return mean_influence
