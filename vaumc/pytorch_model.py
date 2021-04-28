"""
PyTorch model handling
Author: Mathijs de Boer
Date Created: 2021-04-28
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch


def load_model(path: Union[str, Path]) -> torch.nn.Module:
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        return torch.load(path, map_location=torch.device("cuda"))
    else:
        return torch.load(path, map_location=torch.device("cpu"))


def reconstruct_image(image: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    # Pix2Pix assumes 256x256 input, so we do too
    input_size = 256
    step_size = input_size // 4

    has_cuda = torch.cuda.is_available()

    # Assumption that the image is fed in [c, h, w] order
    prediction = np.zeros((image.shape[1], image.shape[2]))

    # We combine several predictions, so we need to average out overlapping areas
    divider = np.zeros_like(prediction)

    for h in range(0, image.shape[1] - input_size + 1, step_size):
        for w in range(0, image.shape[2] - input_size + 1, step_size):
            # Prepare the patch for the model
            patch = image[:, h:h + input_size, w:w + input_size]

            # Run the prediction
            t = torch.from_numpy(patch).to("cuda" if has_cuda else "cpu")
            p = model(t)

            # Get a numpy array from the PyTorch Tensor
            if has_cuda:
                prediction[h:h + input_size, w:w + input_size] += np.squeeze(p.cpu().detach().numpy())
            else:
                prediction[h:h + input_size, w:w + input_size] += np.squeeze(p.detach().numpy())

            # We add 1 to the area we pulled the patch from
            divider[h:h + input_size, w:w + input_size] += 1

    prediction /= divider
    return prediction


def reconstruct_image_set(image_set: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    # Assume a [batch, c, h, w] order
    prediction = np.zeros((image_set.shape[0], image_set.shape[2], image_set.shape[3]))

    for b in image_set.shape[0]:
        prediction[b] = reconstruct_image(image_set[b], model)

    return prediction
