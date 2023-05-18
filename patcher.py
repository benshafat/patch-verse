"""
Module for a class storing the patch state and configuration parameters.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as tvutils
from torch.autograd import Variable

import patcher_utils as putils


class Patcher(object):

    def __init__(self, patch_params, save_path=None):
        self.save_path = save_path
        self.shape = patch_params['shape']
        self.size = patch_params['size']
        self.image_size = patch_params['image_size']

        if self.shape == 'circle':
            self.init_func = putils.init_patch_circle
            self.trans_func = putils.circle_rotate
        elif self.shape == 'square':
            self.init_func = putils.init_patch_square
            self.trans_func = putils.square_rotate

        self.patch, self.dims = self.init_func(image_size=self.image_size, patch_size=self.size)
        os.makedirs(Path(self.save_path) / 'patches', exist_ok=True)

    def prepare_patch(self, image):
        """
        given an image, prepare a patch for it it at a random place & orientation.
        """
        image_shape = image.data.cpu().numpy().shape
        patch_slide, mask = putils.patch_transform(patcher=self, data_shape=image_shape,
                                                   rotate_func=self.trans_func)
        patch_slide, mask = torch.FloatTensor(patch_slide), torch.FloatTensor(mask)
        patch_slide, mask = Variable(patch_slide), Variable(mask)
        return torch.mul((1 - mask), image) + torch.mul(mask, patch_slide), mask

    def extract_patch_from_masked(self, masked_patch, mask):
        masked_patch = torch.mul(mask, masked_patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(self.dims)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = putils.submatrix(patch[i][j])
        return new_patch

    def update_patch(self, patch):
        """
        update patch state
        """
        self.patch = patch

    def save_patch_to_disk(self, tag=None):
        """
        save patch image at path
        """
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(self.save_path) / 'patches' / f'patch-{tag}-{ts}.png'
        tensor_patch = torch.FloatTensor(self.patch)
        tvutils.save_image(tensor_patch, path, normalize=True)

