import logging
import math
import random

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from scipy.ndimage.interpolation import rotate
from torch.utils.data.sampler import SubsetRandomSampler


def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


def init_patch_square(image_size, patch_size):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(1, 3, noise_dim, noise_dim)
    return patch, patch.shape


def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi))
    patch = np.zeros((1, 3, radius*2, radius*2))    
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius  # The center of circle
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


def patch_transform(patcher, data_shape, rotate_func):
    # get dummy image
    patch_slide = np.zeros(data_shape)
    patch = patcher.patch

    # get shape
    m_size = patcher.dims[-1]

    for i in range(patch_slide.shape[0]):
        patch = rotate_func(patch, i)

        # random location
        random_x = np.random.choice(patcher.image_size)
        while random_x + m_size > patch_slide.shape[-1]:
            random_x = np.random.choice(patcher.image_size)
        random_y = np.random.choice(patcher.image_size)
        while random_y + m_size > patch_slide.shape[-1]:
            random_y = np.random.choice(patcher.image_size)

        # apply patch to dummy image
        for j in range(3):
            patch_slide[i][j][random_x:random_x + m_size, random_y:random_y + m_size] = patch[i][j]

    mask = np.copy(patch_slide)
    mask[mask != 0] = 1.0

    return patch_slide, mask


def square_rotate(patch, i):
    rot = np.random.choice(4)
    for j in range(patch[i].shape[0]):
        patch[i][j] = np.rot90(patch[i][j], rot)
    return patch


def circle_rotate(patch, i):
    rot = np.random.choice(360)
    for j in range(patch[i].shape[0]):
        patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)
    return patch


def get_data_loader(classifier, num_workers, idx, normalize):
    return torch.utils.data.DataLoader(
        dset.ImageFolder('./imagenetdata/val', transforms.Compose([
            transforms.Resize(round(max(classifier.input_size) * 1.050)),
            transforms.CenterCrop(max(classifier.input_size)),
            transforms.ToTensor(),
            ToSpaceBGR(classifier.input_space == 'BGR'),
            ToRange255(max(classifier.input_range) == 255),
            normalize,
        ])),
        batch_size=1, shuffle=False, sampler=SubsetRandomSampler(idx),
        num_workers=num_workers, pin_memory=True)


class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def get_patch_logger(logfile, file_level=logging.DEBUG):
    # Configure the root logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # logfile
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(file_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('PatchLogger')

    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    return logger
