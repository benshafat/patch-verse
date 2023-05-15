import os
import sys
import time
import math
import numpy as np
from scipy.ndimage.interpolation import rotate
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' ' + msg)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


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


# todo: can this logic be true? it seems that they are positioning
# todo: each bit in it's own random location. Why???
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


def init_patch_square(image_size, patch_size):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(1, 3, noise_dim, noise_dim)
    return patch, patch.shape


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

