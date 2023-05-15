import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from datetime import datetime

import patcher_utils as putils
import torchvision.utils as tvutils


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

    def save_patch_to_disk(self):
        """
        save patch image at path
        """
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(self.save_path) / 'patches' / f'patch-{ts}.png'
        tensor_patch = torch.FloatTensor(self.patch)
        tvutils.save_image(tensor_patch, path, normalize=True)

    def get_patch(self):
        return self.patch

    def update_patch(self, patch):
        """
        modify patch as defined by params
        """
        self.patch = patch


class PatchAttacker(object):

    def __init__(self, classifier, patch_params, attack_params, train_size, test_size, save_path, log_images=True):
        self.classifier = classifier
        self.params = attack_params
        self.patch_params = patch_params
        self.save_path = save_path
        self.log_images = log_images
        self.patcher = None
        self.train_stats = None
        self.test_stats = None
        os.makedirs(Path(self.save_path) / 'attack_log', exist_ok=True)
        self._init_datasets(train_size, test_size)

    def _init_datasets(self, train_size, test_size):
        print("==> initializing datasets")

        idx = np.arange(50000)  # number of images in validation set of imagenet
        np.random.shuffle(idx)
        training_idx = idx[:train_size]
        test_idx = idx[train_size:(train_size + test_size)]
        normalize = transforms.Normalize(mean=self.classifier.mean, std=self.classifier.std)
        self.train_set = putils.get_data_loader(classifier=self.classifier, num_workers=2,
                                                idx=training_idx, normalize=normalize)
        self.test_set = putils.get_data_loader(classifier=self.classifier, num_workers=2,
                                               idx=test_idx, normalize=normalize)

        min_in, max_in = self.classifier.input_range[0], self.classifier.input_range[1]
        min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
        mean, std = np.array(self.classifier.mean), np.array(self.classifier.std)
        self.min_out, self.max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
        print("==> initialized datasets !!")

    def train_patch(self, target):
        self.classifier.eval()
        self.train_stats = dict()
        self.ignore_idx = set()  # set of image ids to ignore for training
        self.patcher = Patcher(patch_params=self.patch_params, save_path=self.save_path)

        for i in range(self.params['epochs']):
            print(f"Running Training Epoch {i}")
            total, success = self._train_epoch(target, epoch_num=i)
            self.patcher.save_patch_to_disk()
            self.train_stats[i] = {'Train set size': total, 'patch effective': success}
        return self.train_stats

    def _train_epoch(self, target, epoch_num):
        success = 0
        total = 0
        for image_idx, (image, labels) in enumerate(self.train_set):
            print(f'Epoch: {epoch_num}, Image ID: {image_idx}, labels = {labels}')
            image, labels = Variable(image), Variable(labels)
            orig_label = labels.data[0]
            if self._misclassified(image, orig_label):
                print(f"Ignoring image {image_idx} with label {orig_label}")
                continue
            total += 1
            patch, mask = self.patcher.prepare_patch(image)
            patched_image, new_patch = self._adapt_patch_to_image(victim=image, target=target, patch=patch, mask=mask)
            self.patcher.update_patch(
                self.patcher.extract_patch_from_masked(masked_patch=new_patch, mask=mask)
            )
            if self._misclassified(patched_image, target):  # attack failed
                continue
            success += 1
            print(f"**** Fooled image {image_idx} !!!!")
            if success % 50 == 0:  # every so often, save a successfully patched image so we can see how it looks
                self._log_images(image, patched_image, image_idx, orig_label, target)
        return total, success

    def _misclassified(self, image, label):
        prediction = self.classifier(image.data)
        prediction = prediction.data.max(1)[1][0]
        return prediction != label

    def _adapt_patch_to_image(self, victim, target, patch, mask):
        self.classifier.eval()
        smax = F.softmax(self.classifier(victim), dim=1)
        target_prob = smax.data[0][target]  # current state of target probability

        adv_image = self._attack_image(victim, patch, mask)
        count = 0

        while self.params['conf_target'] > target_prob and count < self.params['max_iter']:
            adv_image = Variable(adv_image.data, requires_grad=True)
            adv_out = F.log_softmax(self.classifier(adv_image), dim=1)
            loss = -adv_out[0][target]
            loss.backward()
            adv_grad = adv_image.grad.clone()
            adv_image.grad.data.zero_()
            patch -= adv_grad

            adv_image = self._attack_image(victim, patch, mask)
            out = F.softmax(self.classifier(adv_image), dim=1)
            target_prob = out.data[0][target]
            if count % 50 == 0:
                print(f'Reached {count} iterations. Loss = {loss}, Target success prob: {target_prob}')
            count += 1

        print(f'**** Took {count} iterations - {datetime.now()}')
        print(f'**** Probability of target - {target_prob}')

        return adv_image, patch

    def _attack_image(self, image, patch, mask):
        adv_image = torch.mul((1 - mask), image) + torch.mul(mask, patch)
        adv_image = torch.clamp(adv_image, self.min_out, self.max_out)
        return adv_image

    def _log_images(self, orig_image, patched_image, image_idx, orig_label, target):
        if self.log_images:
            original_path = Path(self.save_path) / 'attack_log' / f'{image_idx}_original_{orig_label}_{target}.png'
            patched_path = Path(self.save_path) / 'attack_log' / f'{image_idx}_adverse_{orig_label}_{target}.png'
            tvutils.save_image(orig_image, original_path, normalize=True)
            tvutils.save_image(patched_image, patched_path, normalize=True)

    def evaluate_patch(self, target):
        """
        Evaluate patch on test set
        """
        self.classifier.eval()
        self.test_stats = dict()

        total = 0
        success = 0
        for image_idx, (image, labels) in enumerate(self.test_set):
            image, labels = Variable(image), Variable(labels)
            orig_label = labels.data[0]
            if self._misclassified(image, orig_label):
                print(f"Ignoring image {image_idx} with label {orig_label}")
                continue
            total += 1
            patch, mask = self.patcher.prepare_patch(image)
            adv_image = self._attack_image(image, patch, mask)
            if self._misclassified(adv_image, target):
                continue
            success += 1
        self.test_stats = {'Test set size': total, 'patch effective': success}
        return self.test_stats
