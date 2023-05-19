"""
Module for attack class.
The class stores and updates the patch along the gradient to minimize the loss function for the given target class.
"""
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as tvutils
from torch.autograd import Variable

import patcher_utils as putils
from patcher import Patcher


class PatchAttacker(object):

    def __init__(self, classifier, patch_params, attack_params, train_size, test_size, 
                 save_path, logger: logging.Logger, log_images=True):
        self.classifier = classifier
        self.params = attack_params
        self.patch_params = patch_params
        self.save_path = save_path
        self.log_images = log_images
        self.patcher = None
        self.train_stats = None
        self.test_stats = None
        self.logger = logger
        os.makedirs(Path(self.save_path) / 'attack_log', exist_ok=True)
        self._init_datasets(train_size, test_size)

    def _init_datasets(self, train_size, test_size):
        self.logger.info("==> initializing datasets")

        idx = np.arange(50000)  # 50K = number of images in validation set of imagenet
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
        self.logger.info("==> initialized datasets !!")

    def train_patch(self, target):
        self.logger.info("=> Start patch training")
        prev_success = -1  # dummy initial value
        self.classifier.eval()
        self.train_stats = dict()
        self.patcher = Patcher(patch_params=self.patch_params, save_path=self.save_path)

        for i in range(self.params['max_epochs']):
            self.logger.info(f"==> Running Training Epoch {i}")
            total, success, iter_sum = self._train_epoch(target, epoch_num=i)
            self.patcher.save_patch_to_disk(tag=f'epoch{i}')  # log patch evolution over epochs
            self.train_stats[i] = {'Train set size': total, 'patch effective': success,
                                   'avg attack iters': round(float(iter_sum)/total, 2)}

            test_results = self.evaluate_patch(target, check_protect=False)  # defense only checked on final patch
            self.logger.info(f'=======================')
            self.logger.info(f'Epoch {i} test results:')
            self.logger.info(test_results)
            self.logger.info(f'=======================')

            # Let's prevent useless training - if we're stable or relatively effective, we should stop.
            attack_success = float(test_results['patch effective']) / test_results['Test set size']
            if attack_success > self.params['success_thresh'] or abs(attack_success - prev_success) < 0.01:
                break  # attack is good enough or not improving, no need to continue
        return self.train_stats

    def _train_epoch(self, target, epoch_num):
        """ Round of training. Iterate over all training set images and modify patch to misclassify as target. """
        success = 0
        total = 0
        iter_sum = 0
        for image_idx, (image, labels) in enumerate(self.train_set):
            self.logger.debug(f'Epoch #{epoch_num} image #{image_idx}')
            image, labels = Variable(image), Variable(labels)
            orig_label = labels.data[0]
            if target == orig_label or self._non_match(image, orig_label):
                self.logger.debug(f"Ignoring image {image_idx} with label {orig_label}")
                continue  # todo: see if we can avoid classifying these more than once. stable id? hash?
            total += 1

            # try to adapt patch so image is classified as target
            patch, mask = self.patcher.prepare_patch(image)
            patched_image, new_patch, iter_count = self._adapt_patch_to_image(victim=image, target=target,
                                                                              patch=patch, mask=mask)
            iter_sum += iter_count  # the number of iterations should gradually decrease over epochs
            self.patcher.update_patch(
                self.patcher.extract_patch_from_masked(masked_patch=new_patch, mask=mask)
            )
            if self._non_match(patched_image, target):  # attack failed
                continue
            success += 1

            if success % 50 == 0:  # every so often, save a successfully patched image so we can see how it looks
                self._log_images(image, patched_image, image_idx, orig_label, target)
        return total, success, iter_sum

    def _non_match(self, image, label):
        """ Return true if image is not classified as label, false if it is """
        prediction = self.classifier(image.data)
        prediction = prediction.data.max(1)[1][0]
        return prediction != label

    def _adapt_patch_to_image(self, victim, target, patch, mask):
        """
        The brain of the attack: adapting the patch to be more successful
        """
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
            count += 1

        if count == self.params['max_iter']:
            self.logger.info(f"Reached max iterations with prob {target_prob} - better luck next epoch!")
        return adv_image, patch, count

    def _attack_image(self, image, patch, mask):
        """ Add patch to image """
        adv_image = torch.mul((1 - mask), image) + torch.mul(mask, patch)
        adv_image = torch.clamp(adv_image, self.min_out, self.max_out)
        return adv_image

    def _log_images(self, orig_image, patched_image, image_idx, orig_label, target):
        if self.log_images:
            original_path = Path(self.save_path) / 'attack_log' / f'{image_idx}_original_{orig_label}_{target}.png'
            patched_path = Path(self.save_path) / 'attack_log' / f'{image_idx}_adverse_{orig_label}_{target}.png'
            tvutils.save_image(orig_image, original_path, normalize=True)
            tvutils.save_image(patched_image, patched_path, normalize=True)

    def evaluate_patch(self, target, check_protect=True):
        """ Evaluate patch on test set """
        self.logger.info("=> Start patch evaluation round...")
        self.classifier.eval()
        self.test_stats = dict()

        total = 0
        success = 0
        attack_disrupted = 0
        orig_resilient = 0
        for image_idx, (image, labels) in enumerate(self.test_set):
            image, labels = Variable(image), Variable(labels)
            orig_label = labels.data[0]
            if orig_label == target or self._non_match(image, orig_label):
                self.logger.debug(f"Ignoring image {image_idx} with label {orig_label}")
                continue
            total += 1

            # Attack image with trained patch
            patch, mask = self.patcher.prepare_patch(image)
            adv_image = self._attack_image(image, patch, mask)
            if self._non_match(adv_image, target):
                continue
            success += 1

            # check - can random noise added to the image protect it from the patch?
            if check_protect:
                noisy_image = self.get_noisy_patched(adv_image)
                if self._non_match(noisy_image, target):
                    attack_disrupted += 1
                if not self._non_match(noisy_image, orig_label):
                    orig_resilient += 1

        self.test_stats = {'Test set size': total, 'patch effective': success,
                           'noise_disrupts_patch': attack_disrupted,
                           'orig_resilient_with_noise': orig_resilient}
        return self.test_stats

    def get_noisy_patched(self, adv_image):
        """ Get image with noise added to it. """
        noise = torch.randn_like(adv_image)
        noise_intensity = 0.15  # magnitude of noise
        noisy_image = adv_image + noise_intensity * noise
        return noisy_image
