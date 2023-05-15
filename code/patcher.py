import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

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

    def prepare_patch(self, image):
        """
        given an image, prepare a patch for it it at a random place & orientation.
        """

        # todo: should there be a distinction b/w patch and patch_slide? I converted all the variables here to slide,
        # todo: but is that correct?
        image_shape = image.data.cpu().numpy().shape
        patch_slide, mask = putils.patch_transform(patcher=self.patch, data_shape=image_shape,
                                                   rotate_func=self.trans_func)
        patch_slide, mask = torch.FloatTensor(patch_slide), torch.FloatTensor(mask)
        patch_slide, mask = Variable(patch_slide), Variable(mask)
        return torch.mul((1 - mask), image) + torch.mul(mask, patch_slide), mask

    def save_patch_to_disk(self):
        """
        save patch image at path
        """
        pass

    def get_patch(self):
        return self.patch

    def update_patch(self, patch):
        """
        modify patch as defined by params
        """
        self.patch = patch





class PatchAttacker(object):

    def __init__(self, classifier, attack_params, train_size, test_size):
        self.classifier = classifier
        self.params = attack_params
        self.patcher = None
        self.attack_stats = dict()
        self.test_stats = None
        self.ignore_idx = set()  # list of image ids to ignore for training
        self._init_datasets(train_size, test_size)

    def _init_datasets(self, train_size, test_size):
        print("==> initializing datasets")

        idx = np.arange(50000)  # number of images in validation set of imagenet
        np.random.shuffle(idx)
        training_idx = idx[:train_size]
        test_idx = idx[train_size:test_size]
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
        # 1. get starting target image from classifier
        # 2. get image for class and init patcher object
        self.patcher = Patcher()
        # 3. for the number of epochs, run attacks
        for i in range(self.params['epochs']):
            print(f"Running Training Epoch {i}")
            total, success = self._train_epoch(target)
            self.attack_stats[i] = (total, success)
        return self.patcher, self.attack_stats

    def _train_epoch(self, target):
        # 1. sample images from image library
        # 2. for each image patch the image
        success = 0
        total = 0
        for image_idx, (image, labels) in enumerate(self.train_set):
            image, labels = Variable(image), Variable(labels)
            if image_idx in self.ignore_idx or self._misclassified(image, labels):
                self.ignore_idx.add(image_idx)
                continue
            total += 1
            patch, mask = self.patcher.prepare_patch(image)
            patched_image, new_patch = self._adapt_patch_to_image(victim=image, target=target, patch=patch, mask=mask)
            self.patcher.update_patch(new_patch)

            # 3. apply classifier to patched
            prediction = self._predict(patched_image)
            if prediction == target:
                success += 1
        return total, success

    def _misclassified(self, image, labels):
        return self._predict(image) != labels.data[0]

    def _adapt_patch_to_image(self, victim, target, patch, mask):
        self.classifier.eval()
        smax = F.softmax(self.classifier(victim))
        target_prob = smax.data[0][target]  # current state of attack
        adv_image = torch.mul((1 - mask), victim) + torch.mul(mask, patch)

        count = 0

        while self.params['conf_target'] > target_prob  and count < self.params['max_iter']:
            adv_image = Variable(adv_image.data, requires_grad=True)
            adv_out = F.log_softmax(self.classifier(adv_image))

            Loss = -adv_out[0][target]
            Loss.backward()
            adv_grad = adv_image.grad.clone()
            adv_image.grad.data.zero_()
            patch -= adv_grad
            adv_image = torch.mul((1 - mask), victim) + torch.mul(mask, patch)
            adv_image = torch.clamp(adv_image, self.min_out, self.max_out)

            out = F.softmax(self.classifier(adv_image))
            target_prob = out.data[0][target]
            count += 1

        return adv_image, patch

    def _predict(self, data):
        return self.classifier(data).data.max(1)[1][0]

    def evaluate_patch(self):
        """
        Evaluate patch on test set
        """
        return self.patcher, self.test_stats

