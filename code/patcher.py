import torch
from torch.autograd import Variable
import torch.nn.functional as F


import patcher_utils as putils


class Patcher(object):

    def __init__(self, patch_params, save_path=None):
        self.save_path = save_path
        self.shape = patch_params['shape']
        self.size = patch_params['size']
        self.dims = patch_params['dimensions']
        self.image_size = patch_params['image_size']

        if self.shape == 'circle':
            self.init_func = putils.init_patch_circle
            self.trans_func = putils.circle_transform
        elif self.shape == 'square':
            self.init_func = putils.init_patch_square
            self.trans_func = putils.square_transform

        self.patch = self.init_func(image_size=self.image_size, patch_size=self.size)

    def patch_image(self, image, randomize=True):
        """
        given an image, patch it at a random place & orientation.
        if randomize = True, make random modifications to patch
        """
        self.trans_func(patch=self.patch, patch_shape=self.dims, image_size=self.image_size, )

        return torch.mul((1 - mask), image) + torch.mul(mask, self.patch)

    def save_patch_to_disk(self):
        """
        save patch image at path
        """
        pass

    def get_patch(self):
        return self.patch

    def modify_patch(self, params):
        """
        modify patch as defined by params
        """
        pass


class PatchAttacker(object):

    def __init__(self, classifier, attack_params):
        self.classifier = classifier
        self.params = attack_params
        self.patcher = None
        self.attack_stats = None
        self.test_stats = None
        self.test_set = None
        self.train_set = None
        self._init()

    def _init(self):
        # sample test and train sets
        pass

    def attack_classifier(self, target):
        # 1. get starting target image from classifier
        # 2. get image for class and init patcher object
        self.patcher = Patcher()
        # 3. for the number of epochs, run attacks
        for i in range(self.params['epochs']):
            print(f"Running Training Epoch {i}")
            self._attack_epoch(target)
        return self.patcher, self.attack_stats

    def _attack_epoch(self, target):
        # 1. sample images from image library
        # 2. for each image patch the image
        for image in self.train_set:
            patched_image = self.patcher.patch_image(image=image)
            # 3. apply classifier to patched
            prediction = self._predict(patched_image)
            # 4. compute loss
            # 5. modify patch
            modify_params = {}
            self.patcher.modify_patch(params=modify_params)


    def _attack_image(self, victim, target, mask):
        self.classifier.eval()

        smax = F.softmax(self.classifier(victim))
        target_prob = smax.data[0][target]

        patch = self.patcher.get_patch()
        adv_x = torch.mul((1 - mask), victim) + torch.mul(mask, patch)

        count = 0

        while self.params['conf_target'] > self.params['target_prob'] \
                and count < self.params['max_iter']:
            adv_x = Variable(adv_x.data, requires_grad=True)
            adv_out = F.log_softmax(netClassifier(adv_x))

            adv_out_probs, adv_out_labels = adv_out.max(1)

            Loss = -adv_out[0][target]
            Loss.backward()

            adv_grad = adv_x.grad.clone()

            adv_x.grad.data.zero_()

            patch -= adv_grad

            adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)
            adv_x = torch.clamp(adv_x, min_out, max_out)

            out = F.softmax(netClassifier(adv_x))
            target_prob = out.data[0][target]
            # y_argmax_prob = out.data.max(1)[0][0]

            # print(count, conf_target, target_prob, y_argmax_prob)

            count += 1

        return adv_x, mask, patch

    def _predict(self, data):
        return self.classifier(data).data.max(1)[1][0]

    def evaluate_patch(self):
        """
        Evaluate patch on test set
        """
        return self.patcher, self.test_stats

