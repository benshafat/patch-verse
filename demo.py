
import random
import torch.nn.parallel
import torch.utils.data

import torch
import torchvision.models as models

# Example usage
model = models.inception_v3(weights=None, init_weights=True)

from pretrained_models_pytorch import pretrainedmodels

from patcher_utils import *
from patcher import PatchAttacker


def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if  __name__ == '__main__':
    seed = None
    train_size = 200
    test_size = 200
    # target = 859 # toaster
    target = 813  # spatula
    patch_params = {'shape': 'circle', 'size': 0.05, 'image_size': 299}
    attack_params = {'conf_target': 0.85, 'max_iter': 500, 'epochs': 10}
    save_path = '/Users/elisharosensweig/workspace/Playground/adverse-test'

    seed_everything(seed=seed)
    print("=> creating model ")

    netClassifier = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')

    print("=> launch patch training")
    patch_attacker = PatchAttacker(classifier=netClassifier, attack_params=attack_params,
                                   train_size=train_size, test_size=test_size,
                                   patch_params=patch_params,
                                   save_path=save_path, log_images=True)
    train_stats = patch_attacker.train_patch(target=target)
    patch_attacker.patcher.save_patch_to_disk()
    test_stats = patch_attacker.evaluate_patch(target=target)
    print('\n= = = = = = = = = = = = = = = = = = = = = = = = = =\n')
    print('* Train Set Stats:\n')
    for i in range(attack_params['epochs']):
        print(f'\tepoch {i}: {train_stats[i]}')
    print(f'* Test Set Stats: {test_stats}')
