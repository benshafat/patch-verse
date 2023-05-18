"""
This module runs the Adversarial Patch attack.
See config.json file for example config
"""


import json
import logging
import random

from patcher_utils import *
from patch_attacker import PatchAttacker
from pretrained_models_pytorch import pretrainedmodels


def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    logger = logging.getLogger('PatchLogger')

    config_path = 'config.json'
    config = json.loads(config_path)

    # target class we want to reach with patch .
    # see https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/ for mapping of name to class
    target = config['attack_params']['target']
    seed_everything(seed=config['seed'])

    print("=> load model ")
    netClassifier = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')

    print("=> launch patch training")
    patch_attacker = PatchAttacker(classifier=netClassifier,
                                   patch_params=config['patch_params'], attack_params=config['attack_params'],
                                   train_size=config['train_size'], test_size=config['test_size'],
                                   save_path=config['save_path'], log_images=True, logger=logger)
    train_stats = patch_attacker.train_patch(target=target)
    patch_attacker.patcher.save_patch_to_disk(tag='final')
    test_stats = patch_attacker.evaluate_patch(target=target)
    print('\n= = = = = = = = = = = = = = = = = = = = = = = = = =\n')
    print('* Train Set Stats:\n')
    for i in range(config['attack_params']['max_epochs']):
        print(f'\tepoch {i} - {train_stats[i]}')
    print(f'* Test Set Stats: {test_stats}')
