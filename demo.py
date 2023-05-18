"""
This module runs the Adversarial Patch attack.
See config.json file for example config
"""


import json

from patch_attacker import PatchAttacker
from patcher_utils import *
from pretrained_models_pytorch import pretrainedmodels

if __name__ == '__main__':

    config_path = 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    logpath = 'patchlog.log'
    logger = get_patch_logger(logfile=logpath)

    # target class we want to reach with patch .
    # see https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/ for mapping of name to class
    target = config['attack_params']['target']
    seed = seed_everything(seed=config['seed'])
    logger.info(f"Seed: {seed}")
    logger.info("=> load model ")
    netClassifier = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')

    patch_attacker = PatchAttacker(classifier=netClassifier,
                                   patch_params=config['patch_params'], attack_params=config['attack_params'],
                                   train_size=config['train_size'], test_size=config['test_size'],
                                   save_path=config['save_path'], log_images=True, logger=logger)
    train_stats = patch_attacker.train_patch(target=target)
    patch_attacker.patcher.save_patch_to_disk(tag='final')
    test_stats = patch_attacker.evaluate_patch(target=target)
    logger.info('\n= = = = = = = = = = = = = = = = = = = = = = = = = =\n')
    logger.info('* Train Set Stats:\n')
    for i in range(config['attack_params']['max_epochs']):
        logger.info(f'\tepoch {i} - {train_stats[i]}')
    logger.info(f'* Test Set Stats: {test_stats}')
