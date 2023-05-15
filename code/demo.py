
import random
import torch.nn.parallel
import torch.utils.data

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
    seed = 42
    train_size = 100
    test_size = 10
    target = 859  # toaster
    patch_params = {'shape': 'circle', 'size': 0.05, 'image_size': 299}
    attack_params = {'conf_target': 0.9, 'max_iter': 500, 'epochs': 5}
    save_path = '/Users/elisharosensweig/workspace/Playground/adverse-test'

    seed_everything(seed=seed)
    print("=> creating model ")
    netClassifier = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')

    print("=> launch patch training")
    patch_attacker = PatchAttacker(classifier=netClassifier, attack_params=attack_params,
                                   train_size=train_size, test_size=test_size,
                                   patch_params=patch_params,
                                   save_path=save_path, log_success=False)
    patch_attacker.train_patch(target=859)
    patch_attacker.patcher.save_patch_to_disk()
    patch_attacker.evaluate_patch(target=target)
