
from datasets.seq_cub200 import SequentialCUB200
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.seq_miniimagenet import SequentialMiniImagenet
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    SequentialMiniImagenet.NAME: SequentialMiniImagenet,
    SequentialCUB200.NAME: SequentialCUB200
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
