
from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--job_number', type=int, default=None,
                        help='The job ID in Slurm.')
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')

    parser.add_argument('--distributed', default=None,
                        choices=[None, 'dp', 'ddp', 'no', 'post_bt'])

    parser.add_argument('--ignore_other_metrics', type=int, choices=[0, 1], default=0,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help="If set, run program with partial epochs and no wandb log.")

    parser.add_argument('--disable_log', action='store_true',
                        help='Disable results logging.')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--savecheck', action='store_true',
                        help='Save checkpoint?')
    parser.add_argument('--start_from', type=int, default=None, help="Task to start from")
    parser.add_argument('--stop_after', type=int, default=None, help="Task limit")

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')


def add_aux_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used to load initial (pretrain) checkpoint
    :param parser: the parser instance
    """
    parser.add_argument('--pre_epochs', type=int, required=False,
                        help='pretrain_epochs.')
    parser.add_argument('--datasetS', type=str, required=False,
                        choices=['cifar100', 'tinyimgR', 'imagenet'])
    parser.add_argument('--load_cp', type=str, default=None)
    parser.add_argument('--stop_after_prep', action='store_true')
