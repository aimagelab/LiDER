
from datetime import datetime
import sys
import os
from time import time
import json
from utils.conf import base_path
from typing import Any, Dict, Union
from torch import nn
from argparse import Namespace
from datasets.utils.continual_dataset import ContinualDataset


def update_status(i: int, max_iter: int, epoch: Union[int, str],
                  task_idx: int, loss: float, job_number: None) -> None:
    if job_number is not None:
        if not os.path.exists('./json/job_{}.json'.format(job_number)):
            time.sleep(10)
        with open('./json/job_{}.json'.format(job_number), "r+") as f:
            data = json.load(f)
            data['i'] = i
            data['max_iter'] = max_iter
            data['epoch'] = epoch
            data['task_idx'] = task_idx + 1
            data['loss'] = loss
            f.seek(0)
            json.dump(data, f)
            f.truncate()

def update_accs(mean_accs, setting, job_number):
    if job_number is not None:
        with open('./json/job_{}.json'.format(job_number), "r+") as f:
            data = json.load(f)
            if setting == 'domain-il':
                data['domain_acc'] = round(mean_accs[0], 2)
            else:
                data['class_acc'] = round(mean_accs[0], 2)
                data['task_acc'] = round(mean_accs[1], 2)
            f.seek(0)
            json.dump(data, f)
            f.truncate()


def create_stash(model: nn.Module, args: Namespace,
                 dataset: ContinualDataset) -> Dict[Any, str]:
    """
    Creates the dictionary where to save the model status.
    :param model: the model
    :param args: the current arguments
    :param dataset: the dataset at hand
    """
    now = datetime.now()
    model_stash = {'task_idx': 0, 'epoch_idx': 0, 'batch_idx': 0}
    name_parts = [args.dataset, model.NAME]
    if 'buffer_size' in vars(args).keys():
        name_parts.append('buf_' + str(args.buffer_size))
    name_parts.append(now.strftime("%Y%m%d_%H%M%S_%f"))
    model_stash['model_name'] = '/'.join(name_parts)
    model_stash['mean_accs'] = []
    model_stash['args'] = args
    model_stash['backup_folder'] = os.path.join(base_path(), 'backups',
                                                dataset.SETTING,
                                                model_stash['model_name'])
    return model_stash


def create_fake_stash(model: nn.Module, args: Namespace) -> Dict[Any, str]:
    """
    Create a fake stash, containing just the model name.
    This is used in general continual, as it is useless to backup
    a lightweight MNIST-360 training.
    :param model: the model
    :param args: the arguments of the call
    :return: a dict containing a fake stash
    """
    now = datetime.now()
    model_stash = {'task_idx': 0, 'epoch_idx': 0}
    name_parts = [args.dataset, model.NAME]
    if 'buffer_size' in vars(args).keys():
        name_parts.append('buf_' + str(args.buffer_size))
    name_parts.append(now.strftime("%Y%m%d_%H%M%S_%f"))
    model_stash['model_name'] = '/'.join(name_parts)

    return model_stash


class ProgressBar():
    def __init__(self, verbose=True):
        self.old_time = 0
        self.running_sum = 0
        self.verbose = verbose

    def prog(self, i: int, max_iter: int, epoch: Union[int, str],
                     task_number: int, loss: float) -> None:
        """
        Prints out the progress bar on the stderr file.
        :param i: the current iteration
        :param max_iter: the maximum number of iteration
        :param epoch: the epoch
        :param task_number: the task index
        :param loss: the current value of the loss function
        """
        if not self.verbose:
            if i == 0:
                print('[ {} ] Task {} | epoch {}\n'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    task_number + 1 if isinstance(task_number, int) else task_number,
                    epoch
                ), file=sys.stderr, end='', flush=True)
            else:
                return
        if i == 0:
            self.old_time = time()
            self.running_sum = 0
        else:
            self.running_sum = self.running_sum + (time() - self.old_time)
            self.old_time = time()
        if i:  # not (i + 1) % 10 or (i + 1) == max_iter:
            progress = min(float((i + 1) / max_iter), 1)
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
            print('\r[ {} ] Task {} | epoch {}: |{}| {} ep/h | loss: {} |'.format(
                datetime.now().strftime("%m-%d | %H:%M"),
                task_number + 1 if isinstance(task_number, int) else task_number,
                epoch,
                progress_bar,
                round(3600 / (self.running_sum / i * max_iter), 2),
                round(loss, 8)
            ), file=sys.stderr, end='', flush=True)

def progress_bar(i: int, max_iter: int, epoch: Union[int, str],
                 task_number: int, loss: float) -> None:
    """
    Prints out the progress bar on the stderr file.
    :param i: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param task_number: the task index
    :param loss: the current value of the loss function
    """
    global static_bar

    if i == 0:
        static_bar = ProgressBar()
    static_bar.prog(i, max_iter, epoch, task_number, loss)

    # if not (i + 1) % 10 or (i + 1) == max_iter:
    #     progress = min(float((i + 1) / max_iter), 1)
    #     progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
    #     print('\r[ {} ] Task {} | epoch {}: |{}| loss: {}'.format(
    #         datetime.now().strftime("%m-%d | %H:%M"),
    #         task_number + 1 if isinstance(task_number, int) else task_number,
    #         epoch,
    #         progress_bar,
    #         round(loss, 8)
    #     ), file=sys.stderr, end='', flush=True)
