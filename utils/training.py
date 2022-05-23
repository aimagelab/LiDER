import torch
from utils.status import ProgressBar, create_stash, update_status, update_accs
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger, LossLogger, ExampleLogger, ExampleFullLogger, DictxtLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
from tqdm import tqdm
from datetime import datetime
import sys
import pickle
import math
from copy import deepcopy

from utils.distributed import wait_for_master

import torch.optim


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, example_logger=None, verbose=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    iterator = enumerate(dataset.test_loaders)
    if verbose:
        iterator = tqdm(iterator, total=len(dataset.test_loaders))
    for k, test_loader in iterator:
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for idx, data in enumerate(test_loader):
            if model.args.debug_mode and idx > 2:  # len(test_loader) / 2:
                continue
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(
                    model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)  # [:,0:100]

            _, pred = torch.max(outputs.data, 1)
            matches = pred == labels
            correct += torch.sum(matches).item()

            if example_logger and type(example_logger) == ExampleLogger:
                example_logger.log_batch(
                    k, idx, matches.cpu().numpy().tolist(), masked_classes=False)
            if example_logger and type(example_logger) == ExampleFullLogger:
                example_logger.log_batch(labels.cpu().byte().numpy(
                ).tolist(), outputs.cpu().half().numpy().tolist())

            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                masked_matches = pred == labels
                correct_mask_classes += torch.sum(masked_matches).item()
                if example_logger and type(example_logger) == ExampleLogger:
                    example_logger.log_batch(
                        k, idx, masked_matches.cpu().numpy().tolist(), masked_classes=True)

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def compute_average_logit(model: ContinualModel, dataset: ContinualDataset, subsample: float):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    prio = torch.zeros(dataset.N_CLASSES_PER_TASK *
                       dataset.N_TASKS).to(model.device)
    c = 0
    for k, test_loader in enumerate(dataset.test_loaders):
        for idx, data in enumerate(test_loader):
            if idx / len(test_loader) > subsample:
                break
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(
                    model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                prio += outputs.sum(0)
                c += len(outputs)
    model.net.train(status)
    return prio.cpu() / c


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # global sig_pause
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    print(args)

    dataset_setting=dataset.get_setting()
    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = DictxtLogger(dataset.SETTING, dataset.NAME, model.NAME)

    if args.start_from is not None:
        for i in range(args.start_from):
            train_loader, test_loader = dataset.get_data_loaders()
            if hasattr(model, 'end_task'):
                model.end_task(dataset)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    model_stash = create_stash(model, args, dataset)

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    if not args.ignore_other_metrics and not 'MAMMOTH_SLAVE' in os.environ:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        # if 'icarl' not in model.NAME and 'pnn' not in model.NAME:
        #     random_results_class, random_results_task = evaluate(
        #         model, dataset_copy)

    wait_for_master()

    print(file=sys.stderr)

    for t in range(0 if args.start_from is None else args.start_from, dataset.N_TASKS if args.stop_after is None else args.stop_after):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics and not 'MAMMOTH_SLAVE' in os.environ:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        wait_for_master()

        model.evaluator = lambda: evaluate(model, dataset)
        model.evaluation_dsets = dataset.test_loaders
        model.evaluate = lambda dataset: evaluate(model, dataset)

        if dataset_setting.scheduler is not None:
            if dataset_setting.scheduler == 'cosine':
                model.opt = torch.optim.SGD(model.net.parameters(), lr=1e-2)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    model.opt, T_max=dataset_setting.n_epochs, eta_min=1e-4, verbose=True)
            elif dataset_setting.scheduler == 'simple':
                model.opt = torch.optim.SGD(model.net.parameters(
                ), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
                assert dataset_setting.scheduler_rate is not None

                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    model.opt, dataset_setting.opt_steps, gamma=dataset_setting.scheduler_rate, verbose=True)

        else:
            scheduler = None

        for epoch in range(dataset_setting.n_epochs):
            if args.model.startswith('joint'):
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break

                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss, loss_stream, loss_buff, loss_streamu, loss_buffu = model.observe(
                        inputs, labels, not_aug_inputs, logits, epoch=epoch)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss, loss_stream, loss_buff, loss_streamu, loss_buffu = model.observe(
                        inputs, labels, not_aug_inputs, epoch=epoch)
                    assert not math.isnan(loss)

                progress_bar.prog(i, len(train_loader), epoch, t, loss)

                if i % 100 == 0 and not 'MAMMOTH_SLAVE' in os.environ:
                    update_status(i, len(train_loader), epoch, t,
                                  loss, job_number=args.job_number)

                if args.tensorboard and not 'MAMMOTH_SLAVE' in os.environ:
                    tb_logger.log_other('loss', loss, args,
                                        epoch, t, i/len(train_loader))
                    try:
                        tb_logger.log_other(
                            'loss_stream', loss_stream, args, epoch, t, i/len(train_loader))
                        tb_logger.log_other(
                            'loss_buffer', loss_buff, args, epoch, t, i/len(train_loader))
                    except:
                        ...

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0

            wait_for_master()

            if scheduler is not None:
                scheduler.step()

        wait_for_master()

        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        # possible checkpoint saving
        if not 'MAMMOTH_SLAVE' in os.environ and (args.model != 'joint' or t == dataset.N_TASKS - 1):
            accs = evaluate(model, dataset,
                            verbose=not model.args.non_verbose)
            print(accs)

            results.append(accs[0])
            results_mask_classes.append(accs[1])

            mean_acc = np.mean(accs, axis=1)
            update_accs(mean_acc, dataset.SETTING, args.job_number)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        
            model_stash['mean_accs'].append(mean_acc)
            if not args.disable_log:
                logger.log(mean_acc)
                logger.log_fullacc(accs)
            if args.tensorboard:
                tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
            
            if not os.path.isdir('checkpoints'):
                create_if_not_exists("checkpoints")
            now = datetime.now()
            if args.savecheck:
                torch.save(model.state_dict(), 'checkpoints/%s_%s_%d_%d_%s.pt' % (model.NAME, dataset.NAME,
                           model.args.buffer_size if 'buffer_size' in model.args else 0, t, str(now)))
                if 'buffer_size' in model.args:
                    with open('checkpoints/%s_%s_buffer_%d_%d_%s.pkl' % (model.NAME, dataset.NAME, model.args.buffer_size if 'buffer_size' in model.args else 0, t, str(now)), 'wb') as f:
                        pickle.dump(obj=deepcopy(
                            model.buffer).to('cpu'), file=f)
                with open('checkpoints/%s_%s_interpr_%d_%d_%s.pkl' % (model.NAME, dataset.NAME, model.args.buffer_size if 'buffer_size' in model.args else 0, t, str(now)), 'wb') as f:
                    pickle.dump(obj=args, file=f)
                with open('checkpoints/%s_%s_results_%d_%d_%s.pkl' % (model.NAME, dataset.NAME, model.args.buffer_size if 'buffer_size' in model.args else 0, t, str(now)), 'wb') as f:
                    pickle.dump(
                        obj=[results, results_mask_classes, logger.dump()], file=f)

        wait_for_master()

    if not 'MAMMOTH_SLAVE' in os.environ:
        if not args.disable_log and not args.ignore_other_metrics:
            logger.add_bwt(results, results_mask_classes)
            logger.add_forgetting(results, results_mask_classes)

        if args.tensorboard:
            tb_logger.close()
        if not args.disable_log:
            logger.write(vars(args))

    wait_for_master()