import csv

import os
import sys
from typing import Dict, Any
from utils.metrics import *
import pandas as pd
from datetime import datetime
from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'disable_log', 'notes', 'load_best_args', 'savecheck']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)

class ExampleLogger:
    def __init__(self, setting_str, dataset_str, model_str, batch_size):
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.batch_size = batch_size
        
        self.task_data = []
        self.index_data = []
        self.matches_data = []
        self.masked_matches_data = []
        self.time = datetime.now()

    def write(self, task='end'):
        create_if_not_exists(base_path() + "examples/" + self.setting +
                             "/" + self.dataset + "/" + self.model)
        path = base_path() + "examples/" + self.setting + "/" + self.dataset\
               + "/" + self.model + "/examples_task_%s_%s.csv" % (str(task), str(self.time))
        
        df = pd.DataFrame()
        df['task'] = self.task_data
        df['index'] = self.index_data
        df['matches'] = self.matches_data
        if len(self.masked_matches_data):
            df['matches_masked'] = self.masked_matches_data
        df.to_csv(path)

        self.task_data = []
        self.index_data = []
        self.matches_data = []
        self.masked_matches_data = []

    def log_batch(self, task, idx, matches, masked_classes=False) -> None:
        if not masked_classes:
            self.task_data += ([task] * len(matches))
            self.index_data += ([idx * self.batch_size + x for x in range(len(matches))])
            self.matches_data += matches
        else:
            self.masked_matches_data += (matches)

class ExampleFullLogger:
    def __init__(self, setting_str, dataset_str, model_str, batch_size):

        print("### WARNING: you are logging all examples: this is very demanding ###")
        
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.batch_size = batch_size
        
        self.epoch_data = []
        self.gt = None
        self.logits_data = []
        self.time = datetime.now()
        self.epoch = 0

    def write(self):
        create_if_not_exists(base_path() + "examples_full/" + self.setting +
                             "/" + self.dataset + "/" + self.model)
        path = base_path() + "examples_full/" + self.setting + "/" + self.dataset\
               + "/" + self.model + "/examples_full_%d_%s.pkl" % (self.batch_size, str(self.time))
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump([self.epoch_data, self.gt, self.logits_data], f)
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def log_batch(self, gt, logits) -> None:
        self.epoch_data.append(self.epoch)
        self.logits_data.append(logits)
        if self.gt is None:
            self.gt = gt

class LossLogger:
    def __init__(self, setting_str, dataset_str, model_str):
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.lossboi = []
        self.time = datetime.now()

    def __del__(self):
        self.write()

    def __def__(self):
        if len(self.lossboi):
            self.write()

    def write(self, task='end'):
        create_if_not_exists(base_path() + "losses/" + self.setting +
                             "/" + self.dataset + "/" + self.model)
        path = base_path() + "losses/" + self.setting + "/" + self.dataset\
               + "/" + self.model + "/loss_task_%s_%s.npy" % (str(task), str(self.time))
        np.save(path, np.array(self.lossboi))
        self.lossboi = []

    def log(self, value) -> None:
        self.lossboi.append(value)

class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in args:
            if cc in useless_args or cc.startswith('conf_'):
                del args[cc]

        columns = list(args.keys())

        new_cols = []
        for i, acc in enumerate(self.accs):
            args['task' + str(i + 1)] = acc
            new_cols.append('task' + str(i + 1))

        args['forward_transfer'] = self.fwt
        new_cols.append('forward_transfer')

        args['backward_transfer'] = self.bwt
        new_cols.append('backward_transfer')

        args['forgetting'] = self.forgetting
        new_cols.append('forgetting')

        columns = new_cols + columns

        create_if_not_exists(base_path() + "results/" + self.setting)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset + "/" + self.model)

        write_headers = False
        path = base_path() + "results/" + self.setting + "/" + self.dataset\
               + "/" + self.model + "/mean_accs.csv"
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)

        if self.setting == 'class-il':
            create_if_not_exists(base_path() + "results/task-il/"
                                 + self.dataset)
            create_if_not_exists(base_path() + "results/task-il/"
                                 + self.dataset + "/" + self.model)

            for i, acc in enumerate(self.accs_mask_classes):
                args['task' + str(i + 1)] = acc

            args['forward_transfer'] = self.fwt_mask_classes
            args['backward_transfer'] = self.bwt_mask_classes
            args['forgetting'] = self.forgetting_mask_classes

            write_headers = False
            path = base_path() + "results/task-il" + "/" + self.dataset + "/"\
                   + self.model + "/mean_accs.csv"
            if not os.path.exists(path):
                write_headers = True
            with open(path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)

class DictxtLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        self.fullaccs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.fullaccs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def dump(self):
        dic = {
            'accs': self.accs,
            'fullaccs': self.fullaccs,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting,
            'fwt_mask_classes': self.fwt_mask_classes,
            'bwt_mask_classes': self.bwt_mask_classes,
            'forgetting_mask_classes': self.forgetting_mask_classes,
        }
        if self.setting == 'class-il':
            dic['accs_mask_classes'] = self.accs_mask_classes
            dic['fullaccs_mask_classes'] = self.fullaccs_mask_classes

        return dic

    def load(self, dic):
        self.accs = dic['accs']
        self.fullaccs = dic['fullaccs']
        self.fwt = dic['fwt']
        self.bwt = dic['bwt']
        self.forgetting = dic['forgetting']
        self.fwt_mask_classes = dic['fwt_mask_classes']
        self.bwt_mask_classes = dic['bwt_mask_classes']
        self.forgetting_mask_classes = dic['forgetting_mask_classes']
        if self.setting == 'class-il':
            self.accs_mask_classes = dic['accs_mask_classes']
            self.fullaccs_mask_classes = dic['fullaccs_mask_classes']

    def rewind(self, num):
        self.accs = self.accs[:-num]
        self.fullaccs = self.fullaccs[:-num]
        try:
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]
            self.fwt_mask_classes = self.fwt_mask_classes[:-num]
            self.bwt_mask_classes = self.bwt_mask_classes[:-num]
            self.forgetting_mask_classes = self.forgetting_mask_classes[:-num]
        except:
            pass
        if self.setting == 'class-il':
            self.accs_mask_classes = self.accs_mask_classes[:-num]
            self.fullaccs_mask_classes = self.fullaccs_mask_classes[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def log_fullacc(self, accs):
        acc_class_il, acc_task_il = accs
        self.fullaccs.append(acc_class_il)
        self.fullaccs_mask_classes.append(acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        wrargs = args.copy()

        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc
        
        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

        wrargs['forward_transfer'] = self.fwt
        wrargs['backward_transfer'] = self.bwt
        wrargs['forgetting'] = self.forgetting

        target_folder = base_path() + "results/"

        create_if_not_exists(target_folder + self.setting)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset + "/" + self.model)

        path = target_folder + self.setting + "/" + self.dataset\
               + "/" + self.model + "/logs.pyd"
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')

        if self.setting == 'class-il':
            create_if_not_exists(os.path.join(*[target_folder, "task-il/", self.dataset]))
            create_if_not_exists(target_folder + "task-il/"
                                 + self.dataset + "/" + self.model)

            for i, acc in enumerate(self.accs_mask_classes):
                wrargs['accmean_task' + str(i + 1)] = acc
            
            for i, fa in enumerate(self.fullaccs_mask_classes):
                for j, acc in enumerate(fa):
                    wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

            wrargs['forward_transfer'] = self.fwt_mask_classes
            wrargs['backward_transfer'] = self.bwt_mask_classes
            wrargs['forgetting'] = self.forgetting_mask_classes

            path = target_folder + "task-il" + "/" + self.dataset + "/"\
                   + self.model + "/logs.pyd"
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')