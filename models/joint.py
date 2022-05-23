
from datetime import datetime
from utils.training import evaluate
from torch.optim import SGD

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import math
from torchvision import transforms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_aux_dataset_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0
        self.net = WideResNet(depth=28, widen_factor=4, num_classes=100).to(self.device)
        
    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            assert len(dataset.train_loader.dataset.data)
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return

            # reinit network
            # self.net = dataset.get_backbone()
            # self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            dataset.train_loader.dataset.data = all_data
            dataset.train_loader.dataset.targets = all_labels
            loader = torch.utils.data.DataLoader(dataset.train_loader.dataset, batch_size=self.setting.batch_size, shuffle=True)

            if self.setting.scheduler is not None:
                self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
                if self.setting.scheduler == 'simple':
                    assert self.setting.scheduler_rate is not None
                    step = None

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.setting.opt_steps, gamma=self.setting.scheduler_rate, verbose=True)

            # from datetime import datetime
            # # now = datetime.now()
            
            print(f'\nmean acc bf training:',np.mean(evaluate(self, dataset)[0]), '\n')
            self.opt.zero_grad()
            for e in range(self.setting.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels, _ = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    # if i % 3 == 0:
                    self.opt.step()
                    self.opt.zero_grad()
                    progress_bar(i, len(loader), e, 'J', loss.item())

                    # with open(f'logs/{now}.txt', 'a') as f:
                    #     f.write(f'{loss.item()}\n')

                self.opt.step()
                self.opt.zero_grad()
                if self.setting.scheduler is not None:
                    scheduler.step()
                if e < 5 or e % 5 == 0:
                    print(f'\nmean acc at e {e}:',np.mean(evaluate(self, dataset)[0]), '\n')
            # print(f"\nTotal training time {datetime.now() - now}\n")
                

        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS: return
            loader_caches = [[] for _ in range(len(self.old_data))]
            sources = torch.randint(5, (128,))
            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.setting.batch_size
            
            if self.setting.scheduler is not None:
                self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
                if self.setting.scheduler == 'simple':
                    assert self.setting.scheduler_rate is not None
                    # if dataset_setting.n_epochs == 50:
                    step = None
                    if self.args.dataset == 'seq-cifar100':
                        step = [35, 45]

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, step, gamma=self.setting.scheduler_rate, verbose=True)

            for e in range(self.setting.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i+1) * bs]
                    labels = all_labels[order][i * bs: (i+1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())
                scheduler.step()

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        return 0,0,0,0,0
