
from copy import deepcopy
import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from datasets import get_dataset
from utils.lipschitz import LipOptimizer, add_lipschitz_args

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER-ACE with future not fixed (as made by authors)'
                                        'Treated with Lipschitz constraints!')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_lipschitz_args(parser)

    return parser


class ErACELipschitz(LipOptimizer):
    NAME = 'er_ace_lipschitz'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACELipschitz, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def begin_task(self, dataset):
        if self.task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

            self.net.set_return_prerelu(True)

            self.init_net(dataset)
        
    def end_task(self, dataset):
        self.task += 1

    def to(self, device):
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        labels = labels.long()
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)

        loss_re = torch.tensor(0.)
        loss_lip_buffer = torch.tensor(0.)
        loss_lip_budget = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.setting.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        if not self.buffer.is_empty():
            if self.args.buffer_lip_lambda>0:
                buf_inputs, _ = self.buffer.get_data(self.setting.minibatch_size, transform=self.transform)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                lip_inputs = [buf_inputs] + buf_output_features[:-1]
                loss_lip_buffer = self.buffer_lip_loss(lip_inputs)
                loss += self.args.buffer_lip_lambda * loss_lip_buffer
            
            if self.args.budget_lip_lambda>0:
                buf_inputs, _ = self.buffer.get_data(self.setting.minibatch_size, transform=self.transform)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                lip_inputs = [buf_inputs] + buf_output_features[:-1]

                loss_lip_budget = self.budget_lip_loss(lip_inputs) 
                loss += self.args.budget_lip_lambda * loss_lip_budget

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item(), 0, 0, 0, 0