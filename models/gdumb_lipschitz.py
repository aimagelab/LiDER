from tqdm import tqdm

from utils.args import *
from torch.optim import SGD, lr_scheduler
from utils.buffer import Buffer
import torch
import numpy as np
from utils.distributed import make_dp
from utils.lipschitz import LipOptimizer, add_lipschitz_args

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='GDumb learns an empty model only on the buffer.'
                                        'Treated with Lipschitz constraints!')
    add_management_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--maxlr', type=float, default=5e-2,
                        help='Penalty weight.')
    parser.add_argument('--minlr', type=float, default=5e-4,
                        help='Penalty weight.')
    parser.add_argument('--wd', type=float, default=1e-6,
                        help='Penalty weight.')
    parser.add_argument('--num_passes', type=int, default=250,
                        help='Penalty weight.')
    parser.add_argument('--do_cutmix', type=int, default=1)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='Penalty weight.')
    parser.add_argument('--straight_to_end', type=int, default=1, choices=[0, 1])

    add_experiment_args(parser)
    add_aux_dataset_args(parser)
    add_lipschitz_args(parser)

    return parser

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def get_batch_indexes(N, batch_size):
    """
    Given an iterable, returns a list of batches
    """
    idxs = torch.arange(0, N*batch_size)
    idxs = torch.from_numpy(np.random.permutation(idxs.numpy()))
    for i in range(0, N*batch_size, batch_size):
        yield idxs[i:i + batch_size]

class GDumbLipschitz(LipOptimizer):
    NAME = 'gdumb_lipschitz'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(GDumbLipschitz, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.task = 0
        self.net.set_return_prerelu(True)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        labels=labels.long()
        
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)
        return 0,0,0,0,0

    def fit_buffer(self, epochs):
        pbar = tqdm(range(epochs), total=epochs)
        for epoch in pbar:

            optimizer = SGD(self.net.parameters(), lr=self.args.maxlr, momentum=0.9, weight_decay=self.args.wd)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=self.args.minlr) 

            if epoch <= 0: # Warm start of 1 epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.maxlr * 0.1
            elif epoch == 1: # Then set to maxlr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.maxlr
            else: # oh **** here we go again!
                scheduler.step()

            all_inputs, all_labels = self.buffer.get_data(
                len(self.buffer.examples), transform=self.transform)

            N = len(all_inputs) // self.setting.batch_size
            for i, idxs in enumerate(get_batch_indexes(N, self.setting.batch_size)):
                
                if self.args.debug_mode and i > 2:
                    break
                
                buf_inputs, buf_labels = all_inputs[idxs], all_labels[idxs]
                optimizer.zero_grad()

                if self.args.do_cutmix == 1:
                    inputs, labels_a, labels_b, lam = cutmix_data(x=buf_inputs.cpu(), y=buf_labels.cpu(), alpha=self.args.cutmix_alpha)
                    buf_inputs = inputs.to(self.device)
                    buf_labels_a = labels_a.to(self.device)
                    buf_labels_b = labels_b.to(self.device)
                    buf_outputs = self.net(buf_inputs)
                    loss = lam * self.loss(buf_outputs, buf_labels_a) + (1 - lam) * self.loss(buf_outputs, buf_labels_b)
                else:        
                    buf_outputs = self.net(buf_inputs)
                    loss = self.loss(buf_outputs, buf_labels)

                loss_lip_buffer = torch.tensor(0.)
                loss_lip_budget = torch.tensor(0.)

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

                loss.backward()
                optimizer.step()
                pbar.set_postfix_str = f'Epoch {epoch}/{epochs}: {loss.item():5f}'

    def end_task(self, dataset):
        # new model
        self.task += 1

        if not (self.task == dataset.N_TASKS) and self.args.straight_to_end:
            return
        self.net = dataset.get_backbone().to(self.device)
        if self.args.distributed == 'dp':
            self.net = make_dp(self.net)
            self.net.to(self.device)
        self.net.set_return_prerelu(True)
        self._begin_task(dataset)
        self.fit_buffer(self.args.num_passes)

    def _begin_task(self, dataset):
        self.load_initial_checkpoint()
        self.reset_classifier()

        self.net.set_return_prerelu(True)

        self.init_net(dataset)

