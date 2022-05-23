
from copy import deepcopy

from utils.augmentations import normalize
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.distributed import make_dp
from utils.lipschitz import LipOptimizer, add_lipschitz_args
from utils.no_bn import bn_track_stats
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.'
                            'Treated with Lipschitz constraints!')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_lipschitz_args(parser)

    parser.add_argument('--wd_reg', type=float, required=True,
                        help='L2 regularization applied to the parameters.')
                        
    return parser

def icarl_fill_buffer(self: ContinualModel, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // (dataset.N_CLASSES_PER_TASK * (t_idx + 1))

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_l = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx]
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class],
                logits=_y_l[:samples_per_class]
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    mean, std = dataset.get_denormalization_transform().mean, dataset.get_denormalization_transform().std
    classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx+1) * dataset.N_CLASSES_PER_TASK
    # # todo add normalize to features for other datasets

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
        a_x.append(not_norm_x.to('cpu'))
        a_y.append(y.to('cpu'))
        try:
            feats = self.net.features(normalize(not_norm_x, mean, std)).float()
            outs = self.net.classifier(feats)
        except:
            replica_hack = False
            if self.args.distributed == 'dp' and len(not_norm_x) < torch.cuda.device_count():
                # yup, that's exactly right! I you have more GPUs than inputs AND you use kwargs,
                # dataparallel breaks down. So we pad with mock data and then ignore the padding.
                # ref https://github.com/pytorch/pytorch/issues/31460
                replica_hack = True
                not_norm_x = not_norm_x.repeat(torch.cuda.device_count(), 1, 1, 1)

            outs, feats = self.net(normalize(not_norm_x, mean, std), returnt='both')

            if replica_hack:
                outs, feats = outs.split(len(not_norm_x) // torch.cuda.device_count())[0], feats.split(len(not_norm_x) // torch.cuda.device_count())[0]

        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)

    
    # 2.2 Compute class means
    for _y in range(classes_start, classes_end):
        idx = (a_y == _y)
        _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
        feats = a_f[idx]
        feats = feats.reshape(len(feats), -1)
        mean_feat = feats.mean(0, keepdim=True)

        running_sum = torch.zeros_like(mean_feat)
        i = 0
        while i < samples_per_class and i < feats.shape[0]:
            cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

            idx_min = cost.argmin().item()

            mem_buffer.add_data(
                examples=_x[idx_min:idx_min + 1].to(self.device),
                labels=_y[idx_min:idx_min + 1].to(self.device),
                logits=_l[idx_min:idx_min + 1].to(self.device)
            )

            running_sum += feats[idx_min:idx_min + 1]
            feats[idx_min] = feats[idx_min] + 1e6
            i += 1

    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size

    self.net.train(mode)


class ICarlLipschitz(LipOptimizer):
    NAME = 'icarl_lipschitz'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ICarlLipschitz, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.icarl_old_net = None
        self.current_task = 0
        self.num_classes = self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS

    def to(self, device):
        self.eye = self.eye.to(device)
        return super().to(device)

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        try:
            feats = self.net.features(x).float().squeeze()
        except:
            feats = self.net(x, returnt='both')[1].float().squeeze()
        
        feats = feats.reshape(feats.shape[0], -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, logits=None, epoch=None):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to(self.classes_so_far.device))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to(self.classes_so_far.device))).unique())

        self.class_means = None
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.icarl_old_net(inputs))
        self.opt.zero_grad()
        loss, output_features = self.get_loss(inputs, labels, self.current_task, logits)
        
        # Lipschitz losses
        if not self.buffer.is_empty():
            lip_inputs = [inputs] + output_features[:-1]

            if self.args.buffer_lip_lambda>0:
                loss += self.args.buffer_lip_lambda * self.buffer_lip_loss(lip_inputs)
            
            if self.args.budget_lip_lambda>0:
                loss += self.args.budget_lip_lambda * self.budget_lip_loss(lip_inputs) 

        loss.backward()

        self.opt.step()

        return loss.item(), 0, 0, 0, 0

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """
        labels = labels.long()
        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK
        
        outputs, output_features = self.net(inputs, returnt='full')
        outputs = outputs[:, :ac]

        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        if self.args.wd_reg:
            try:
                loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)
            except: # distributed 
                loss += self.args.wd_reg * torch.sum(self.net.module.get_params() ** 2)

        return loss, output_features

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()
                                
            self.net.set_return_prerelu(True)

            self.init_net(dataset)

        if self.current_task > 0:
            dataset.train_loader.dataset.targets = np.concatenate(
                [dataset.train_loader.dataset.targets,
                 self.buffer.labels.cpu().numpy()[:self.buffer.num_seen_examples]])
            if type(dataset.train_loader.dataset.data) == torch.Tensor:
                dataset.train_loader.dataset.data = torch.cat(
                    [dataset.train_loader.dataset.data, torch.stack([(
                        self.buffer.examples[i].type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).squeeze(1)])
            else:
                dataset.train_loader.dataset.data = np.concatenate(
                    [dataset.train_loader.dataset.data, torch.stack([((
                        self.buffer.examples[i] * 255).type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).numpy().swapaxes(1, 3)])


    def end_task(self, dataset) -> None:
        self.icarl_old_net = get_dataset(self.args).get_backbone().to(self.device)
        if self.args.distributed == 'dp':
            self.icarl_old_net = make_dp(self.icarl_old_net)
        _, unexpected = self.icarl_old_net.load_state_dict(deepcopy(self.net.state_dict()), strict=False)
        assert len([k for k in unexpected if 'lip_coeffs' not in k]) == 0, f"Unexpected keys in pretrained model: {unexpected}"
        self.icarl_old_net.eval()
        self.icarl_old_net.set_return_prerelu(True)

        self.net.train()
        with torch.no_grad():
            icarl_fill_buffer(self, self.buffer, dataset, self.current_task)
        self.current_task += 1
        self.class_means = None

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _ = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                class_means.append(self.net(x_buf, returnt='features').mean(0).flatten())
        self.class_means = torch.stack(class_means)
