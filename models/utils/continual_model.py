
import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace

from datasets import get_dataset
from utils.conf import base_path, get_device
from datasets.seq_cifar100 import MyCIFAR100, SequentialCIFAR100
from datasets.seq_tinyimagenet import MyTinyImagenet, SequentialTinyImagenet32R
from torchvision import transforms
from tqdm import tqdm
import os
from datetime import datetime

from utils.distributed import CustomDP

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def get_aux_dataset(self):
        if self.args.datasetS == 'cifar100':
            aux_dset = MyCIFAR100(base_path(
            ) + 'CIFAR100', train=True, download=True, transform=SequentialCIFAR100.TRANSFORM)
            aux_test_dset = MyCIFAR100(base_path(
            ) + 'CIFAR100', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), SequentialCIFAR100.get_normalization_transform()]))

        elif self.args.datasetS == 'tinyimgR':
            aux_dset = MyTinyImagenet(base_path(
            ) + 'TINYIMG', train=True, download=True, transform=SequentialTinyImagenet32R.TRANSFORM)
            aux_test_dset = MyTinyImagenet(base_path(
            ) + 'TINYIMG', train=False, download=True, transform=SequentialTinyImagenet32R.TEST_TRANSFORM)

        elif self.args.datasetS == 'imagenet':
            ilsvrc_mean = (0.485, 0.456, 0.406)
            ilsvrc_std = (0.229, 0.224, 0.225)
            ilsvrc_transform = transforms.Compose(
                    [transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(ilsvrc_mean, ilsvrc_std)])

            aux_dset = Namespace(
                **{"N_CLASSES": 1000, "transform": ilsvrc_transform})
            aux_test_dset = None
        else:
            raise NotImplementedError(
                f"Dataset `{self.args.datasetS}` not implemented")

        self.num_aux_classes = aux_dset.N_CLASSES
        self.aux_transform = transforms.Compose(
            [transforms.ToPILImage(), aux_dset.transform])

        if self.args.datasetS != 'imagenet':
            self.aux_dl = torch.utils.data.DataLoader(
                aux_dset, batch_size=self.setting.batch_size, shuffle=True, num_workers=0, drop_last=True)
            self.aux_iter = iter(self.aux_dl)

        else:
            self.aux_dl = None
            self.aux_iter=iter([[torch.randn((1,3,224,224)).to(self.device)]])

        return aux_dset, aux_test_dset

    def mini_eval(self):
        model = self
        tg = model.training
        test_dl = torch.utils.data.DataLoader(
            self.aux_test_dset, batch_size=self.setting.batch_size, shuffle=False, num_workers=0)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target, _ in test_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                correct += (torch.argmax(output, dim=1) == target).sum().item()
                total += len(data)
        model.train(tg)
        return correct / total

    def reset_classifier(self):
        self.net.classifier = torch.nn.Linear(
                self.net.classifier.in_features, self.net.num_classes).to(self.device)
        self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                           weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

    def load_initial_checkpoint(self, ignore_classifier=False):
        self.aux_dset, self.aux_test_dset = None, None
        if self.args.datasetS is not None:
            if self.args.datasetS == 'imagenet':
                if not os.path.exists(self.args.load_cp):
                    backbone = get_dataset(self.args).get_backbone()
                    from backbone.ResNet50 import ResNet50
                    from backbone.ResNet18 import ResNet as ResNet18
                    if isinstance(backbone, ResNet50):
                        from torchvision.models import resnet50
                        torch.save(resnet50(pretrained=True).state_dict(),self.args.load_cp)
                    elif isinstance(backbone, ResNet18):
                        from torchvision.models import resnet18
                        torch.save(resnet50(pretrained=True).state_dict(),self.args.load_cp)
                    else:
                        raise NotImplementedError("Imagenet auto-load not supported, provide path for existing checkpoint!")

                self.load_cp(self.args.load_cp, ignore_classifier=ignore_classifier)
            else:
                if self.args.load_cp is None or not os.path.exists(self.args.load_cp):
                    self.aux_dset, self.aux_test_dset = self.get_aux_dataset()
                    self.net.classifier = torch.nn.Linear(
                        self.net.classifier.in_features, self.aux_dset.N_CLASSES).to(self.device)

                    self.opt = SGD(self.net.parameters(
                    ), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
                    sched = None

                    if 'tinyimg' in self.args.datasetS.lower():
                        sched = torch.optim.lr_scheduler.MultiStepLR(
                            self.opt, milestones=[20, 30, 40, 45], gamma=0.5)

                    for e in range(self.setting.pre_epochs):
                        for i, (x, y, _) in tqdm(enumerate(self.aux_dl), desc='Pre-training epoch {}'.format(e), leave=False, total=len(self.aux_dl)):
                            if self.args.debug_mode == 1 and i > 3:
                                break
                            y = y.long()
                            self.net.train()
                            self.opt.zero_grad()
                            x = x.to(self.device)
                            y = y.to(self.device)
                            aux_out = self.net(x)
                            aux_loss = self.loss(aux_out, y)
                            aux_loss.backward()
                            self.opt.step()
                        if sched is not None:
                            sched.step()
                        if e % 5 == 4:
                            print(
                                e, f"{self.mini_eval()*100:.2f}%")
                                
                        if self.args.debug_mode == 1:
                            break
                    # save the model
                    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    if self.args.debug_mode:
                        modelpath = "DEBUG_" + self.NAME + '_' + now + '.pth'
                    else:
                        modelpath = self.NAME + '_' + now + '.pth'
                    torch.save(self.net.state_dict(), modelpath)
                    print(modelpath)
                else:                    
                    assert os.path.isfile(self.args.load_cp), f"File not found: {self.args.load_cp}"

                    self.load_cp(self.args.load_cp, ignore_classifier=ignore_classifier)
                    print("Loaded!")

            if self.aux_test_dset is not None:
                pre_acc = self.mini_eval()
                print(f"Pretrain accuracy: {pre_acc:.2f}")

            if self.args.stop_after_prep:
                exit()


    def to(self, device):
        super().to(device)
        self.device = device
        for d in [x for x in self.__dir__() if hasattr(getattr(self, x), 'device')]:
            getattr(self, d).to(device)

    def load_cp(self, cp_path, new_classes=None, ignore_classifier=False) -> None:
        """
        Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

        :param cp_path: path to checkpoint
        :param new_classes: ignore and rebuild classifier with size `new_classes`
        :param moco: if True, allow load checkpoint for Moco pretraining
        """
        s = torch.load(cp_path, map_location=self.device)
        if 'state_dict' in s:  
            s = {k.replace('encoder_q.', ''): i for k,
                 i in s['state_dict'].items() if 'encoder_q' in k}

        if not ignore_classifier:
            if new_classes is not None:
                self.net.classifier = torch.nn.Linear(
                    self.net.classifier.in_features, self.num_aux_classes).to(self.device)
                for k in list(s):
                    if 'classifier' in k:
                        s.pop(k)
            else:
                cl_weights = [s[k] for k in list(s.keys()) if 'classifier' in k]
                if len(cl_weights) > 0:
                    cl_size = cl_weights[-1].shape[0]
                    self.net.classifier = torch.nn.Linear(
                        self.net.classifier.in_features, cl_size).to(self.device)
        else:
            for k in list(s):
                if 'classifier' in k:
                    s.pop(k)
                    
        for k in list(s):
            if 'net' in k:
                s[k[4:]] = s.pop(k)
        for k in list(s):
            if 'wrappee.' in k:
                s[k.replace('wrappee.', '')] = s.pop(k)
        for k in list(s):
            if '_features' in k:
                s.pop(k)

        try:
            if type(self.net) == CustomDP:
                s = {'module.'+a:b for a,b in s.items()}
            self.net.load_state_dict(s)
        except:
            _, unm = self.net.load_state_dict(s, strict=False)

            if new_classes is not None or ignore_classifier:
                assert all(['classifier' in k for k in unm]
                           ), f"Some of the keys not loaded where not classifier keys: {unm}"
            else:
                assert unm is None, f"Missing keys: {unm}"

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.setting = get_dataset(args).get_setting()
        self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                       weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        self.device = get_device() if self.args.distributed != 'ddp' else 'cuda:0'

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x, **kwargs)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass