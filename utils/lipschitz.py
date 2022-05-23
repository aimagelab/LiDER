import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import List
from models.utils.continual_model import ContinualModel

def add_lipschitz_args(parser):
    # BUFFER LIP LOSS
    parser.add_argument('--buffer_lip_lambda', type=float, required=False, default=0,
                        help='Lambda parameter for lipschitz minimization loss on buffer samples')

    # BUDGET LIP LOSS
    parser.add_argument('--budget_lip_lambda', type=float, required=False, default=0,
                        help='Lambda parameter for lipschitz budget distribution loss')

    # Extra
    parser.add_argument('--headless_init_act', type=str, choices=["relu","lrelu"], default="relu") #TODO:""
    parser.add_argument('--grad_iter_step', type=int, required=False, default=-2,
                            help='Step from which to enable gradient computation.') #TODO:""


class LipOptimizer(ContinualModel):

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone,loss,args,transform)
        self.args = args

        self.args.lip_compute_mode = self.args.lip_compute_mode if hasattr(args, "lip_compute_mode") else "different_layer"
        self.args.lip_difference_mode = self.args.lip_difference_mode if hasattr(args, "lip_difference_mode") else "sample"

    def to(self, device):
        self.device = device
        return super().to(device)

    def transmitting_matrix(self, fm1: torch.Tensor, fm2: torch.Tensor):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(-2), fm2.size(-1)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def compute_transition_matrix(self, front: torch.Tensor, latter: torch.Tensor):
        return torch.bmm(self.transmitting_matrix(front, latter), self.transmitting_matrix(front, latter).transpose(2,1))

    def top_eigenvalue(self, K: torch.Tensor, n_power_iterations=10):
        if self.args.grad_iter_step<0:
            start_grad_it = n_power_iterations+self.args.grad_iter_step+1
        else:
            start_grad_it = self.args.grad_iter_step
        assert start_grad_it>=0 and start_grad_it<=n_power_iterations

        v = torch.rand(K.shape[0], K.shape[1], 1).to(K.device, dtype=K.dtype)
        for itt in range(n_power_iterations):
            with torch.set_grad_enabled(itt>=start_grad_it):
                m = torch.bmm(K, v)
                n = (torch.norm(m, dim=1).unsqueeze(1) + torch.finfo(torch.float32).eps)
                v = m / n

        top_eigenvalue = torch.sqrt(n / (torch.norm(v, dim=1).unsqueeze(1) + torch.finfo(torch.float32).eps))
        return top_eigenvalue

    def get_single_feature_lip_coeffs(self, feature: torch.Tensor) -> torch.Tensor:
        B = len(feature[0])
        p=torch.from_numpy(np.random.permutation(B)).to(self.device, dtype=torch.int64)
        features_a, features_b = feature, feature[p] 

        features_a, features_b = features_a.double(), features_b.double()
        features_a, features_b = features_a / self.get_norm(features_a), features_b / self.get_norm(features_b)

        TM_s = self.compute_transition_matrix(features_a, features_b)
        L = self.top_eigenvalue(K=TM_s)
        return L

    def get_layer_lip_coeffs(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        features_a, features_b = features_a.double(), features_b.double()
        features_a, features_b = features_a / self.get_norm(features_a), features_b / self.get_norm(features_b)

        TM_s = self.compute_transition_matrix(features_a, features_b)
        L = self.top_eigenvalue(K=TM_s)
        return L

    def get_feature_lip_coeffs(self, features: List[torch.Tensor], create_att_map=False) -> List[torch.Tensor]:
        if self.args.lip_compute_mode == "different_sample":
            N = len(features)
        else:
            N = len(features)-1

        B = len(features[0])

        lip_values = [torch.zeros(B, device=self.device, dtype=features[0].dtype)]*N

        for i in range(N):
            if self.args.lip_compute_mode == "different_sample":
                L = self.get_single_feature_lip_coeffs(features[i])
            else:
                fma,fmb = features[i], features[i+1]
                fmb = F.adaptive_avg_pool1d(fmb.reshape(*fmb.shape[:2],-1).permute(0,2,1), fma.shape[1]).permute(0,2,1).reshape(fmb.shape[0],-1,*fmb.shape[2:])
                L = self.get_layer_lip_coeffs(fma, fmb)

            L = L.reshape(B)

            lip_values[i] = L if not create_att_map else torch.sigmoid(L)
        return lip_values

    @torch.no_grad()
    def init_net(self, dataset):
        # Eval L for initial model
        self.net.eval()
        
        all_lips = []
        for i, (inputs, labels, _) in enumerate(tqdm(dataset.train_loader, desc="Evaluating initial L")):
            if i>3 and self.args.debug_mode:
                continue
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if len(inputs.shape) == 5:
                B, n, C, H, W = inputs.shape
                inputs = inputs.view(B*n, C, H, W)
            else:
                B, C, H, W = inputs.shape

            _, partial_features = self.net(inputs, returnt='full')

            lip_inputs = [inputs] + partial_features[:-1]

            lip_values = self.get_feature_lip_coeffs(lip_inputs)
            # (B, F)
            lip_values = torch.stack(lip_values, dim=1)

            all_lips.append(lip_values)
            
        self.budget_lip = torch.cat(all_lips, dim=0).mean(0)
        
        inp = next(iter(dataset.train_loader))[0]
        _, teacher_feats = self.net(inp.to(self.device), returnt='full')

        self.net.lip_coeffs = torch.autograd.Variable(torch.randn(len(teacher_feats)-1, dtype=torch.float), requires_grad=True).to(self.device)
        self.net.lip_coeffs.data = self.budget_lip.detach().clone()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr,
                        weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

        self.net.train()


    def buffer_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)

        return lip_values.mean()

    def budget_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)

        if self.args.headless_init_act == "relu":
            tgt = F.relu(self.net.lip_coeffs[:len(lip_values[0])])
        elif self.args.headless_init_act == "lrelu":
            tgt = F.leaky_relu(self.net.lip_coeffs[:len(lip_values[0])])
        else:
            assert False
        tgt = tgt.unsqueeze(0).expand(lip_values.shape)

        loss += F.l1_loss(lip_values, tgt)

        return loss

    def get_norm(self, t: torch.Tensor):
        return torch.norm(t, dim=1, keepdim=True)+torch.finfo(torch.float32).eps
    
    def measure_lip_base(self, s_feats_a, s_feats_b, t_feats_a, t_feats_b):
        with torch.no_grad():
            s_feats_a, s_feats_b = s_feats_a / self.get_norm(s_feats_a), s_feats_b / self.get_norm(s_feats_b)
            t_feats_a, t_feats_b = t_feats_a / self.get_norm(t_feats_a), t_feats_b / self.get_norm(t_feats_b)

            TM_s = self.compute_transition_matrix(s_feats_a, s_feats_b)
            TM_t = self.compute_transition_matrix(t_feats_a, t_feats_b)

            L_s = self.top_eigenvalue(K=TM_s).mean().item()
            L_t = self.top_eigenvalue(K=TM_t).mean().item()

        return L_s, L_t
