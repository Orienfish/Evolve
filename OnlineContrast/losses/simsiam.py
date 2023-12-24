from __future__ import print_function

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing.
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


model_dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'cnn': 128
}


class CaSSLe_Predictor(nn.Module):
    def __init__(self,
                 distill_proj_hidden_dim,
                 dim_in=2048):
        super().__init__()
        self.distill_predictor = nn.Sequential(
            nn.Linear(dim_in, distill_proj_hidden_dim),
            nn.BatchNorm1d(distill_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(distill_proj_hidden_dim, dim_in),
        )

    def forward(self, x):
        return self.distill_predictor(x)


class SimSiamLoss(nn.Module):
    def __init__(self,
                 model='resnet50',
                 lifelong_method='none',
                 distill_lamb=1,
                 distill_proj_hidden_dim=2048):
        super(SimSiamLoss, self).__init__()

        dim_in = model_dim_dict[model]
        self.projector = projection_MLP(dim_in)
        self.predictor = prediction_MLP()
        self.distill_lamb = distill_lamb

        self.cassle = (lifelong_method == 'cassle')
        if self.cassle:
            self.cassle_predictor = CaSSLe_Predictor(distill_proj_hidden_dim)
        self.frozen_backbone = None

    def freeze_backbone(self, backbone):
        self.frozen_backbone = copy.deepcopy(backbone)
        set_requires_grad(self.frozen_backbone, False)

    def forward(self, backbone_stu, backbone_tch, x_stu, x_tch):
        """Compute loss for model
        Args:
            backbone_stu: backbone for student
            backbone_tch: backbone for teacher
            x_stu: raw augmented vector of shape [bsz, ...].
            x_tch: raw augmented vector of shape [bsz, ...].
        Returns:
            A loss scalar.
        """
        z_stu = self.projector(backbone_stu(x_stu))
        z_tch = self.projector(backbone_tch(x_tch))

        p_stu = self.predictor(z_stu)
        p_tch = self.predictor(z_tch)

        loss = D(p_stu, z_tch) / 2 + D(p_tch, z_stu) / 2

        # compute the cassle distillation loss if using cassle
        if self.cassle:
            assert self.frozen_backbone is not None, 'frozen encoder has not been created yet'
            p_stu = self.cassle_predictor(z_stu)
            p_tch = self.cassle_predictor(z_tch)
            z_stu_frozen = self.projector(self.frozen_backbone(x_stu))
            z_tch_frozen = self.projector(self.frozen_backbone(x_tch))
            loss_distill = (D(p_stu, z_stu_frozen) +
                            D(p_tch, z_tch_frozen)) / 2

            print('cassle loss: {} loss_distill: {}'.format(loss.item(), loss_distill.item()))

            return loss + self.distill_lamb * loss_distill

        return loss


if __name__ == "__main__":
    model = SimSiam()
    model = torch.nn.DataParallel(model).cuda()
    x1 = torch.randn((128, 3, 32, 32))
    x2 = torch.randn_like(x1)

    for i in range(50):
        model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469




