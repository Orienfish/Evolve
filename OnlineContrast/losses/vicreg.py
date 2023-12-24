from __future__ import print_function

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss


def vicreg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Computes VICReg's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.
    Returns:
        torch.Tensor: VICReg loss.
    """

    sim_loss = invariance_loss(z1, z2)
    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)

    loss = sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
    # print(sim_loss, var_loss, cov_loss)
    return loss


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


class VICRegLoss(nn.Module):
    def __init__(self,
                 model='resnet50',
                 lifelong_method='cassle',
                 sim_coeff=25.0,
                 std_coeff=25.0,
                 cov_coeff=1.0,
                 output_dim=2048,
                 proj_hidden_dim=2048,
                 distill_lamb=1,
                 distill_proj_hidden_dim=2048):
        super(VICRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.distill_lamb = distill_lamb

        dim_in = model_dim_dict[model]
        self.projector = nn.Sequential(  # projector
            nn.Linear(dim_in, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        self.cassle = (lifelong_method == 'cassle')
        if self.cassle:
            self.cassle_predictor = nn.Sequential(
                nn.Linear(output_dim, distill_proj_hidden_dim),
                nn.BatchNorm1d(distill_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(distill_proj_hidden_dim, output_dim),
            )
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

        vicreg_loss = vicreg_loss_func(
            z_stu,
            z_tch,
            sim_loss_weight=self.sim_coeff,
            var_loss_weight=self.std_coeff,
            cov_loss_weight=self.cov_coeff,
        )

        # compute the cassle distillation loss if using cassle
        if self.cassle:
            assert self.frozen_backbone is not None, 'frozen encoder has not been created yet'
            p_stu = self.cassle_predictor(z_stu)
            p_tch = self.cassle_predictor(z_tch)
            z_stu_frozen = self.projector(self.frozen_backbone(x_stu))
            z_tch_frozen = self.projector(self.frozen_backbone(x_tch))
            loss_distill = (F.mse_loss(p_stu, z_stu_frozen) +
                            F.mse_loss(p_tch, z_tch_frozen)) / 2

            print('cassle loss: {} loss_distill: {}'.format(vicreg_loss.item(), loss_distill.item()))

            return vicreg_loss + self.distill_lamb * loss_distill

        return vicreg_loss