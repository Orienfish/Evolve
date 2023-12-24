import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Implementation adapted from https://github.com/lucidrains/byol-pytorch

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


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


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

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

class BYOL(nn.Module):
    def __init__(self,
                 model='resnet50',
                 lifelong_method='none',
                 moving_average_decay=0.99,
                 use_momentum=True,
                 distill_lamb=1,
                 distill_proj_hidden_dim=2048):
        super().__init__()
        self.device = (torch.device('cuda')
                       if torch.cuda.is_available()
                       else torch.device('cpu'))

        dim_in = model_dim_dict[model]
        self.projector = projection_MLP(dim_in).to(self.device)
        self.online_encoder = None
        self.target_encoder = None

        self.use_momentum = use_momentum
        self.target_ema_updater = EMA(moving_average_decay)
        self.distill_lamb = distill_lamb

        # self.predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.online_predictor = prediction_MLP().to(self.device)

        self.cassle = (lifelong_method == 'cassle')
        if self.cassle:
            self.cassle_predictor = CaSSLe_Predictor(distill_proj_hidden_dim).to(self.device)
        self.frozen_backbone = None

    def freeze_backbone(self, backbone):
        self.frozen_backbone = copy.deepcopy(backbone)
        set_requires_grad(self.frozen_backbone, False)

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, backbone_stu, backbone_tch, x_stu, x_tch):

        online_proj_stu = self.projector(backbone_stu(x_stu))
        online_proj_tch = self.projector(backbone_tch(x_tch))

        online_pred_stu = self.online_predictor(online_proj_stu)
        online_pred_tch = self.online_predictor(online_proj_tch)

        with torch.no_grad():
            self.online_encoder = nn.Sequential(
                backbone_stu,
                self.projector
            )
            if self.target_encoder is None:  # first forward
                self.target_encoder = copy.deepcopy(self.online_encoder)
                set_requires_grad(self.target_encoder, False)

            # Use online encoder or target encoder depending on self.use_momentum
            target_encoder = self.target_encoder if self.use_momentum else self.online_encoder

            target_proj_stu = target_encoder(x_stu)
            target_proj_tch = target_encoder(x_tch)
            target_proj_stu.detach()
            target_proj_tch.detach()

        loss_one = loss_fn(online_pred_stu, target_proj_tch.detach())
        loss_two = loss_fn(online_pred_tch, target_proj_stu.detach())

        loss = (loss_one + loss_two).mean()

        # compute the cassle distillation loss if using cassle
        if self.cassle:
            assert self.frozen_backbone is not None, 'frozen encoder has not been created yet'
            p_stu = self.cassle_predictor(online_proj_stu)
            p_tch = self.cassle_predictor(online_proj_tch)
            z_stu_frozen = self.projector(self.frozen_backbone(x_stu))
            z_tch_frozen = self.projector(self.frozen_backbone(x_tch))
            loss_distill = (loss_fn(p_stu, z_stu_frozen) +
                            loss_fn(p_tch, z_tch_frozen)).mean() / 2

            print('cassle loss: {} loss_distill: {}'.format(loss.item(), loss_distill.item()))

            return loss + self.distill_lamb * loss_distill

        return loss
