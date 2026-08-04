"""Policy/critic networks and utilities for the SAC and TD3 best-response agents.

Vendored (lightly adapted) from the DDiffPG codebase
(github.com/supersglzc/ddiffpg) so that this repository is self-contained:
- TanhDiagGaussianMLPPolicy (SAC actor), TanhMLPPolicy (TD3 actor)
- DoubleQ / DistributionalDoubleQ critics with categorical projection
- soft_update target-network helper

Adaptation: actor `forward` takes an `eval` flag (eval=True -> deterministic
mean action, eval=False -> sampled action) to match the interface used by the
fictitious-play policy pool (`policy(state, eval=...)`), shared with the
diffusion actor.
"""
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import distributions as pyd


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # Clamp to keep atanh finite at the boundaries.
        return self.atanh(y.clamp(-0.99999997, 0.99999997))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


def create_simple_mlp(in_dim, out_dim, hidden_layers, act=nn.ELU):
    layer_nums = [in_dim, *hidden_layers, out_dim]
    model = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        model.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            model.append(act())
    return nn.Sequential(*model)


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None):
        super().__init__()
        if isinstance(in_dim, Sequence):
            in_dim = in_dim[0]
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        self.net = create_simple_mlp(in_dim=in_dim,
                                     out_dim=out_dim,
                                     hidden_layers=hidden_layers)

    def forward(self, x):
        return self.net(x)


class TanhDiagGaussianMLPPolicy(MLPNet):
    """SAC actor: tanh-squashed diagonal Gaussian."""

    def __init__(self, state_dim, act_dim, hidden_layers=None):
        super().__init__(in_dim=state_dim,
                         out_dim=act_dim * 2,
                         hidden_layers=hidden_layers)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.log_std_min = -5
        self.log_std_max = 5

    def forward(self, state: Tensor, eval: bool = False) -> Tensor:
        # eval=True -> deterministic (distribution mean); else sample.
        return self.get_actions(state, sample=not eval)

    def get_actions(self, state: Tensor, sample=True) -> Tensor:
        dist = self.get_action_dist(state)
        if sample:
            actions = dist.rsample()
        else:
            actions = dist.mean
        return actions

    def get_action_dist(self, state: Tensor):
        mu, log_std = self.net(state).chunk(2, dim=-1)
        std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
        return SquashedNormal(mu, std)

    def get_actions_logprob(self, state: Tensor):
        dist = self.get_action_dist(state)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        return actions, dist, log_prob


class TanhMLPPolicy(MLPNet):
    """TD3 actor: deterministic tanh policy (eval flag accepted for interface
    compatibility with the policy pool; exploration noise is added by the agent)."""

    def __init__(self, state_dim, act_dim, hidden_layers=None):
        super().__init__(in_dim=state_dim, out_dim=act_dim, hidden_layers=hidden_layers)

    def forward(self, state, eval: bool = False):
        return super().forward(state).tanh()


class DoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))

    def get_q1_q2(self, state: Tensor, action: Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x), self.net_q2(input_x)

    def get_q1(self, state: Tensor, action: Tensor) -> Tensor:
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x)


class DistributionalDoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim, v_min=-10, v_max=10, num_atoms=51, device="cuda"):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.device = device
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)
        self.v_min = v_min
        self.v_max = v_max
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        Q1, Q2 = self.get_q1_q2(state, action)
        Q1 = torch.sum(Q1 * self.z_atoms.to(self.device), dim=1)
        Q2 = torch.sum(Q2 * self.z_atoms.to(self.device), dim=1)
        return torch.min(Q1, Q2)

    def get_q1_q2(self, state: Tensor, action: Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1), torch.softmax(self.net_q2(input_x), dim=1)

    def get_q1(self, state: Tensor, action: Tensor) -> Tensor:
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1)


def projection(next_dist, reward, done, gamma, v_min=-10, v_max=10, num_atoms=51, support=None, device="cuda:0"):
    """Categorical (C51-style) projection of the target distribution."""
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = reward.shape[0]

    target_z = (reward + (1 - done) * gamma * support).clamp(min=v_min, max=v_max)
    b = (target_z - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    l[torch.logical_and((u > 0), (l == u))] -= 1
    u[torch.logical_and((l < (num_atoms - 1)), (l == u))] += 1

    proj_dist = torch.zeros_like(next_dist)
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=device).unsqueeze(1).expand(batch_size, num_atoms).long()
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    return proj_dist
