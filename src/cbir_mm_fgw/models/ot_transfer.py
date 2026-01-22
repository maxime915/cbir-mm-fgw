"ot_transfer: models used to learn transport plans in optimal transport"

import torch
from torch.nn.functional import normalize


def crosswise_cos_dist(x: torch.Tensor, y: torch.Tensor):
    """Cosine distance ([N, K], [M, K] -> [N, M])"""
    x_norm = normalize(x, p=2, dim=1)  # [N, K]
    y_norm = normalize(y, p=2, dim=1)  # [M, K]

    c_sim = x_norm @ y_norm.T  # [N, M]
    return 1.0 - c_sim


def crosswise_l2(x: torch.Tensor, y: torch.Tensor):
    """Squared L2 distance ([N, K], [M, K] -> [N, M])"""

    x_norm_2 = (x ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    y_norm_2 = (y ** 2).sum(dim=1, keepdim=True).T  # [1, M]

    # sum (a_i - b_i) ** 2 = (sum a_i)**2 + (sum b_i)**2 - 2 (sum a_i b_i)
    dist_2 = x_norm_2 + y_norm_2 - 2.0 * (x @ y.T)
    return torch.clamp(dist_2, min=0.0)


def linear(feat: int):
    return torch.nn.Linear(feat, feat)


def linear_eye(feat: int):
    layer = linear(feat)
    with torch.no_grad():
        layer.weight.data.zero_()
        layer.weight.data += torch.eye(feat)
        layer.bias.data.zero_()
    return layer


def mlp(feat: int, k: int = 2, dropout_p: float = 0.1):
    """builds an MLP structure (on CPU)"""
    return torch.nn.Sequential(
        torch.nn.Linear(feat, k * feat),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_p),
        torch.nn.Linear(k * feat, k * feat),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_p),
        torch.nn.Linear(k * feat, k * feat),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_p),
        torch.nn.Linear(k * feat, feat),
    )


class mlp_eye(torch.nn.Module):
    def __init__(self, feat: int) -> None:
        super().__init__()
        self.l1 = linear_eye(feat)
        self.l2 = linear_eye(feat)
        self.act = torch.nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor):
        res = self.l2(self.act(self.l1(x)))
        return x + res


class mlp2_eye(torch.nn.Module):
    def __init__(self, feat: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            linear_eye(feat),
            torch.nn.LeakyReLU(0.01),
            linear_eye(feat),
            torch.nn.LeakyReLU(0.01),
            linear_eye(feat),
            torch.nn.LeakyReLU(0.01),
            linear_eye(feat),
        )

    def forward(self, x: torch.Tensor):
        return x + self.backbone(x)
