from typing import Literal, Protocol

import numpy as np
import torch

from .utils import model_directory
from ..models import ot_transfer as m


Modality = Literal["HE", "PHH3"]

_prefix = "corrector_"


class Corrector(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        "apply a correction to x"
        ...


def _eye(x: np.ndarray):
    return x


class ZScoreCorrector(Corrector):
    def __init__(self, mu: np.ndarray, s1: np.ndarray):
        super().__init__()
        self.mu = mu
        self.s1 = s1
        self.eps = 1e-7

    def __call__(self, x: np.ndarray):
        return (x - self.mu) / (self.eps + self.s1)


class ZScoreMapCorrector(ZScoreCorrector):
    def __init__(
        self, mu_s: np.ndarray, s1_s: np.ndarray, mu_d: np.ndarray, s1_d: np.ndarray
    ):
        super().__init__(mu_s, s1_s)
        self.mu_d = mu_d
        self.s1_d = s1_d

    def __call__(self, x: np.ndarray):
        normalized = super().__call__(x)
        return self.s1_d * normalized + self.mu_d


def _z_score_corr(tag: str, mod1: Modality, mod2: Modality):
    if not tag.startswith("z-score-corr_") and not tag.startswith("z-score-corr/"):
        return None
    cache = model_directory() / (_prefix + tag)
    if not cache.is_dir():
        raise FileNotFoundError(f"{cache=} not found")

    # find arrays inside
    mu_1 = cache / f"{mod1}-mu.npy"
    s1_1 = cache / f"{mod1}-s1.npy"
    mu_2 = cache / f"{mod2}-mu.npy"
    s1_2 = cache / f"{mod2}-s1.npy"

    missing = [f for f in (mu_1, s1_1, mu_2, s1_2) if not f.exists()]
    if missing:
        raise FileExistsError(f"missing: {missing}")

    c1 = ZScoreMapCorrector(np.load(mu_1), np.load(s1_1), np.load(mu_2), np.load(s1_2))
    c2 = _eye
    return c1, c2


def _z_score_corr_biased(tag: str, mod1: Modality, mod2: Modality):
    if not tag.startswith("z-score-corr-biased_"):
        return None
    cache = model_directory() / (_prefix + tag)
    if not cache.is_dir():
        raise FileNotFoundError(f"{cache=} not found")

    mu_1 = cache / f"{mod1}-mu.npy"
    s1_1 = cache / f"{mod1}-s1.npy"
    mu_2 = cache / f"{mod2}-mu.npy"
    s1_2 = cache / f"{mod2}-s1.npy"

    missing = [f for f in (mu_1, s1_1, mu_2, s1_2) if not f.exists()]
    if missing:
        raise FileExistsError(f"missing: {missing}")

    c1 = ZScoreMapCorrector(np.load(mu_1), np.load(s1_1), np.load(mu_2), np.load(s1_2))
    c2 = _eye
    return c1, c2


def _z_score_auto_corr(tag: str, mod1: Modality, mod2: Modality):
    if tag != "z-score-auto-corr":
        return None

    def _corr(x: np.ndarray):
        return (x - x.mean(axis=0)) / (1e-7 + x.std(axis=0))

    return _corr, _corr


def _corrector_ot_5(tag: str, mod1: Modality, mod2: Modality):
    if not tag.startswith("5-ot-allpix-patches/"):
        return None
    cache = model_directory() / (_prefix + tag)
    if not cache.is_dir():
        raise FileNotFoundError(f"{cache=} not found")

    params = torch.load(cache / "parameters.pth")

    feat = params["feat"]
    assert isinstance(feat, int)
    model = m.mlp(feat)

    path = cache / f"m_{mod1.lower()}_to_{mod2.lower()}.pth"
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))

    def apply_(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    return apply_, _eye


def _corrector_ot_7(tag: str, mod1: Modality, mod2: Modality):
    if not tag.startswith("7-gwot-allpix-patches/"):
        return None
    cache = model_directory() / (_prefix + tag)
    if not cache.is_dir():
        raise FileNotFoundError(f"{cache=} not found")

    params = torch.load(cache / "parameters.pth")

    feat = params["feat"]
    assert isinstance(feat, int)
    arch = params.get("arch", "linear")
    if arch == "linear":
        model = m.linear(feat)
    elif arch == "mlp":
        model = m.mlp(feat)
    elif arch == "linear-eye":
        model = m.linear_eye(feat)
    elif arch == "mlp-eye":
        model = m.mlp_eye(feat)
    else:
        raise ValueError(f"{params=} at {cache / 'parameters.pth'} has an invalid arch")

    path = cache / f"best_m_{mod1.lower()}_to_{mod2.lower()}.pth"
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))

    def apply_(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    return apply_, _eye


def get_corrector(
    tag: str | None, mod1: Modality, mod2: Modality
) -> tuple[Corrector, Corrector]:
    if tag is None or tag == "no-correction":
        return _eye, _eye  # captain

    # make it works if tag starts with "corrector_" or not
    tag = tag.removeprefix(_prefix)

    if (cp := _z_score_corr_biased(tag, mod1, mod2)) is not None:
        return cp

    if (cp := _z_score_corr(tag, mod1, mod2)) is not None:
        return cp

    if (cp := _z_score_auto_corr(tag, mod1, mod2)) is not None:
        return cp

    if (cp := _corrector_ot_5(tag, mod1, mod2)) is not None:
        return cp

    if (cp := _corrector_ot_7(tag, mod1, mod2)) is not None:
        return cp

    raise ValueError(f"{tag=} is not supported")
