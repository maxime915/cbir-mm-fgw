"""Train Fused Gromov Wasserstein

This is the entry point to train the FGW corrector module.
"""

from typing import Literal

from pydantic import BaseModel
from runexp.config_file import runexp_main


class Config(BaseModel):
    ds_dir: str
    model: str
    tiling_size_um: float
    tiling_size_px: int
    n_samples_per_image: int
    fold: Literal["train", "val", "test"]
    arch: Literal["linear", "mlp", "linear-eye", "mlp-eye", "mlp2-eye"]
    cyclic: bool
    rec_coef: float
    n_epochs: int
    lr_max: float
    lr_min: float
    random_pairing: bool


def train_fotgw(c: Config):
    from .train_fgw_core import main

    main(c)


def main():
    runexp_main(train_fotgw)


if __name__ == "__main__":
    main()
