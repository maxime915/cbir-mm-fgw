"learn the z-score correction statistics"

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


def train_zsc(c: Config):
    from .train_zscore_corr_core import main

    main(c)


def main():
    runexp_main(train_zsc)


if __name__ == "__main__":
    main()
