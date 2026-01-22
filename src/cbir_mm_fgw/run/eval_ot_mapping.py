from typing import Literal

from pydantic import BaseModel
from runexp.config_file import runexp_main


class Config(BaseModel):
    ds_dir: str
    model: str
    tiling_size_um: float
    tiling_size_px: Literal[224]
    n_samples_per_image: int
    no_multi_modal: bool
    fot_coef: float
    fold: Literal["train", "val", "test"]
    seed: int


def eval_ot(c: Config):
    from .eval_ot_mapping_core import main

    main(c)


def main():
    runexp_main(eval_ot)


if __name__ == "__main__":
    main()
