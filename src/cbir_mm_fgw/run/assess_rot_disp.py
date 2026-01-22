from typing import Literal

from pydantic import BaseModel
from runexp.config_file import runexp_main


class Config(BaseModel):
    ds_dir: str
    model: str
    corr_tag: str
    tile_size_um: float
    tile_size_px: Literal[224]
    tiling_mode: Literal["basic", "overlap"]
    fold: Literal["train", "val", "test"]
    no_multi_modal: bool
    displacement: float
    angle: float


def search_rot_disp(c: Config):
    from .assess_rot_disp_core import run

    run(c)


def main():
    runexp_main(search_rot_disp)


if __name__ == "__main__":
    main()
