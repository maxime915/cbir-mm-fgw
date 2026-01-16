"""Search

This experiment estimates the search accuracy of a configuration.
"""

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


def search(c: Config):
    from .search_core import run

    run(c)


def main():
    runexp_main(search)


if __name__ == "__main__":
    main()
