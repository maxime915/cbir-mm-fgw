"hyreco: utils to read the HyReCo dataset"


import pathlib
from typing import NamedTuple

import pandas as pd

from .mrvips import MRImage


class Pair(NamedTuple):
    img_he: pathlib.Path
    ann_he_yx: pathlib.Path
    img_phh3: pathlib.Path
    ann_phh3_yx: pathlib.Path

    @property
    def key(self):
        return self.img_he.stem

    @property
    def img_he_(self):
        return MRImage.new_from_file(self.img_he, access="random")

    @property
    def ann_he_yx_(self):
        return _read_img_ann(self.img_he_, self.ann_he_yx)[1]

    @property
    def img_phh3_(self):
        return MRImage.new_from_file(self.img_phh3, access="random")

    @property
    def ann_phh3_yx_(self):
        return _read_img_ann(self.img_phh3_, self.ann_phh3_yx)[1]


def _read_dir(dir: pathlib.Path):
    items = [p for p in dir.iterdir() if p.is_file()]
    images = {p.stem: p for p in items if p.suffix == ".tif"}
    ann_s = {p.stem: p for p in items if p.suffix == ".csv"}

    image_keys = sorted(images.keys())
    assert image_keys == sorted(ann_s.keys())

    return image_keys, [(images[k], ann_s[k]) for k in image_keys]


def _read_img_ann(img: MRImage, ann_path: pathlib.Path):
    assert img.raw().get("resolution-unit") == "cm"

    x_res = float(img.raw().get("xres"))  # type: ignore
    y_res = float(img.raw().get("yres"))  # type: ignore

    annotations = [
        (int(y_cm * y_res), int(x_cm * x_res))
        for (_, x_cm, y_cm, _) in pd.read_csv(ann_path).itertuples()
    ]

    return img, annotations


def read_additional_filter(base_dir: pathlib.Path, whitelist: list[str] | None):
    assert base_dir.is_dir()
    he_dir = base_dir / "HE"
    phh3_dir = base_dir / "PHH3"
    assert he_dir.is_dir()
    assert phh3_dir.is_dir()

    he_keys, he_pairs = _read_dir(he_dir)
    phh3_keys, phh3_pairs = _read_dir(phh3_dir)

    assert he_keys == phh3_keys
    good_set = set(whitelist) if whitelist is not None else None

    return [
        Pair(*he, *phh3)
        for key, he, phh3 in zip(he_keys, he_pairs, phh3_pairs, strict=True)
        if good_set is None or key in good_set
    ]


def read_additional(base_dir: pathlib.Path):
    "read the additional part of the HyReCo dataset"
    return read_additional_filter(base_dir, None)

# maybe at a later point, we will do the main part of the dataset, which has more modalities
