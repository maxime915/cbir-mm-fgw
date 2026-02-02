import json
import uuid
import pathlib
from datetime import datetime

import numpy as np

from cbir_mm_fgw.hyreco import read_additional
from cbir_mm_fgw.mrvips import MRImage
from cbir_mm_fgw.pipeline import Modality, get_pipeline, model_directory
from cbir_mm_fgw.tiling import _resize, make_patches

from .train_zscore_corr import Config


def sample_points_yx_abs(
    img_: MRImage,
    tiling_size_um: float,
    tiling_size_px: int,
    n: int,
    prg: np.random.Generator,
) -> list[tuple[int, int]]:
    gap_px_, _, _ = _resize(img_, tiling_size_um, tiling_size_px)
    ys = prg.integers(0, img_.height - gap_px_, (n,))
    xs = prg.integers(0, img_.width - gap_px_, (n,))
    # go from top-left to center
    ys += gap_px_ // 2
    xs += gap_px_ // 2

    return list(map(tuple, zip(ys, xs)))


def save_stats(batches_: list[np.ndarray], p: pathlib.Path, mod: Modality):
    whole = np.concatenate(batches_, axis=0)

    mu = np.mean(whole, axis=0)
    s1 = np.std(whole, axis=0, ddof=1)

    np.save(p / f"{mod}-mu.npy", mu)
    np.save(p / f"{mod}-s1.npy", s1)


def main(c: Config):
    if not pathlib.Path(c.ds_dir).is_dir():
        raise FileNotFoundError(f"{c.ds_dir!r} is not a directory")

    all_pairs = read_additional(pathlib.Path(c.ds_dir))
    tr_offset = round(0.5 * len(all_pairs))
    vl_offset = round(0.7 * len(all_pairs))
    if c.fold == "train":
        pairs = all_pairs[:tr_offset]
    elif c.fold == "val":
        pairs = all_pairs[tr_offset:vl_offset]
    elif c.fold == "test":
        pairs = all_pairs[vl_offset:]
    else:
        raise ValueError(f"{c.fold=} is invalid")
    del tr_offset, vl_offset, c.ds_dir, all_pairs

    pipeline, _ = get_pipeline(c.model, "queue")

    key = datetime.now().strftime("%Y%m%d-%H%M") + f"-{uuid.uuid4().hex[:8]}"
    run_tag = f"{c.model}_px{c.tiling_size_px}_um{c.tiling_size_um}_{key}"
    save_dir = model_directory() / "corrector_z-score-corr" / run_tag
    save_dir.parent.mkdir(exist_ok=True, parents=False)  # models-cache must exist
    save_dir.mkdir(exist_ok=True, parents=False)

    (save_dir / "config.json").write_text(json.dumps(c.model_dump(), indent=2))

    embeddings_he: list[np.ndarray] = []
    embeddings_phh3: list[np.ndarray] = []

    prg = np.random.default_rng(0)
    for pair in pairs:
        yx_he_px = sample_points_yx_abs(
            pair.img_he_, c.tiling_size_um, c.tiling_size_px, c.n_samples_per_image, prg
        )
        patches_he = make_patches(
            str(pair.img_he),
            c.tiling_size_um,
            c.tiling_size_px,
            yx_he_px,
            0.0,
            read_img=True,
            allow_overflow=False,
        )
        embeddings_he.append(pipeline(patches_he))

        yx_phh3_px = sample_points_yx_abs(
            pair.img_phh3_, c.tiling_size_um, c.tiling_size_px, c.n_samples_per_image, prg
        )
        patches_phh3 = make_patches(
            str(pair.img_phh3),
            c.tiling_size_um,
            c.tiling_size_px,
            yx_phh3_px,
            0.0,
            read_img=True,
            allow_overflow=False,
        )
        embeddings_phh3.append(pipeline(patches_phh3))

    save_stats(embeddings_he, save_dir, "HE")
    save_stats(embeddings_phh3, save_dir, "PHH3")
