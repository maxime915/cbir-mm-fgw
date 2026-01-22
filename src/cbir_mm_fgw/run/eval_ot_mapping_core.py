import pathlib

import numpy as np
import ot
import torch

from cbir_mm_fgw.hyreco import read_additional, Pair
from cbir_mm_fgw.models import ot_transfer as m
from cbir_mm_fgw.mrvips import MRImage
from cbir_mm_fgw.pipeline import get_pipeline
from cbir_mm_fgw.tiling import _resize, make_patches, make_tiles
from cbir_mm_fgw.profiler import TagProfiler

from .eval_ot_mapping import Config


def ot_fused_gw(
    e_l: torch.Tensor,
    c_l: torch.Tensor,
    e_r: torch.Tensor,
    c_r: torch.Tensor,
    alpha_: float,
):
    # coordinate distance for the in-domain cost
    dist_c_l = m.crosswise_l2(c_l, c_l)
    dist_c_r = m.crosswise_l2(c_r, y=c_r)
    dist_c_l /= dist_c_l.max()
    dist_c_r /= dist_c_r.max()

    # embedding cos distance for inter-domain cost
    dist_emb = m.crosswise_cos_dist(e_l, e_r)
    dist_emb /= dist_emb.max()

    # fused OT
    plan: torch.Tensor = ot.gromov.fused_gromov_wasserstein(  # type: ignore
        dist_emb,
        dist_c_l,
        dist_c_r,
        p=None,
        q=None,
        loss_fun="square_loss",
        symmetric=True,
        alpha=alpha_,
        max_iter=10000,
        tol_rel=1e-9,
        tol_abs=1e-9,
    )

    return plan


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


def hits(ground_truth: np.ndarray, center: np.ndarray, patch_width_px: int):
    gt_y, gt_x = ground_truth
    c_y, c_x = center

    if gt_y < c_y - patch_width_px // 2:
        return False
    if gt_y > c_y + patch_width_px // 2:
        return False
    if gt_x < c_x - patch_width_px // 2:
        return False
    if gt_x > c_x + patch_width_px // 2:
        return False

    return True


def main(c: Config):
    p = TagProfiler()
    all_pairs_ = read_additional(pathlib.Path(c.ds_dir))
    tr_offset_ = round(0.5 * len(all_pairs_))
    vl_offset_ = round(0.7 * len(all_pairs_))
    if c.fold == "train":
        pairs_ = all_pairs_[:tr_offset_]
    elif c.fold == "val":
        pairs_ = all_pairs_[tr_offset_:vl_offset_]
    elif c.fold == "test":
        pairs_ = all_pairs_[vl_offset_:]
    else:
        raise ValueError(f"{c.fold=} is invalid")
    del tr_offset_, vl_offset_, c.ds_dir, all_pairs_

    pipeline_, _ = get_pipeline(c.model, "queue", batch_size=48)

    n_tiles_basic = len(
        make_tiles(
            str(pairs_[0].img_he), c.tiling_size_um, c.tiling_size_px, "basic", read_img=False
        )[1]
    )
    n_tiles_overlap = len(
        make_tiles(
            str(pairs_[0].img_he),
            c.tiling_size_um,
            c.tiling_size_px,
            "overlap",
            read_img=False,
        )[1]
    )

    print(f"OT matching with {c.tiling_size_um=} {c.tiling_size_px=}")
    print(f"\t* {c.n_samples_per_image=}")
    print(f"\t* tiling uniformly without overlap would yield {n_tiles_basic} tiles")
    print(f"\t* tiling uniformly with overlap would yield {n_tiles_overlap} tiles")

    # get the width of each patch (to compute matches)
    patch_width_px, *_ = _resize(pairs_[0].img_he_, c.tiling_size_um, c.tiling_size_px)
    accuracies: list[float] = []
    accuracies_goal: list[float] = []

    prg = np.random.default_rng(c.seed)
    for pair in pairs_:
        with p.profile("time"):
            print(f"reading: {pair.key=}")
            if c.no_multi_modal:
                pair = Pair(pair.img_he, pair.ann_he_yx, pair.img_he, pair.ann_he_yx)

            with p.profile("time/he-random"):
                yx_he_px = sample_points_yx_abs(
                    pair.img_he_, c.tiling_size_um, c.tiling_size_px, c.n_samples_per_image, prg
                )
                c_he_rnd_ = np.array(yx_he_px)
                e_he_rnd_ = pipeline_(
                    make_patches(
                        str(pair.img_he),
                        c.tiling_size_um,
                        c.tiling_size_px,
                        yx_he_px,
                        0.0,
                        read_img=False,
                        allow_overflow=True,
                    )
                )

            with p.profile("time/he-landmark"):
                c_he_lm_ = np.array(pair.ann_he_yx_)
                e_he_lm_ = pipeline_(
                    make_patches(
                        str(pair.img_he),
                        c.tiling_size_um,
                        c.tiling_size_px,
                        pair.ann_he_yx_,
                        0.0,
                        read_img=False,
                        allow_overflow=True,
                    )
                )

            with p.profile("time/phh3-random"):
                # add more points to match the total number of points in H&E (rnd + landmarks)
                yx_phh3_px = sample_points_yx_abs(
                    pair.img_phh3_,
                    c.tiling_size_um,
                    c.tiling_size_px,
                    c.n_samples_per_image + len(pair.ann_phh3_yx_),  # rnd + landmarks
                    prg,
                )
                c_phh3_rnd_ = np.array(yx_phh3_px)
                e_phh3_rnd_ = pipeline_(
                    make_patches(
                        str(pair.img_phh3),
                        c.tiling_size_um,
                        c.tiling_size_px,
                        yx_phh3_px,
                        0.0,
                        read_img=False,
                        allow_overflow=True,
                    )
                )

            # still track the true landmark coordinates: they will be necessary for the evaluation
            c_phh3_gt_lm_ = np.array(pair.ann_phh3_yx_)

            with p.profile("time/optimal-transport-plan"):
                print("mapping")
                e_he = np.concatenate((e_he_lm_, e_he_rnd_))
                c_he = np.concatenate((c_he_lm_, c_he_rnd_))

                mapping_idx = (
                    ot_fused_gw(
                        torch.from_numpy(e_he),
                        torch.from_numpy(c_he),
                        torch.from_numpy(e_phh3_rnd_),
                        torch.from_numpy(c_phh3_rnd_),
                        c.fot_coef,
                    )
                    .numpy()
                    .argmax(axis=1)
                )

            with p.profile("time/accuracy-estimation"):
                n_hits = 0
                n_hit_any_other = 0
                for l_idx, gt in enumerate(c_phh3_gt_lm_):
                    # is the center of the landmark in the selected patch ?
                    if hits(gt, c_phh3_rnd_[mapping_idx[l_idx]], patch_width_px):
                        n_hits += 1
                    else:
                        if any(hits(gt, c, patch_width_px) for c in c_phh3_rnd_):
                            n_hit_any_other += 1

                accuracies.append(n_hits / len(c_phh3_gt_lm_))
                accuracies_goal.append((n_hits + n_hit_any_other) / len(c_phh3_gt_lm_))
            print(f"{accuracies[-1] * 100=:.2f}")
            print(f"{accuracies_goal[-1] * 100=:.2f}")

    print(f"{np.mean(accuracies) * 100=:.2f}")
    print(f"{np.median(accuracies) * 100=:.2f}")
    print(f"{np.mean(accuracies_goal) * 100=:.2f}")
    print(f"{np.median(accuracies_goal) * 100=:.2f}")
    for key, time, num in p.averages():
        print(f"{key}: {time:.03f} (n={num})")
