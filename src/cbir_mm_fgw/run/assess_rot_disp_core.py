import pathlib
import pprint
import time

import numpy as np
import pandas as pd

from cbir_mm_fgw import server_api
from cbir_mm_fgw.hyreco import Pair, read_additional
from cbir_mm_fgw.profiler import TagProfiler
from cbir_mm_fgw.tiling import TilingMode, _resize

from .assess_rot_disp import Config


def build_requests(
    dataset: list[Pair],
    tiling_size_microns: float,
    tiling_size_pixel: int,
    model: str,
    corrector_tag: str,
    tiling_mode: TilingMode,
    no_phh3: bool,
    angle: float,
    disp: float,
    seed: int,
):
    prg = np.random.default_rng(seed)
    payloads: list[server_api.Query] = []
    ground_truth: list[list[tuple[int, int]]] = []
    for pair in dataset:
        # remove PHH3 information
        if no_phh3:
            pair = Pair(pair.img_he, pair.ann_he_yx, pair.img_he, pair.ann_he_yx)

        p_he, p_phh3 = [pair.ann_he_yx_], [pair.ann_phh3_yx_]

        # add a fixed displacement at a random angle
        crop_size_px, _, _ = _resize(
            pair.img_he_, tiling_size_microns, tiling_size_pixel
        )
        disp_angles = prg.random((len(p_he[0]),), dtype=np.float64) * np.pi * 2.0
        dy, dx = (
            np.sin(disp_angles) * disp * crop_size_px,
            np.cos(disp_angles) * disp * crop_size_px,
        )
        p_he[0] = [
            (int(y + dy_), int(x + dx_)) for ((y, x), dy_, dx_) in zip(p_he[0], dy, dx)
        ]

        for c_he, c_phh3 in zip(p_he, p_phh3, strict=True):
            query = server_api.Query(
                model=model,
                corrector_tag=corrector_tag,
                tiling_size_microns=tiling_size_microns,
                tiling_size_pixels=tiling_size_pixel,
                tiling_mode=tiling_mode,
                query_image_path=str(pair.img_he),
                query_patches_rotation=angle,
                query_image_key_points_yx=c_he,
                query_image_modality="HE",
                target_image_path=str(pair.img_phh3),
                target_image_modality="PHH3" if not no_phh3 else "HE",
                limit_results_per_query=-1,
            )

            payloads.append(query)
            ground_truth.append(c_phh3)

    return payloads, ground_truth


def run(c: Config):
    if not pathlib.Path(c.ds_dir).is_dir():
        raise FileNotFoundError(f"{c.ds_dir!r} is not a directory")

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

    # everything parsed
    print("starting")
    print(f"{len(pairs_)=} ({c.fold=})")
    print(f"{c.angle=} {c.displacement=}")
    print(
        f"using {c.model} with s_um={c.tile_size_um} corr={c.corr_tag} t_mode={c.tiling_mode}"
    )

    print("===== Warmup =====")
    start = time.time()

    profiler = TagProfiler()
    warmup_queries, _ = build_requests(
        pairs_[:2],
        c.tile_size_um,
        c.tile_size_px,
        c.model,
        c.corr_tag,
        c.tiling_mode,
        c.no_multi_modal,
        c.angle,
        c.displacement,
        0,
    )

    with profiler.profile("warmup-first"):
        pred = server_api.prediction(warmup_queries[0])
    print(f'{profiler["warmup-first"][0]=}')
    pprint.pp(pred["time"])

    for q in warmup_queries:
        with profiler.profile("warmup-all"):
            server_api.prediction(q)

    for key, time_, n in profiler.averages():
        print(f"{key:30s}: {time_:.3e} ({n=:3d})")

    print(f"Warmed up in {time.time() - start}s")

    print("==== Inspect ====")
    # inspeact the builtin pipeline after the warmup (and reset the profiler)

    for key, mean, std, n in TagProfiler(server_api.infos("", "")["time"]).stats():
        print(f"{key:25s}: {mean:.3e} +/- {std:.3e} tot: {mean * n:.3e} ({n=:3d})")
    server_api.clear_profiler()

    print("==== Accuracy ====")

    start = time.time()

    profiler.clear()
    queries, ground_truths = build_requests(
        pairs_,
        c.tile_size_um,
        c.tile_size_px,
        c.model,
        c.corr_tag,
        c.tiling_mode,
        c.no_multi_modal,
        c.angle,
        c.displacement,
        0,
    )

    ranks: list[int] = []
    distance_to_first_fail: list[float] = []
    distance_to_first_success: list[float] = []

    for q, gt_lst in zip(queries, ground_truths):
        with profiler.profile("request"):
            res = server_api.prediction(q)

        # for each point, for each candidate (closest first), (y, x)
        prediction: list[list[tuple[float, float, float, float]]] = res["results"]
        for pred_lst, (g_y, g_x) in zip(prediction, gt_lst):
            assert len(pred_lst) > 0
            for rank, (top, left, bottom, right) in enumerate(pred_lst):
                hit = True
                if g_y < top or g_y > bottom:
                    hit = False
                if g_x < left or g_x > right:
                    hit = False
                if rank == 0:
                    dst2 = (g_y - 0.5 * (top + bottom)) ** 2.0
                    dst2 += (g_x - 0.5 * (top + bottom)) ** 2.0
                    if hit:
                        distance_to_first_success.append(dst2**0.5)
                    else:
                        distance_to_first_fail.append(dst2**0.5)
                if hit:
                    break
            else:
                rank = len(pred_lst)
            ranks.append(rank)  # type: ignore

    print(f"Done {len(queries)} queries in {time.time() - start}s")

    # compute per rank accuracy
    ranks_ = np.asarray(ranks)
    top_1_acc = (ranks_ < 1).mean()
    top_3_acc = (ranks_ < 3).mean()
    top_5_acc = (ranks_ < 5).mean()
    q95_rank = float(np.quantile(ranks_, 0.95))

    df = pd.DataFrame(
        data=[(top_1_acc, top_3_acc, top_5_acc, q95_rank)],
        columns=["top1", "top3", "top5", "rank-q.95"],  # type: ignore
    )
    print(df.to_csv(index=False))

    # compute distance statistics
    distance_to_first_all = distance_to_first_fail + distance_to_first_success
    print(
        f"n-fail: {len(distance_to_first_fail)}; "
        f"n-success: {len(distance_to_first_success)}; "
        f"n-all: {len(distance_to_first_all)}"
    )
    print(
        f"mu[fail]: {np.mean(distance_to_first_fail)} px; s1[fail]: {np.std(distance_to_first_fail)} px"
        f"mu[success]: {np.mean(distance_to_first_success)} px; s1[success]: {np.std(distance_to_first_success)} px"
        f"mu[all]: {np.mean(distance_to_first_all)} px; s1[all]: {np.std(distance_to_first_all)} px"
    )
    qtl = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    qtl_v_f = np.quantile(distance_to_first_fail, qtl)
    qtl_v_s = np.quantile(distance_to_first_success, qtl)
    qtl_v_a = np.quantile(distance_to_first_all, qtl)
    qtl_df = pd.DataFrame(
        {str(q): [f, s, a] for q, f, s, a in zip(qtl, qtl_v_f, qtl_v_s, qtl_v_a)},
        index=["fail", "success", "all"],  # type:ignore
    )
    print(qtl_df.to_csv(index=True))

    print("===== Timing =====")

    for key, mean, std, n in profiler.stats():
        print(f"{key:30s}: {mean:.3e} +/- {std:.3e} ({n=:3d})")

    print("==== Inspect ====")

    for key, mean, std, n in TagProfiler(server_api.infos("", "")["time"]).stats():
        print(f"{key:25s}: {mean:.3e} +/- {std:.3e} tot: {mean * n:.3e} ({n=:3d})")
