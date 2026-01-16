"see train_fgw.py"

import json
import pathlib
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
import ot
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.nn import Sequential
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from cbir_mm_fgw.hyreco import read_additional
from cbir_mm_fgw.models import ot_transfer as m
from cbir_mm_fgw.mrvips import MRImage
from cbir_mm_fgw.pipeline import get_pipeline
from cbir_mm_fgw.server_api import _match
from cbir_mm_fgw.tiling import _resize, make_patches, make_tiles

from .train_fgw import Config

# Model definition


def ot_fused_gw(
    e_l: torch.Tensor,
    c_l: torch.Tensor,
    e_r: torch.Tensor,
    c_r: torch.Tensor,
    alpha_: float = 0.5,
):
    # coordinate distance for the in-domain cost
    dist_c_l = m.crosswise_l2(c_l, c_l)
    dist_c_r = m.crosswise_l2(c_r, c_r)
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


def _img_acc(
    p_e: np.ndarray,
    p_c: np.ndarray,
    t_e: np.ndarray,
    t_c: np.ndarray,
    rank_limit: int = 3,
):
    _, indices = _match(tiles=t_e, patches=p_e, limit=50, profiler=None)
    n_hits = 0
    for (y, x), t_idx in zip(p_c, indices, strict=True):
        for rank in range(rank_limit):
            hit = True
            t, l_, b, r = t_c[t_idx[rank]]
            if y < t or y > b:
                hit = False
            if x < l_ or x > r:
                hit = False
            if hit:
                n_hits += 1
                break

    return n_hits / len(p_c)


# Computing the embeddings


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

    # re-do a split:
    #   we should only be doing this on the training fold
    #   the validation fold should be to tune later method (hier and maybe regression)
    #   the test set should be used at the very end
    tr_pairs, vl_pairs = train_test_split(
        pairs, test_size=0.3, random_state=3, shuffle=True
    )

    pipeline, _ = get_pipeline(c.model, "queue")

    embeddings_he: list[np.ndarray] = []
    embeddings_phh3: list[np.ndarray] = []

    coords_he: list[np.ndarray] = []
    coords_phh3: list[np.ndarray] = []

    prg = np.random.default_rng(0)
    for pair in tr_pairs:

        yx_he_px = sample_points_yx_abs(
            pair.img_he_,
            c.tiling_size_um,
            c.tiling_size_px,
            c.n_samples_per_image,
            prg,
        )
        coords_he.append(np.array(yx_he_px))

        yx_he_px = list(map(tuple, coords_he[-1]))
        patches_he = make_patches(
            str(pair.img_he),
            c.tiling_size_um,
            c.tiling_size_px,
            yx_he_px,
            0.0,
            read_img=False,
            allow_overflow=False,
        )
        embeddings_he.append(pipeline(patches_he))

        yx_phh3_px = sample_points_yx_abs(
            pair.img_phh3_,
            c.tiling_size_um,
            c.tiling_size_px,
            c.n_samples_per_image,
            prg,
        )
        coords_phh3.append(np.array(yx_phh3_px))

        yx_phh3_px = list(map(tuple, coords_phh3[-1]))
        patches_phh3 = make_patches(
            str(pair.img_phh3),
            c.tiling_size_um,
            c.tiling_size_px,
            yx_phh3_px,
            0.0,
            read_img=False,
            allow_overflow=False,
        )
        embeddings_phh3.append(pipeline(patches_phh3))

    vl_lm_emb_he: list[np.ndarray] = []
    vl_lm_coord_he: list[np.ndarray] = []
    vl_ti_emb_phh3: list[np.ndarray] = []
    vl_ti_tlbr_phh3: list[np.ndarray] = []
    vl_lm_emb_phh3: list[np.ndarray] = []
    prg = np.random.default_rng(0)
    t_prg = torch.Generator().manual_seed(0)
    for pair in vl_pairs:
        yx_he_px = pair.ann_he_yx_
        vl_lm_coord_he.append(np.array(yx_he_px))

        patches_he = make_patches(
            str(pair.img_he),
            c.tiling_size_um,
            c.tiling_size_px,
            list(map(tuple, vl_lm_coord_he[-1])),
            0.0,
            read_img=False,
            allow_overflow=True,
        )
        vl_lm_emb_he.append(pipeline(patches_he))

        tc_phh3_, ti_phh3 = make_tiles(
            str(pair.img_phh3),
            c.tiling_size_um,
            c.tiling_size_px,
            tiling_mode="overlap",
            read_img=False,
        )
        vl_ti_emb_phh3.append(pipeline(ti_phh3))
        vl_ti_tlbr_phh3.append(tc_phh3_)

        patches_phh3 = make_patches(
            str(pair.img_phh3),
            c.tiling_size_um,
            c.tiling_size_px,
            pair.ann_phh3_yx_,
            0.0,
            read_img=False,
            allow_overflow=True,
        )
        vl_lm_emb_phh3.append(pipeline(patches_phh3))

    key = datetime.now().strftime("%Y%m%d-%H%M") + f"-{uuid.uuid4().hex[:8]}"
    run_tag = f"{c.model}_px{c.tiling_size_px}_um{c.tiling_size_um}_{key}"
    if c.random_pairing:
        run_tag += "_rndpaired"
    save_dir = (
        pathlib.Path("models-cache") / "corrector_7-gwot-allpix-patches" / run_tag
    )
    save_dir.parent.mkdir(exist_ok=True, parents=False)  # models-cache must exist
    save_dir.mkdir(exist_ok=True, parents=False)

    # Start rum
    wandb.init(
        project="part_reg.run.7_optimal_transport_gw",
        config={
            **c.model_dump(mode="json"),
            "tag": run_tag,
            "save_dir": str(save_dir),
        },
    )

    (save_dir / "config.json").write_text(json.dumps(dict(wandb.config), indent=2))

    # Fit GW Optimal Transport

    tsr_he: list[torch.Tensor] = []
    tsr_phh3: list[torch.Tensor] = []
    inv_simpson_index: list[float] = []

    for e_he, c_he, e_phh3, c_ti_phh3 in zip(
        embeddings_he, coords_he, embeddings_phh3, coords_phh3
    ):
        t_he = torch.from_numpy(e_he)
        t_phh3 = torch.from_numpy(e_phh3)

        # plan[i][j] is how much mass goes from t_he[i] to t_phh3[j]
        # sum_j plan[i][j] = 1/len(t_he)
        # sum_i plan[i][j] = 1/len(t_phh3)
        if c.random_pairing:
            plan = torch.zeros((len(t_he), len(t_phh3)), dtype=torch.float32)
            rows = torch.arange(len(t_he))
            cols = torch.randperm(len(t_phh3), generator=t_prg)
            plan[rows, cols] = 1.0
            plan /= plan.sum()  # -> one 1/N entry per row&col, the rest is zero
        else:
            plan = ot_fused_gw(
                t_he,
                torch.from_numpy(c_he),
                t_phh3,
                torch.from_numpy(c_ti_phh3),
                alpha_=0.5,
            )

        # N_i = 1 / sum_j (p_ij)**2
        # p_ij = t_ij / sum_j t_ij
        # N_i = square(sum_j t_ij) / sum_j square(t_ij)
        inv_simpson_index.extend(
            (plan.sum(dim=-1).square() / plan.square().sum(dim=-1)).tolist()
        )

        # assert plan.shape == (len(t_he), len(t_phh3)), f"{plan.shape=} {t_he.shape=} {t_phh3.shape=}"

        # assert (plan.sum(0) - torch.ones(len(t_phh3)) / len(t_he)).abs().max() <= 1e-5
        # assert (plan.sum(1) - torch.ones(len(t_he)) / len(t_phh3)).abs().max() <= 1e-5

        # # how many phh3 to have all the mass transferred ?
        # plan_per_he = plan / plan.sum(dim=1, keepdim=True)
        # plan_per_he, _ = plan_per_he.sort(dim=1, descending=True)  # most massive elements last
        # plan_per_he.cumsum_(dim=1)  # cumulative sum --> reaches 1.0 increasingly fast
        # num_items = (plan_per_he >= 0.9).to(torch.int).argmax(dim=1) + 1
        # med_num_items = torch.median(num_items).item()
        # print(f"{med_num_items=}")

        # make a dataset from HE to corresponding PHH3 according to GW-OT
        tsr_he.append(t_he)
        tsr_phh3.append(t_phh3[plan.argmax(dim=1), :])

    hist = wandb.plot.histogram(
        wandb.Table(["Ni"], [[s] for s in inv_simpson_index]),
        "Ni",
        title="Avg Inv Simpson Index (per pair)",
    )
    wandb.log(
        {
            "my_histogram": hist,
            "median": np.median(inv_simpson_index),
            "std": np.std(inv_simpson_index, ddof=1),
        },
    )

    # Training from optimal transport

    dataset = TensorDataset(
        torch.cat(tsr_he),
        torch.cat(tsr_phh3),
    )

    dataloader = DataLoader(
        dataset,
        4096,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(0),
    )

    feat = embeddings_he[0].shape[-1]
    if c.arch == "linear":
        m_he_to_ihc = m.linear(feat).to("cuda:0")
        m_ihc_to_he = m.linear(feat).to("cuda:0")
    elif c.arch == "mlp":
        m_he_to_ihc = m.mlp(feat).to("cuda:0")
        m_ihc_to_he = m.mlp(feat).to("cuda:0")
    elif c.arch == "linear-eye":
        m_he_to_ihc = m.linear_eye(feat).to("cuda:0")
        m_ihc_to_he = m.linear_eye(feat).to("cuda:0")
    elif c.arch == "mlp-eye":
        m_he_to_ihc = m.mlp_eye(feat).to("cuda:0")
        m_ihc_to_he = m.mlp_eye(feat).to("cuda:0")
    elif c.arch == "mlp2-eye":
        m_he_to_ihc = m.mlp2_eye(feat).to("cuda:0")
        m_ihc_to_he = m.mlp2_eye(feat).to("cuda:0")
    else:
        raise ValueError(f"invalid value for {c.arch=}")

    def loss_fn_mse(emb_tr_: torch.Tensor, emb_other_: torch.Tensor):
        return torch.square(emb_tr_ - emb_other_).mean()

    optim = AdamW(Sequential(m_he_to_ihc, m_ihc_to_he).parameters(), lr=c.lr_max)
    scheduler = CosineAnnealingLR(optim, T_max=c.n_epochs, eta_min=c.lr_min)

    best_vl_loss = float("inf")
    best_vl_epoch = -1

    for epoch in range(c.n_epochs):
        loss_tr: dict[str, list[float]] = defaultdict(list)
        m_he_to_ihc.train()
        m_ihc_to_he.train()

        m_he_to_ihc.train()
        for emb_he, emb_phh3 in dataloader:
            optim.zero_grad()

            emb_he = torch.Tensor.to(emb_he, "cuda:0")
            emb_phh3 = torch.Tensor.to(emb_phh3, "cuda:0")

            loss_he = loss_fn_mse(m_he_to_ihc(emb_he), emb_phh3)

            loss_tr["loss_he"].append(float(loss_he.detach()))
            loss = loss_he

            if c.cyclic:
                loss_ihc = loss_fn_mse(emb_he, m_ihc_to_he(emb_phh3))
                cyc_he = torch.square(m_ihc_to_he(m_he_to_ihc(emb_he)) - emb_he).mean()
                cyc_ihc = torch.square(
                    m_he_to_ihc(m_ihc_to_he(emb_phh3)) - emb_phh3
                ).mean()

                loss_tr["loss_ihc"].append(float(loss_ihc.detach()))
                loss_tr["cyc_he"].append(float(cyc_he.detach()))
                loss_tr["cyc_ihc"].append(float(cyc_ihc.detach()))
                loss += loss_ihc + c.rec_coef * (cyc_he + cyc_ihc)

            loss.backward()
            optim.step()
            loss_tr["loss"].append(float(loss.detach()))

        acc_lst: list[float] = []
        acc_baseline_lst: list[float] = []
        loss_vl: dict[str, list[float]] = defaultdict(list)
        m_he_to_ihc.eval()
        for e_he_, c_he, e_ti_phh3_, c_ti_phh3, e_lm_phh3_ in zip(
            vl_lm_emb_he,
            vl_lm_coord_he,
            vl_ti_emb_phh3,
            vl_ti_tlbr_phh3,
            vl_lm_emb_phh3,
        ):
            emb_he = torch.from_numpy(e_he_).to("cuda:0")
            emb_phh3 = torch.from_numpy(e_lm_phh3_).to("cuda:0")
            with torch.no_grad():
                tr_he: torch.Tensor = m_he_to_ihc(emb_he)
            acc_lst.append(_img_acc(tr_he.cpu().numpy(), c_he, e_ti_phh3_, c_ti_phh3))
            acc_baseline_lst.append(_img_acc(e_he_, c_he, e_ti_phh3_, c_ti_phh3))

            with torch.no_grad():
                loss_he = loss_fn_mse(m_he_to_ihc(emb_he), emb_phh3)

            loss_vl["loss_he"].append(float(loss_he))
            loss = loss_he

            if c.cyclic:
                with torch.no_grad():
                    loss_ihc = loss_fn_mse(emb_he, m_ihc_to_he(emb_phh3))
                    cyc_he = torch.square(
                        m_ihc_to_he(m_he_to_ihc(emb_he)) - emb_he
                    ).mean()
                    cyc_ihc = torch.square(
                        m_he_to_ihc(m_ihc_to_he(emb_phh3)) - emb_phh3
                    ).mean()

                loss_vl["loss_ihc"].append(float(loss_ihc))
                loss_vl["cyc_he"].append(float(cyc_he))
                loss_vl["cyc_ihc"].append(float(cyc_ihc))
                loss += loss_ihc + c.rec_coef * (cyc_he + cyc_ihc)

            loss_vl["loss"].append(float(loss.detach()))

        scheduler.step()
        log_dct = {"tr_" + k: sum(v, 0.0) / len(v) for k, v in loss_tr.items()}
        log_dct.update({"vl_" + k: sum(v, 0.0) / len(v) for k, v in loss_vl.items()})
        log_dct["vl_acc"] = sum(acc_lst, 0.0) / len(acc_lst)
        log_dct["vl_acc_baseline"] = sum(acc_baseline_lst, 0.0) / len(acc_baseline_lst)
        wandb.log(log_dct)

        avg_vl_loss = sum(loss_vl["loss"], 0.0) / len(loss_vl["loss"])
        if avg_vl_loss < best_vl_loss:
            best_vl_loss = avg_vl_loss
            best_vl_epoch = epoch
            torch.save(m_he_to_ihc.state_dict(), save_dir / "best_m_he_to_phh3.pth")
            torch.save(m_ihc_to_he.state_dict(), save_dir / "best_m_phh3_to_he.pth")

        if epoch - best_vl_epoch > 20:
            break

    epoch = epoch  # type: ignore #  epoch is not unbound
    torch.save(
        {"feat": feat, "arch": c.arch, "epoch": epoch}, save_dir / "parameters.pth"
    )
    torch.save(m_he_to_ihc.state_dict(), save_dir / "m_he_to_phh3.pth")
    torch.save(m_ihc_to_he.state_dict(), save_dir / "m_phh3_to_he.pth")
