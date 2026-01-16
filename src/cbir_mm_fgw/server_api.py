"""start_simple_server: starts a processing server able to do a 1-level search"""

import os
import time
from typing import Literal, TypedDict

import faiss
import numpy as np
from pydantic import BaseModel

from .pipeline import Modality, _profiler, get_corrector, get_pipeline
from .profiler import TagProfiler
from .tiling import make_patches, make_tiles_offset

start_time = time.time()


def infos(model_name: str, corrector_tag: str | None):
    "useful information about the server"

    return {
        "model": model_name,
        "corrector_tag": corrector_tag,
        "MODELS_CACHE": os.environ.get("MODELS_CACHE"),
        "BATCH_SIZE": os.environ.get("BATCH_SIZE"),
        "running_time": time.time() - start_time,
        # database size ?
        "time": dict(_profiler),
    }


def clear_profiler():
    _profiler.clear()
    return "cleared"


class Query(BaseModel):
    model: str
    corrector_tag: str
    tiling_size_microns: float
    tiling_size_pixels: int
    tiling_mode: Literal["basic", "overlap"]
    query_image_path: str
    query_patches_rotation: float
    query_image_key_points_yx: list[tuple[int, int]]
    query_image_modality: Modality
    target_image_path: str
    target_image_modality: Modality
    limit_results_per_query: int


class Result(TypedDict):
    time: dict[str, list[float]]
    results: list[list[tuple[float, float, float, float]]]
    distances: list[list[float]]
    infos: dict[str, str | int | float | bool]


def _normalized(x: np.ndarray):
    "return a normalized copy of an array usable for indexing"
    x = x.copy()
    faiss.normalize_L2(x)
    return x


def _match(
    tiles: np.ndarray, patches: np.ndarray, limit: int, profiler: TagProfiler | None
) -> tuple[np.ndarray, np.ndarray]:
    if profiler is None:
        profiler = TagProfiler()

    with profiler.profile("index/creation"):
        idx = faiss.IndexFlatIP(tiles.shape[1])  # type:ignore
        idx.add(_normalized(tiles))  # type: ignore
    with profiler.profile("index/search"):
        limit = min(limit, len(tiles))
        distances, indices = idx.search(_normalized(patches), limit)  # type: ignore

    return 1.0 - distances, indices


def prediction(query: Query) -> Result:
    pipeline, _ = get_pipeline(query.model)
    corr_query, corr_target = get_corrector(
        query.corrector_tag, query.query_image_modality, query.target_image_modality
    )
    profiler = TagProfiler()

    # make tiles out of the target image
    with profiler.profile("tiling/coordinates"):
        tiles_tlbr, tiles_img = make_tiles_offset(
            query.target_image_path,
            query.tiling_size_microns,
            query.tiling_size_pixels,
            query.tiling_mode,
            tlbr_px=None,
            read_img=False,
        )
    with profiler.profile("tiling/embeddings"):
        tiles = pipeline(tiles_img)
        tiles = corr_target(tiles)

    # make patches around the key-points in the query image
    with profiler.profile("query/coordinates"):
        patches_img = make_patches(
            query.query_image_path,
            query.tiling_size_microns,
            query.tiling_size_pixels,
            query.query_image_key_points_yx,
            query.query_patches_rotation,
            read_img=False,
        )
    with profiler.profile("query/embeddings"):
        patches = pipeline(patches_img)
        patches = corr_query(patches)

    distances, indices = _match(tiles, patches, query.limit_results_per_query, profiler)

    # return the appropriate values
    return {
        "time": dict(profiler),
        "results": [
            [tuple(tiles_tlbr[k].tolist()) for k in p_indices] for p_indices in indices
        ],
        "distances": [p_dist.tolist() for p_dist in distances],
        "infos": {"n-tiles": len(tiles_img), "n-patches": len(patches_img)},
    }
