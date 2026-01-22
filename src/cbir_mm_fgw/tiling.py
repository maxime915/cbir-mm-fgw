"tiling: image tiling and patch extraction"

from typing import Literal, overload

import numpy as np

from .mrvips import ImgInfo, MRImage, get_patch, VIPSImage


def _resolution_px_per_um(img: MRImage):
    "computes the resolution in pixels per microns"
    res_x = img.raw().get("xres")
    res_y = img.raw().get("yres")

    if res_x != res_y:
        raise ValueError(f"inconsistent: {res_x=!r} {res_y=!r}")

    res_unit = img.raw().get("resolution-unit")
    if res_unit.lower() != "cm":  # type: ignore
        raise ValueError(f"expected centimeter resolution, got: {res_unit=!r}")

    if not isinstance(res_x, float):
        raise TypeError(f"{res_x=!r} should be a float")

    pixels_per_microns = res_x * 1e-4
    return pixels_per_microns


def _resize(img: MRImage, size_microns: float, size_pixels: int):
    pixels_per_microns = _resolution_px_per_um(img)

    crop_size_px = round(size_microns * pixels_per_microns)

    rescaling = size_pixels / crop_size_px
    resized = img.resize(rescaling)

    return crop_size_px, rescaling, resized


def resized_n_pixels(path: str, size_microns: float, size_pixels: int):
    """compute the number of pixels that would be if the image was resized such
    that the size in microns would correspond to the size in pixels"""
    image = MRImage.new_from_file(path)
    # resolution (level 0)
    pixels_per_microns = _resolution_px_per_um(image)
    # area (level 0)
    area = image.width * image.height

    crop_size_px = round(size_microns * pixels_per_microns)
    rescaling = size_pixels / crop_size_px

    return area * rescaling * rescaling


def rasterize_coordinates(
    img: MRImage,
    patch_size_microns: float,
    patch_size_pixels: int,
    coords_rel_yx: list[tuple[float, float]],
):
    "map relative coordinates to rasterized pixel coordinates"
    _, _, resized = _resize(img, patch_size_microns, patch_size_pixels)

    return [
        (round(y * resized.height), round(x * resized.width))  # type: ignore
        for y, x in coords_rel_yx
    ]


TilingMode = Literal["basic", "overlap"]


@overload
def make_tiles(
    path: str,
    tile_size_microns: float,
    tile_size_pixels: int,
    tiling_mode: TilingMode,
    *,
    read_img: Literal[False],
) -> tuple[np.ndarray, list[ImgInfo]]: ...


@overload
def make_tiles(
    path: str,
    tile_size_microns: float,
    tile_size_pixels: int,
    tiling_mode: TilingMode,
    *,
    read_img: Literal[True],
) -> tuple[np.ndarray, list[VIPSImage]]: ...


@overload
def make_tiles(
    path: str,
    tile_size_microns: float,
    tile_size_pixels: int,
    tiling_mode: TilingMode,
    *,
    read_img: bool,
) -> tuple[np.ndarray, list[VIPSImage]] | tuple[np.ndarray, list[ImgInfo]]: ...


def make_tiles(
    path: str,
    tile_size_microns: float,
    tile_size_pixels: int,
    tiling_mode: Literal["basic", "overlap"],
    *,
    read_img: bool,
):
    img = MRImage.new_from_file(path, access="sequential")
    return make_tiles_offset(
        path,
        tile_size_microns=tile_size_microns,
        tile_size_pixels=tile_size_pixels,
        tiling_mode=tiling_mode,
        tlbr_px=(0, 0, img.height, img.width),
        read_img=read_img,
    )


TLBR = tuple[int, int, int, int]


def _overlap(left: TLBR, right: TLBR):
    if left[3] <= right[1] or right[3] <= left[1]:
        return False
    if left[2] <= right[0] or right[2] <= left[0]:
        return False
    return True


def _cluster_tiles(windows_tlbr: list[TLBR]):
    graph: dict[TLBR, list[TLBR]] = {n: [] for n in windows_tlbr}

    # build a graph and use DFS to build connected components
    for i, left in enumerate(windows_tlbr):
        for right in windows_tlbr[i + 1 :]:
            if _overlap(left, right):
                graph[left].append(right)
                graph[right].append(left)

    visited = {k: False for k in windows_tlbr}

    def dfs_visit(node_: TLBR, cluster_: list[TLBR]):
        visited[node_] = True
        cluster_.append(node_)
        for neighbor in graph[node_]:
            if visited[neighbor]:
                continue
            dfs_visit(neighbor, cluster_)
        return cluster_

    return [dfs_visit(node, []) for node in windows_tlbr if not visited[node]]


def _split_cluster(cluster: list[TLBR]):
    "transform a cluster of overlapping rectangles into a set of rectangles that covers the same area without any overlap"

    if len(cluster) < 2:
        return cluster

    # build a grid by every possible step
    all_ys = sorted(set(sum((tlbr[0::2] for tlbr in cluster), ())))
    all_xs = sorted(set(sum((tlbr[1::2] for tlbr in cluster), ())))
    y_idx = {y: i for i, y in enumerate(all_ys)}
    x_idx = {x: i for i, x in enumerate(all_xs)}

    # for every possible coordinate, mark it if there is one rectangle that covers it
    splat: list[TLBR] = []
    for top, left, bottom, right in cluster:
        for i_x in range(x_idx[left], x_idx[right]):
            for i_y in range(y_idx[top], y_idx[bottom]):
                splat.append(
                    (all_ys[i_y], all_xs[i_x], all_ys[i_y + 1], all_xs[i_x + 1])
                )

    # remove duplicates
    return sorted(set(splat))


def _get_rect_hull(cluster: list[TLBR]) -> TLBR:
    t, l_, b, r = cluster[0]
    for rect in cluster[1:]:
        t = min(t, rect[0])
        l_ = min(l_, rect[1])
        b = max(b, rect[2])
        r = max(r, rect[3])
    return t, l_, b, r


def remove_overlap(windows_tlbr: list[TLBR]):
    """
    We receive a list (order doesn't matter) of tiles (square, but we way
    assume rectangle to be forgiving) which can overlap. The goal is to provide
    other rectangles that: (1) cover the same area (2) do not overlap
    """

    # step 1: cluster the windows
    clusters = _cluster_tiles(windows_tlbr)

    # step 2: split each cluster into smaller rectangles that do not overlap
    clusters = [_split_cluster(cl) for cl in clusters]

    # step 3: make a list containing all smaller parts
    return sum(clusters, [])


def make_tiles_offset_multi(
    path: str,
    tile_size_microns: float,
    tile_size_pixels: int,
    tiling_mode: Literal["basic", "overlap"],
    tlbr_px: list[TLBR],
    *,
    read_img: bool,
    auto_clamp: bool = False,
):
    img = MRImage.new_from_file(path, access="sequential")
    if not tlbr_px:
        tlbr_px = [(0, 0, img.height, img.width)]

    # step 1: cluster the windows
    clusters = _cluster_tiles(tlbr_px)

    # step 2: get the hull of each cluster
    hulls = [_get_rect_hull(cl) for cl in clusters]

    tiling_res = [
        make_tiles_offset(
            path=path,
            tile_size_microns=tile_size_microns,
            tile_size_pixels=tile_size_pixels,
            tiling_mode=tiling_mode,
            tlbr_px=hull,
            read_img=False,
            auto_clamp=auto_clamp,
        )
        for hull in hulls
    ]

    # flatten
    tiles_tlbr = np.concatenate([p[0] for p in tiling_res])
    tiles: list[ImgInfo] = sum((p[1] for p in tiling_res), [])  # type: ignore

    # NOTE the hulls of two non overlapping clusters may overlap
    # we need to get rid of all duplicates
    unique_tlbr: set[TLBR] = set()
    good_indices: list[int] = []
    for idx, tile_ in enumerate(tiles_tlbr):
        tlbr: TLBR = tuple(tile_)
        if tlbr in unique_tlbr:
            continue
        unique_tlbr.add(tlbr)
        good_indices.append(idx)

    tiles_tlbr = tiles_tlbr[good_indices]
    tiles = [tiles[i] for i in good_indices]

    if read_img:
        _, _, resized = _resize(img, tile_size_microns, tile_size_pixels)
        return tiles_tlbr, [
            get_patch(resized, t_.left_top, t_.patch_size, t_.angle) for t_ in tiles
        ]

    return tiles_tlbr, tiles


def make_tiles_offset(
    path: str,
    tile_size_microns: float,
    tile_size_pixels: int,
    tiling_mode: Literal["basic", "overlap"],
    tlbr_px: tuple[int, int, int, int] | None,
    *,
    read_img: bool,
    auto_clamp: bool = False,
):
    img = MRImage.new_from_file(path, access="sequential")
    crop_size_px, rescaling, resized = _resize(img, tile_size_microns, tile_size_pixels)
    h = crop_size_px // 2

    if tlbr_px is None:
        y_px_lo = x_px_lo = 0
        y_px_hi = img.height
        x_px_hi = img.width
    else:
        y_px_lo, x_px_lo, y_px_hi, x_px_hi = tlbr_px

    if auto_clamp:
        y_px_lo, x_px_lo = max(0, y_px_lo), max(0, x_px_lo)
        y_px_hi, x_px_hi = min(img.height, y_px_hi), min(img.width, x_px_hi)

    # matching lists to store the tiles (top, left, bottom, right)
    tile_coord_tlbr_lst: list[tuple[float, float, float, float]] = []
    tile_lst_info: list[ImgInfo] = []
    tile_lst_img: list[VIPSImage] = []

    def _add(y: int, x: int):
        "coords in px from the full resolution"
        assert y < img.height and x < img.width

        # map coordinates: um to px in resized
        tl_xy = int(x * rescaling), int(y * rescaling)

        # add the resized patch for the prediction
        if read_img:
            tile_lst_img.append(get_patch(resized, tl_xy, tile_size_pixels, 0.0))
        else:
            tile_lst_info.append(ImgInfo(path, rescaling, tl_xy, tile_size_pixels, 0.0))

        # add the coordinates in the raw resolution
        tile_coord_tlbr_lst.append((y, x, y + crop_size_px, x + crop_size_px))

    for y in range(y_px_lo, y_px_hi - 1, crop_size_px):
        for x in range(x_px_lo, x_px_hi - 1, crop_size_px):
            _add(y, x)
            if tiling_mode == "overlap":
                if y < y_px_hi - h - 1:
                    _add(y + h, x)
                if x < x_px_hi - h - 1:
                    _add(y, x + h)
                if y < y_px_hi - h - 1 and x < x_px_hi - h - 1:
                    _add(y + h, x + h)

    # add the missing patches that overlap with the top and left edges
    if tiling_mode == "overlap":
        _add(y_px_lo - h, x_px_lo - h)
        for x in range(x_px_lo, x_px_hi - 1, crop_size_px):
            if x < x_px_hi - 1:
                _add(y_px_lo - h, x)
            if x < x_px_hi - h - 1:
                _add(y_px_lo - h, x + h)
        for y in range(y_px_lo, y_px_hi - 1, crop_size_px):
            if y < y_px_hi - 1:
                _add(y, x_px_lo - h)
            if y < y_px_hi - h - 1:
                _add(y + h, x_px_lo - h)

    tile_coord_tlbr = np.stack(tile_coord_tlbr_lst)

    if read_img:
        return tile_coord_tlbr, tile_lst_img
    return tile_coord_tlbr, tile_lst_info


@overload
def make_patches(
    path: str,
    patch_size_microns: float,
    patch_size_pixels: int,
    key_points_yx: list[tuple[int, int]],
    angle: float,
    *,
    read_img: Literal[False],
    allow_overflow: bool = True,
) -> list[ImgInfo]: ...


@overload
def make_patches(
    path: str,
    patch_size_microns: float,
    patch_size_pixels: int,
    key_points_yx: list[tuple[int, int]],
    angle: float,
    *,
    read_img: Literal[True],
    allow_overflow: bool = True,
) -> list[VIPSImage]: ...


@overload
def make_patches(
    path: str,
    patch_size_microns: float,
    patch_size_pixels: int,
    key_points_yx: list[tuple[int, int]],
    angle: float,
    *,
    read_img: bool,
    allow_overflow: bool = True,
) -> list[VIPSImage] | list[ImgInfo]: ...


def make_patches(
    path: str,
    patch_size_microns: float,
    patch_size_pixels: int,
    key_points_yx: list[tuple[int, int]],
    angle: float,
    *,
    read_img: bool,
    allow_overflow: bool = True,
):
    img = MRImage.new_from_file(path)
    crop_0, rescaling, resized = _resize(img, patch_size_microns, patch_size_pixels)
    h = patch_size_pixels // 2
    h0 = crop_0 // 2

    def _inside(y: int, x: int):
        return not (
            y - h0 < 0 or y + h0 >= img.height or x - h0 < 0 or x + h0 >= img.width
        )

    if read_img:
        return [
            get_patch(
                resized,
                (x * rescaling - h, y * rescaling - h),
                patch_size_pixels,
                angle,
            )
            for (y, x) in key_points_yx
            if allow_overflow or _inside(y, x)
        ]

    return [
        ImgInfo(
            path,
            rescaling,
            (x * rescaling - h, y * rescaling - h),
            patch_size_pixels,
            angle,
        )
        for (y, x) in key_points_yx
        if allow_overflow or _inside(y, x)
    ]
