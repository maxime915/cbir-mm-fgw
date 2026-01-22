"mrvips: multi resolution image support for pyvips"

import math
from pathlib import Path
from typing import Any, NamedTuple

from pyvips import Image as VIPSImage


class MRImage:
    """Multi Resolution pyvips Image

    NOTE This makes the hypothesis that the first level is the highest resolution,
    and subsequent levels are down-scaled using consecutive powers of two. Failure
    of this hypothesis will result in the wrong layers used for resizing, which
    may make images blurry and/or affect performances.
    """

    def __init__(
        self,
        raw: VIPSImage,
        path: str | Path,
        kwargs: dict[str, Any],
        mr_keyword: str | None,
        n_levels: int | None,
    ):
        self._raw = raw
        self.path = path
        self.base_kwargs = kwargs
        self.mr_keyword = mr_keyword
        self.n_levels = n_levels
        assert (self.mr_keyword is None and self.n_levels is None) or (
            self.mr_keyword is not None and self.n_levels is not None
        )

    @property
    def width(self) -> int:
        return self._raw.width  # type: ignore

    @property
    def height(self) -> int:
        return self._raw.height  # type: ignore

    @staticmethod
    def new_from_file(vips_filename: str | Path, *, strict_mr: bool = False, **kwargs):
        img: VIPSImage = VIPSImage.new_from_file(vips_filename, **kwargs)  # type: ignore

        mr_keyword: str | None = None
        n_levels: int | None = None

        # other formats ?

        if img.get("vips-loader") == "openslideload":
            n_levels = int(img.get("openslide.level-count"))  # type: ignore
            mr_keyword = "level"

        elif img.get("vips-loader") == "tiffload":
            n_levels = int(img.get("n-pages"))  # type: ignore
            mr_keyword = "page"
        elif strict_mr:
            raise ValueError(f"format unrecognized for file: {vips_filename}")

        if mr_keyword is not None and mr_keyword in kwargs:
            raise ValueError(f"{kwargs} must not contain {mr_keyword}")

        return MRImage(img, vips_filename, kwargs, mr_keyword, n_levels)

    def resize(self, rescale: float | None) -> VIPSImage:
        kwargs = self.base_kwargs.copy()
        if rescale is not None and self.mr_keyword is not None:
            assert self.n_levels is not None

            level = 0
            for level_ in range(self.n_levels):
                # this is a heuristic, the real one will be evaluated later for correctness
                down_sample_ = 2**level_
                if down_sample_ * rescale < 1.1:
                    level = level_

            kwargs[self.mr_keyword] = level
            img: VIPSImage = VIPSImage.new_from_file(self.path, **kwargs)  # type: ignore
            down_sample: float = max(self._raw.width, self._raw.height) / max(img.width, img.height)  # type: ignore
            rescale *= down_sample

        else:
            img: VIPSImage = VIPSImage.new_from_file(self.path, **kwargs)  # type: ignore

        if rescale is not None:
            img = img.resize(rescale)  # type: ignore

        return img

    def raw(self) -> VIPSImage:
        return self._raw

    def resize_then_get_patch(
        self,
        rescale: float | None,
        top_left_xy: tuple[float, float],
        patch_size: int | tuple[int, int],
        angle: float,
    ):
        """crop an area in an image

        Args:
            rescale: float | None: parameter passed to img.resize before cropping
            top_left_xy (tuple[float, float]): coordinates of the top left corner (before any rotation), in the rescaled coordinates
            patch_size (int): dimension of the square patch after the crop
            angle (float): rotation angle (clockwise, around the center of the patch)

        Returns:
            pyvips.Image: cropped image
        """

        return get_patch(self.resize(rescale), top_left_xy, patch_size, angle)


def get_patch(
    img: VIPSImage, top_left_xy: tuple[float, float], patch_size: tuple[int, int] | int, angle: float
) -> VIPSImage:
    """crop an area in an image

    Args:
        img: Image: image to resize
        top_left_xy (tuple[float, float]): coordinates of the top left corner (before any rotation), in the rescaled coordinates
        patch_size (int): dimension of the square patch after the crop
        angle (float): rotation angle (clockwise, around the center of the patch)

    Returns:
        pyvips.Image: cropped image
    """

    c, s = math.cos(angle), math.sin(angle)
    if isinstance(patch_size, tuple):
        w, h = patch_size
    else:
        w = h = patch_size
    return img.affine(  # type: ignore
        [c, -s, s, c],
        idx=-top_left_xy[0] - w / 2.0,  # type: ignore
        idy=-top_left_xy[1] - h / 2.0,  # type: ignore
        odx=w / 2.0,  # type: ignore
        ody=h / 2.0,  # type: ignore
        oarea=(0, 0, w, h),  # type: ignore
    )


class ImgInfo(NamedTuple):
    path: str
    rescaling: float
    left_top: tuple[float, float]
    patch_size: int
    angle: float

    def read(self) -> VIPSImage:
        path, rescaling, (left, top), p, angle = self
        img = MRImage.new_from_file(path, access="random")
        crop = get_patch(img.resize(rescaling), (left, top), p, angle)
        return crop
