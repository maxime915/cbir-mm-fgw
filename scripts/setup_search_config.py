import argparse
import re

import runexp

from cbir_mm_fgw.pipeline.utils import model_directory

# search for everything in model cache
def main():
    parser = argparse.ArgumentParser("setup_search_config.py")
    parser.add_argument("--ignore-pattern", type=str, default=None, help="re pattern to ignore")

    args = runexp.parse(parser)

    model_dir = model_directory()
    if not model_dir.is_dir():
        raise RuntimeError(f"model directory not properly setup: {model_dir}")

    if args.ignore_pattern is not None:
        pattern = re.compile(args.ignore_pattern)
    else:
        # https://stackoverflow.com/a/1845097/5770818
        pattern = re.compile(r'(?!x)x')

    fgw_dir = model_dir / "corrector_7-gwot-allpix-patches"
    zscore_dir = model_dir / "corrector_z-score-corr"

    items: list[tuple[str, str, str]] = []

    for conf in fgw_dir.iterdir():
        if pattern.match(str(conf)) is not None:
            continue

        name = conf.name
        model, name = name.split("_px")
        ts_px, name = name.split("_um")
        ts_um, _ = name.split("_")
        del name

        # get tag
        tag = conf.relative_to(model_dir)
        items.append((model, ts_um, str(tag)))

    for conf in zscore_dir.iterdir():
        if pattern.match(str(conf)) is not None:
            continue

        name = conf.name
        model, name = name.split("_px")
        ts_px, name = name.split("_um")
        ts_um, *_ = name.split("_")
        del name

        # get tag
        tag = conf.relative_to(model_dir)
        items.append((model, ts_um, str(tag)))

    models, tile_sizes_um, corr_tags = zip(*items)
    print("  - model: [" + ", ".join(models) + "]")
    print("    tile_size_um: [" + ", ".join(tile_sizes_um) + "]")
    print("    corr_tag: [" + ", ".join(corr_tags) + "]")


if __name__ == "__main__":
    main()
