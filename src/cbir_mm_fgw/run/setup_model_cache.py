import argparse

from cbir_mm_fgw.pipeline.utils import model_directory
from cbir_mm_fgw.pipeline.encoder import Model, get_pipeline


def main():
    all_models: list[str] = list(Model.__args__)
    parser = argparse.ArgumentParser("model-cache setup")
    parser.add_argument("--allow-mkdir-parent", action="store_true")
    parser.add_argument("--model", nargs="*", choices=all_models, help="Models to download. Leave empty for all models")
    # either all models
    # or choose some models from the list
    parser.set_defaults(allow_mkdir_parent=False)

    args = parser.parse_args()
    if not args.model:
        args.model = all_models

    directory = model_directory(do_not_verify=True)
    directory.mkdir(parents=args.allow_mkdir_parent, exist_ok=True)

    # download all models one by one
    for model in args.model:
        get_pipeline(model, "naive", 8)
