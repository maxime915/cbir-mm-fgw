import logging
import os
import pathlib

import huggingface_hub


def model_directory():
    env_dir = os.environ.get("MODELS_CACHE", "models-cache")
    env_dir_p = pathlib.Path(env_dir).expanduser().resolve()
    if not env_dir_p.is_dir():
        raise RuntimeError(
            f"Model cache directory does not exist: {env_dir_p}\n"
            "Create it manually or set MODELS_CACHE to an existing directory."
        )
    logging.debug(f"Using models-cache: {env_dir_p}")

    return env_dir_p


def _login():
    huggingface_hub.login(os.environ.get("HUGGING_FACE_TOKEN", ""))
