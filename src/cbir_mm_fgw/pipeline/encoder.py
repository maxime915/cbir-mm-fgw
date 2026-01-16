import functools
from typing import Callable, Literal, Protocol

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import AutoModel

from .dataloader import get_dataloader
from ._profiler import _profiler
from .utils import model_directory, _login
from ..mrvips import ImgInfo, VIPSImage
from ..plip import PLIP


DEVICE = "cuda:0"  # if torch.cuda.is_available() else "cpu"


@functools.lru_cache()
def _pipeline_vgg():
    cache = model_directory() / "vgg"
    try:
        cache.mkdir(exist_ok=False)
        hf_hub_download(
            "timm/vgg16.tv_in1k",
            filename="pytorch_model.bin",
            local_dir=cache,
            revision="b8d8aa2dd860af9233c8c67385a8097fd6c35d3f",
        )
    except FileExistsError:
        pass

    model = timm.create_model(
        "vgg16.tv_in1k",
        pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(cache / "pytorch_model.bin", map_location=DEVICE),
        strict=True,
    )

    model.eval()

    transform = transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def _predict(x_: torch.Tensor):
        output_ = model.forward_features(x_)
        return model.forward_head(output_, pre_logits=True).to(torch.float32)

    return transform, _predict, 224


@functools.lru_cache()
def _pipeline_plip():
    cache = model_directory() / "plip"
    try:
        cache.mkdir(exist_ok=False)
        for f in [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]:
            hf_hub_download(
                "vinid/plip",
                filename=f,
                local_dir=cache,
                revision="93491cf95091bbd92c439b36972fffb58312dc4d",
            )
    except FileExistsError:
        pass

    model = PLIP(cache)
    model.model = model.model.to(DEVICE)  # type: ignore
    model.model_name = "vinid/plip"  # type: ignore

    def _transform(x_: np.ndarray):
        return model.preprocess(images=x_, return_tensors="pt")["pixel_values"][0]  # type: ignore

    def _predict(x_: torch.Tensor):
        return model.model.get_image_features(pixel_values=x_).to(torch.float32)  # type: ignore

    return _transform, _predict, 224


@functools.lru_cache()
def _pipeline_core_h_optimus_0():
    cache = model_directory() / "h-optimus-0"
    try:
        cache.mkdir(exist_ok=False)
        for f in [
            "config.json",
            "pytorch_model.bin",
        ]:
            hf_hub_download(
                "bioptimus/H-optimus-0",
                filename=f,
                local_dir=cache,
                revision="a35a7fb622d30f806b231a6aeb9a953aebb12d74",
            )
    except FileExistsError:
        pass

    # build model
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=False,  # state loaded later
        init_values=1e-5,
        dynamic_img_size=False,
        cache_dir=cache,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(cache / "pytorch_model.bin", map_location=DEVICE),
        strict=True,
    )

    model.eval()

    transform = transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )

    return transform, model, 224


@functools.lru_cache()
def _pipeline_core_uni_1():
    cache = model_directory() / "uni-1"
    try:
        cache.mkdir(exist_ok=False)
        _login()
        hf_hub_download(
            "MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=cache,
            revision="b55a5ec6cade1a39edfe6534189a9b8ca7a022f0",
        )
    except FileExistsError:
        pass

    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model = model.to(DEVICE)  # type: ignore
    model.load_state_dict(
        torch.load(cache / "pytorch_model.bin", map_location=DEVICE),
        strict=True,
    )
    transform = transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model.eval()

    return transform, model, 224


@functools.lru_cache()
def _pipeline_core_uni_2_h():
    cache = model_directory() / "uni-2-h"
    try:
        cache.mkdir(exist_ok=False)
        _login()
        hf_hub_download(
            "MahmoodLab/UNI2-h",
            filename="pytorch_model.bin",
            local_dir=cache,
            revision="d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
        )
    except FileExistsError:
        pass

    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,  # type: ignore
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs).to(DEVICE)  # type: ignore
    model.load_state_dict(
        torch.load(cache / "pytorch_model.bin", map_location=DEVICE),
        strict=True,
    )
    transform = transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model.eval()

    return transform, model, 224


@functools.lru_cache()
def _pipeline_core_gigapath():
    cache = model_directory() / "giga-path"
    try:
        cache.mkdir(exist_ok=False)
        _login()
        for f in [
            "config.json",
            "pytorch_model.bin",
        ]:
            hf_hub_download(
                "prov-gigapath/prov-gigapath",
                filename=f,
                local_dir=cache,
                revision="eba85dd46097c3eedfcc2a3a9a930baecb6bcc19",
            )
    except FileExistsError:
        pass

    model = timm.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        pretrained=False,
        cache_dir=cache,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(cache / "pytorch_model.bin", map_location=DEVICE),
        strict=True,
    )
    transform = transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model.eval()

    return transform, model, 224


@functools.lru_cache()
def _pipeline_core_keep():
    _login()
    cache = model_directory() / "keep"
    cache.mkdir(exist_ok=True)
    model = AutoModel.from_pretrained(
        "Astaxanthin/KEEP",
        trust_remote_code=True,
        cache_dir=cache,
        revision="1776acd9d33abecc142378d293d10c9b5a13662c",
    )
    model = model.to(DEVICE)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model.eval()

    return transform, model.encode_image, 224


@functools.lru_cache()
def _pipeline_core_Virchow():
    cache = model_directory() / "Virchow"
    try:
        cache.mkdir(exist_ok=False)
        _login()
        for f in [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
        ]:
            hf_hub_download(
                "paige-ai/Virchow",
                filename=f,
                local_dir=cache,
                revision="19eebc84ae33e79f1b2d866e6ff90ae50e522f9a",
            )
    except FileExistsError:
        pass

    model = timm.create_model(
        "hf_hub:paige-ai/Virchow",
        pretrained=False,
        cache_dir=cache,
        mlp_layer=timm.layers.SwiGLUPacked,  # type: ignore
        act_layer=torch.nn.SiLU,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(cache / "pytorch_model.bin", map_location=DEVICE),
        strict=True,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model.eval()

    def _predict(image: torch.Tensor):
        # code from the model card

        output = model(image)  # size: 1 x 257 x 1280

        class_token = output[:, 0]  # size: 1 x 1280
        patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

        # concatenate class token and average pool of patch tokens
        embedding = torch.cat(
            [class_token, patch_tokens.mean(1)], dim=-1
        )  # size: 1 x 2560
        return embedding

    return transform, _predict, 224


class Pipeline(Protocol):
    def __call__(self, images: list[ImgInfo] | list[VIPSImage]) -> np.ndarray:
        "make a predictions from a list of images"
        ...


def random_pipeline(images: list[ImgInfo] | list[VIPSImage]):
    return np.random.rand(len(images), 64).astype(np.float32)


def build_pipeline(
    transform: Callable[[np.ndarray], torch.Tensor],
    model: Callable[[torch.Tensor], torch.Tensor],
    dataloader: Literal["pytorch", "naive", "queue"],
    batch_size: int | None,
):
    def make_prediction(images: list[ImgInfo] | list[VIPSImage]) -> np.ndarray:
        dl = get_dataloader(dataloader, transform, images, batch_size)

        with _profiler.profile("prediction"):
            res: list[torch.Tensor] = []
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16),
                torch.inference_mode(),
            ):
                img_batch: torch.Tensor
                for img_batch in dl:
                    with _profiler.profile("prediction/copy-to-gpu"):
                        img_batch = img_batch.to(DEVICE, non_blocking=True)
                    with _profiler.profile("prediction/model"):
                        img_batch = model(img_batch)
                    with _profiler.profile("prediction/copy-from-gpu"):
                        img_batch = img_batch.detach().to("cpu", non_blocking=True)
                    res.append(img_batch)

            with _profiler.profile("prediction/sync"):
                torch.cuda.synchronize(0)

            with _profiler.profile("prediction/conversion"):
                return np.concatenate([r.numpy() for r in res])

    return make_prediction


Model = Literal[
    "h-optimus-0",
    "uni-2-h",
    "uni-1",
    "vgg",
    "gigapath",
    "plip",
    "keep",
    "virchow",
    "random",
]


def get_pipeline(
    model_name: Model | str,
    dataloader: Literal["pytorch", "naive", "queue"] = "queue",
    batch_size: int | None = None,
) -> tuple[Pipeline, int]:
    transform: transforms.Compose
    if model_name == "h-optimus-0":
        transform, model, patch_size = _pipeline_core_h_optimus_0()
    elif model_name == "uni-2-h":
        transform, model, patch_size = _pipeline_core_uni_2_h()
    elif model_name == "uni-1":
        transform, model, patch_size = _pipeline_core_uni_1()
    elif model_name == "vgg":
        transform, model, patch_size = _pipeline_vgg()
    elif model_name == "gigapath":
        transform, model, patch_size = _pipeline_core_gigapath()
    elif model_name == "plip":
        transform, model, patch_size = _pipeline_plip()  # type: ignore
    elif model_name == "keep":
        transform, model, patch_size = _pipeline_core_keep()
    elif model_name == "virchow":
        transform, model, patch_size = _pipeline_core_Virchow()
    elif model_name == "random":
        return random_pipeline, -1
    else:
        raise ValueError(f"{model_name=} is invalid")

    return build_pipeline(transform, model, dataloader, batch_size), patch_size
