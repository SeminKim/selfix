import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from glob import iglob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline

from flux.sampling import (
    denoise,
    denoise_aidi_e,
    denoise_fireflow,
    denoise_renoise,
    denoise_rf_solver,
    denoise_selfix,
    get_schedule,
    prepare,
    unpack,
)
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


@dataclass
class SamplingConfig:
    name: str = "flux-dev"
    source_img_dir: str = ""
    source_prompt: str = ""
    target_prompt: str = ""
    feature_path: str = "feature"
    guidance: float = 5.0
    num_steps: int | None = None
    inv_stop_ratio: float = 1.0
    inject_steps: int = 20
    start_layer: int = 20
    end_layer: int = 37
    output_dir: str = "output"
    output_prefix: str = "editing"
    method: str = "rf_solver"
    offload: bool = False
    reuse_v: int = 1
    editing_strategy: str = "replace_v"
    qkv_ratio: str = "1.0,1.0,1.0"
    seed: int = 0
    num_iterations: int = 10
    initialization: str = "clone"
    momentum: float = 0.0
    window_size: int = 1
    alpha1: float = 0.5
    delta: float = 1.0
    renoise_avg_mode: str = "uniform_all"
    renoise_first_step_range: str = "0,3"
    renoise_step_range: str = "7,9"
    renoise_enhance_editability: bool = False
    save_trace: bool = False
    save_latent_trajectory: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    add_sampling_metadata: bool = True
    enable_nsfw_filter: bool = True

    def __post_init__(self) -> None:
        if self.num_steps is None:
            self.num_steps = 4 if self.name == "flux-schnell" else 25
        self.num_steps = int(self.num_steps)
        self.num_iterations = max(int(self.num_iterations), 1)
        self.window_size = max(int(self.window_size), 1)
        if self.initialization not in {"euler", "clone"}:
            raise ValueError(f"initialization must be 'euler' or 'clone', got {self.initialization!r}")
        if not 0.0 <= float(self.momentum) < 1.0:
            raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {self.momentum}")
        if not 0.0 < self.alpha1 < 1.0:
            raise ValueError(f"alpha1 must satisfy 0 < alpha1 < 1, got {self.alpha1}")
        if self.delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {self.delta}")

    def resolved_seed(self) -> int | None:
        return self.seed if self.seed > 0 else None

    def qkv_ratio_values(self) -> list[float]:
        if isinstance(self.qkv_ratio, str):
            return [float(value.strip()) for value in self.qkv_ratio.split(",")]
        return [float(value) for value in self.qkv_ratio]


@dataclass
class ModelComponents:
    device: torch.device
    t5: Any
    clip: Any
    model: Any
    ae: Any
    nsfw_classifier: Any = None


@dataclass
class SamplingResult:
    image: Image.Image | None
    elapsed_seconds: float
    nsfw_score: float | None
    output_path: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    trace: dict[str, Any] | None = None
    inverted_latent: torch.Tensor | None = None
    latent_trajectory: dict[str, Any] | None = None


def encode(init_image: np.ndarray, torch_device: torch.device, ae: Any) -> torch.Tensor:
    init_tensor = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_tensor = init_tensor.unsqueeze(0).to(torch_device)
    return ae.encode(init_tensor).to(torch.bfloat16)


def resolve_pipeline_device(device: str | torch.device) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        return -1
    return device.index if device.index is not None else 0


def load_components(config: SamplingConfig) -> ModelComponents:
    if config.name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {config.name}, choose from {available}")

    torch_device = torch.device(config.device)
    t5 = load_t5(torch_device, max_length=256 if config.name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(config.name, device="cpu" if config.offload else torch_device)
    ae = load_ae(config.name, device="cpu" if config.offload else torch_device)

    if config.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)

    nsfw_classifier = None
    if config.enable_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=resolve_pipeline_device(torch_device),
        )

    return ModelComponents(
        device=torch_device,
        t5=t5,
        clip=clip,
        model=model,
        ae=ae,
        nsfw_classifier=nsfw_classifier,
    )


def get_denoise_strategy(name: str):
    denoise_strategies = {
        "reflow": denoise,
        "aidi_e": denoise_aidi_e,
        "selfix": denoise_selfix,
        "fpi": denoise_selfix,
        "renoise": denoise_renoise,
        "rf_solver": denoise_rf_solver,
        "fireflow": denoise_fireflow,
    }
    if name not in denoise_strategies:
        available = ", ".join(denoise_strategies.keys())
        raise ValueError(f"Unknown denoising method: {name}. Available: {available}")
    return denoise_strategies[name]


def crop_image_to_multiple_of_16(image: Image.Image) -> np.ndarray:
    init_image_array = np.array(image.convert("RGB"))
    height, width = init_image_array.shape[:2]
    new_height = height if height % 16 == 0 else height - height % 16
    new_width = width if width % 16 == 0 else width - width % 16
    return init_image_array[:new_height, :new_width, :]


def resolve_inv_stop_ratio(inv_stop_ratio: float) -> float:
    ratio = float(inv_stop_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"inv_stop_ratio must be in [0, 1], got {inv_stop_ratio}")
    return ratio


def get_effective_step_counts(num_steps: int, inv_stop_ratio: float) -> tuple[int, int]:
    ratio = resolve_inv_stop_ratio(inv_stop_ratio)
    effective_inv_steps = math.floor(num_steps * ratio)
    effective_gen_steps = effective_inv_steps if effective_inv_steps < num_steps else num_steps
    return effective_inv_steps, effective_gen_steps


def build_info(config: SamplingConfig) -> dict[str, Any]:
    feature_path = Path(config.feature_path)
    feature_path.mkdir(parents=True, exist_ok=True)
    return {
        "feature_path": str(feature_path),
        "feature": {},
        "inject_step": config.inject_steps,
        "start_layer_index": config.start_layer,
        "end_layer_index": config.end_layer,
        "reuse_v": config.reuse_v,
        "editing_strategy": config.editing_strategy,
        "qkv_ratio": config.qkv_ratio_values(),
        "inv_stop_ratio": config.inv_stop_ratio,
        "method": config.method,
        "num_iterations": config.num_iterations,
        "initialization": config.initialization,
        "momentum": config.momentum,
        "window_size": config.window_size,
        "alpha1": config.alpha1,
        "delta": config.delta,
        "renoise_avg_mode": config.renoise_avg_mode,
        "renoise_first_step_range": config.renoise_first_step_range,
        "renoise_step_range": config.renoise_step_range,
        "renoise_enhance_editability": config.renoise_enhance_editability,
        "save_trace": config.save_trace,
        "save_latent_trajectory": config.save_latent_trajectory,
    }


def get_output_pattern(config: SamplingConfig) -> str:
    prefix = (
        f"{config.output_prefix}"
        f"_inject_{config.inject_steps}"
        f"_start_layer_{config.start_layer}"
        f"_end_layer_{config.end_layer}"
    )
    return os.path.join(config.output_dir, prefix + "_img_{idx}.jpg")


def get_next_output_path(config: SamplingConfig) -> str:
    output_pattern = get_output_pattern(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        file_name
        for file_name in iglob(output_pattern.format(idx="*"))
        if re.search(r"img_[0-9]+\.jpg$", file_name)
    ]
    if files:
        index = max(int(file_name.split("_")[-1].split(".")[0]) for file_name in files) + 1
    else:
        index = 0
    return output_pattern.format(idx=index)


def save_image_with_metadata(
    image: Image.Image,
    output_path: str,
    model_name: str,
    source_prompt: str,
    add_sampling_metadata: bool,
) -> None:
    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    exif_data[ExifTags.Base.Model] = model_name
    if add_sampling_metadata:
        exif_data[ExifTags.Base.ImageDescription] = source_prompt
    image.save(output_path, exif=exif_data, quality=95, subsampling=0)


@torch.inference_mode()
def run_image_edit(
    config: SamplingConfig,
    components: ModelComponents | None = None,
) -> SamplingResult:
    torch.set_grad_enabled(False)
    if components is None:
        components = load_components(config)

    with Image.open(config.source_img_dir) as image_handle:
        init_image = crop_image_to_multiple_of_16(image_handle)

    height, width = init_image.shape[:2]
    opts = SamplingOptions(
        source_prompt=config.source_prompt,
        target_prompt=config.target_prompt,
        width=width,
        height=height,
        num_steps=config.num_steps,
        guidance=config.guidance,
        seed=config.resolved_seed(),
    )

    if opts.seed is None:
        opts.seed = torch.Generator(device="cpu").seed()

    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)

    print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
    start_time = time.perf_counter()

    if config.offload:
        components.model.cpu()
        torch.cuda.empty_cache()
        components.ae.encoder.to(components.device)

    encoded_image = encode(init_image, components.device, components.ae)

    if config.offload:
        components.ae = components.ae.cpu()
        torch.cuda.empty_cache()
        components.t5 = components.t5.to(components.device)
        components.clip = components.clip.to(components.device)

    info = build_info(config)
    inp = prepare(components.t5, components.clip, encoded_image, prompt=opts.source_prompt)
    inp_target = prepare(components.t5, components.clip, encoded_image, prompt=opts.target_prompt)
    full_inv_timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(config.name != "flux-schnell"))
    inv_stop_ratio = resolve_inv_stop_ratio(config.inv_stop_ratio)
    effective_inv_steps, effective_gen_steps = get_effective_step_counts(opts.num_steps, inv_stop_ratio)
    partial_inversion_enabled = effective_inv_steps < opts.num_steps
    if partial_inversion_enabled:
        inv_timesteps = full_inv_timesteps[-(effective_inv_steps + 1):]
        gen_timesteps = inv_timesteps
    else:
        inv_timesteps = full_inv_timesteps
        gen_timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(config.name != "flux-schnell"))
    if config.save_trace:
        trace = info.setdefault("trace", {})
        trace["inv_stop_ratio"] = inv_stop_ratio
        trace["effective_inv_steps"] = effective_inv_steps
        trace["effective_gen_steps"] = effective_gen_steps
        trace["partial_inversion_enabled"] = partial_inversion_enabled
        trace["generation_start_timestep"] = float(gen_timesteps[0])

    if config.offload:
        components.t5 = components.t5.cpu()
        components.clip = components.clip.cpu()
        torch.cuda.empty_cache()
        components.model = components.model.to(components.device)

    denoise_strategy = get_denoise_strategy(config.method)
    z, info = denoise_strategy(components.model, **inp, timesteps=inv_timesteps, guidance=1, inverse=True, info=info)
    inverted_latent = z.detach().cpu()
    inp_target["img"] = z
    x, _ = denoise_strategy(
        components.model,
        **inp_target,
        timesteps=gen_timesteps,
        guidance=config.guidance,
        inverse=False,
        info=info,
    )

    if config.offload:
        components.model.cpu()
        torch.cuda.empty_cache()
        components.ae.decoder.to(x.device)

    batch_x = unpack(x.float(), opts.height, opts.width)
    decoded = batch_x[0].unsqueeze(0)

    with torch.autocast(device_type=components.device.type, dtype=torch.bfloat16):
        decoded = components.ae.decode(decoded)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time
    decoded = decoded.clamp(-1, 1)
    decoded = embed_watermark(decoded.float())
    decoded = rearrange(decoded[0], "c h w -> h w c")
    image = Image.fromarray((127.5 * (decoded + 1.0)).cpu().byte().numpy())

    nsfw_score = None
    trace = info.get("trace")
    latent_trajectory = info.get("latent_trajectory")
    if components.nsfw_classifier is not None:
        scores = [item["score"] for item in components.nsfw_classifier(image) if item["label"] == "nsfw"]
        nsfw_score = scores[0] if scores else None
        if nsfw_score is not None and nsfw_score >= NSFW_THRESHOLD:
            print("Your generated image may contain NSFW content.")
            if config.offload:
                components.ae = components.ae.cpu()
                torch.cuda.empty_cache()
            return SamplingResult(
                image=None,
                elapsed_seconds=elapsed_seconds,
                nsfw_score=nsfw_score,
                width=opts.width,
                height=opts.height,
                seed=opts.seed,
                trace=trace,
                inverted_latent=inverted_latent,
                latent_trajectory=latent_trajectory,
            )

    if config.offload:
        components.ae = components.ae.cpu()
        torch.cuda.empty_cache()

    return SamplingResult(
        image=image,
        elapsed_seconds=elapsed_seconds,
        nsfw_score=nsfw_score,
        width=opts.width,
        height=opts.height,
        seed=opts.seed,
        trace=trace,
        inverted_latent=inverted_latent,
        latent_trajectory=latent_trajectory,
    )


def save_sampling_result(config: SamplingConfig, result: SamplingResult) -> str:
    if result.image is None:
        raise ValueError("Cannot save an empty sampling result.")
    output_path = get_next_output_path(config)
    print(f"Done in {result.elapsed_seconds:.1f}s. Saving {output_path}")
    save_image_with_metadata(
        result.image,
        output_path,
        config.name,
        config.source_prompt,
        config.add_sampling_metadata,
    )
    result.output_path = output_path
    return output_path


def config_from_args(args: argparse.Namespace) -> SamplingConfig:
    config_dict = vars(args).copy()
    config_dict.pop("disable_nsfw_filter", None)
    config_dict["renoise_enhance_editability"] = config_dict.pop("enable_renoise_editability", False)
    return SamplingConfig(**config_dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SelFix image editing")
    parser.add_argument("--name", default="flux-dev", type=str, help="flux model")
    parser.add_argument("--source_img_dir", default="", type=str, help="path of the source image")
    parser.add_argument("--source_prompt", type=str, default="", help="source image prompt")
    parser.add_argument("--target_prompt", type=str, default="", help="target editing prompt")
    parser.add_argument("--feature_path", type=str, default="feature", help="path to save injected features")
    parser.add_argument("--guidance", type=float, default=5, help="guidance scale")
    parser.add_argument("--num_steps", type=int, default=25, help="number of inversion and generation timesteps")
    parser.add_argument("--inv_stop_ratio", type=float, default=1.0, help="fraction of inversion updates to execute")
    parser.add_argument("--inject_steps", type=int, default=20, help="number of timesteps that apply feature sharing")
    parser.add_argument("--start_layer", type=int, default=20, help="first transformer layer for feature sharing")
    parser.add_argument("--end_layer", type=int, default=37, help="last transformer layer for feature sharing")
    parser.add_argument("--output_dir", default="output", type=str, help="path of the edited image")
    parser.add_argument("--output_prefix", default="editing", type=str, help="prefix name of the edited image")
    parser.add_argument(
        "--method",
        default="rf_solver",
        choices=("selfix", "fpi", "reflow", "rf_solver", "fireflow", "renoise", "aidi_e"),
        type=str,
        help="sampling method",
    )
    parser.add_argument("--offload", action="store_true", help="offload modules to CPU between stages")
    parser.add_argument("--reuse_v", type=int, default=1, help="reuse V during inversion and reconstruction/editing")
    parser.add_argument("--editing_strategy", default="replace_v", type=str, help="feature editing strategy")
    parser.add_argument("--qkv_ratio", type=str, default="1.0,1.0,1.0", help="comma-separated Q,K,V ratios")
    parser.add_argument("--seed", type=int, default=0, help="random seed; 0 selects a random seed")
    parser.add_argument("--num_iterations", type=int, default=10, help="number of inversion iterations")
    parser.add_argument(
        "--initialization",
        type=str,
        choices=("euler", "clone"),
        default="clone",
        help="fixed-point initialization",
    )
    parser.add_argument("--momentum", type=float, default=0.0, help="fixed-point relaxation momentum")
    parser.add_argument("--window_size", type=int, default=1, help="SelFix straightness window size")
    parser.add_argument("--alpha1", type=float, default=0.5, help="SelFix alpha_1")
    parser.add_argument("--delta", type=float, default=1.0, help="SelFix alpha schedule delta")
    parser.add_argument("--renoise_avg_mode", type=str, choices=("uniform_all", "paper"), default="uniform_all")
    parser.add_argument("--renoise_first_step_range", type=str, default="0,3")
    parser.add_argument("--renoise_step_range", type=str, default="7,9")
    parser.add_argument("--enable_renoise_editability", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device")
    parser.add_argument("--save_trace", action="store_true", help="save iterative trace data in the result object")
    parser.add_argument("--save_latent_trajectory", action="store_true", help="store every latent in the result object")
    parser.add_argument("--disable_nsfw_filter", action="store_true", help="disable the NSFW classifier gate")
    return parser


def main(args: argparse.Namespace | SamplingConfig | None = None) -> SamplingResult:
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    if isinstance(args, argparse.Namespace):
        config = config_from_args(args)
        config.enable_nsfw_filter = not args.disable_nsfw_filter
    elif isinstance(args, SamplingConfig):
        config = args
    else:
        raise TypeError(f"Unsupported args type: {type(args)!r}")

    result = run_image_edit(config)
    if result.image is not None:
        save_sampling_result(config, result)
    return result


if __name__ == "__main__":
    main()
