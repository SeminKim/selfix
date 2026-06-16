import argparse
import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from edit import SamplingConfig, get_effective_step_counts

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ALLOWED_METHODS = {
    "selfix",
    "fpi",
    "reflow",
    "rf_solver",
    "fireflow",
    "renoise",
    "aidi_e",
}

DEFAULT_TEMPLATE = {
    "pie_bench": {
        "data_root": ".data",
        "mapping_file": ".data/mapping_file.json",
        "src_image_folder": ".data/annotation_images",
    },
    "output": {
        "root": "./runs/release_selfix",
        "result_dir": "results",
        "image_dir": "images",
        "feature_dir": "features",
        "skip_existing": True,
    },
    "evaluation": {
        "device": "cuda",
        "metrics": [],
    },
    "selfix": {
        "name": "flux-dev",
        "method": "selfix",
        "num_steps": 15,
        "num_iterations": 10,
        "window_size": 1,
        "momentum": 0.5,
        "alpha1": 0.5,
        "delta": 1.0,
        "initialization": "clone",
        "guidance": 1,
        "inject_steps": 0,
        "start_layer": 0,
        "end_layer": 37,
        "reuse_v": 1,
        "editing_strategy": "replace_v",
        "qkv_ratio": "1,1,1",
        "offload": False,
        "seed": 0,
        "save_trace": False,
        "save_latent_trajectory": False,
        "enable_nsfw_filter": False,
    },
}


def get_default_template() -> dict[str, Any]:
    return deepcopy(DEFAULT_TEMPLATE)


def merge_with_default_config(config: dict[str, Any]) -> dict[str, Any]:
    if "fireflow" in config:
        raise ValueError("Config section 'fireflow' is no longer supported; use 'selfix'.")
    merged = get_default_template()
    for section, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section].update(value)
        else:
            merged[section] = value
    validate_config(merged)
    return merged


CONFIG_EXCLUDED_FIELDS = {
    "source_img_dir",
    "source_prompt",
    "target_prompt",
    "feature_path",
    "output_dir",
    "output_prefix",
    "device",
    "add_sampling_metadata",
    "enable_nsfw_filter",
}

ARG_ALIASES = {
    "enable_renoise_editability": "renoise_enhance_editability",
}


def apply_selfix_arg_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    selfix = config["selfix"]
    config_fields = {
        field.name for field in fields(SamplingConfig)
        if field.name not in CONFIG_EXCLUDED_FIELDS
    }

    for field_name in config_fields:
        if not hasattr(args, field_name):
            continue
        value = getattr(args, field_name)
        if isinstance(value, bool):
            if value:
                selfix[field_name] = value
        elif value is not None:
            selfix[field_name] = value

    for arg_name, field_name in ARG_ALIASES.items():
        if not hasattr(args, arg_name):
            continue
        value = getattr(args, arg_name)
        if isinstance(value, bool):
            if value:
                selfix[field_name] = value
        elif value is not None:
            selfix[field_name] = value

    validate_config(config)
    return config


def load_pie_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(config_path)
    raw_text = config_path.read_text()
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(raw_text)
    else:
        yaml_error = None
        if yaml is not None:
            try:
                data = yaml.safe_load(raw_text)
            except Exception as exc:  # pragma: no cover
                yaml_error = exc
            else:
                return data or {}
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as json_error:
            if yaml_error is not None:
                raise ValueError(
                    f"Failed to parse config {config_path} as YAML or JSON. "
                    f"YAML error: {yaml_error}. JSON error: {json_error}."
                ) from json_error
            raise ValueError(f"Failed to parse config {config_path} as JSON.") from json_error
    return data or {}


def validate_config(config: dict[str, Any]) -> None:
    if "fireflow" in config:
        raise ValueError("Config section 'fireflow' is no longer supported; use 'selfix'.")
    required_sections = ["pie_bench", "output", "evaluation", "selfix"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    pie_bench = config["pie_bench"]
    output = config["output"]
    required_paths = {
        "pie_bench.data_root": pie_bench.get("data_root"),
        "pie_bench.mapping_file": pie_bench.get("mapping_file"),
        "pie_bench.src_image_folder": pie_bench.get("src_image_folder"),
        "output.root": output.get("root"),
    }
    for label, value in required_paths.items():
        if not value:
            raise ValueError(f"Missing required config value: {label}")

    selfix = config["selfix"]
    required_values = [
        "name",
        "method",
        "num_steps",
        "num_iterations",
        "window_size",
        "momentum",
        "alpha1",
        "delta",
        "initialization",
        "guidance",
        "inject_steps",
        "start_layer",
        "end_layer",
        "reuse_v",
        "editing_strategy",
        "qkv_ratio",
    ]
    for key in required_values:
        if key not in selfix:
            raise ValueError(f"Missing required config value: selfix.{key}")

    if selfix["method"] not in ALLOWED_METHODS:
        raise ValueError(f"Unsupported public method {selfix['method']!r}. Expected one of {sorted(ALLOWED_METHODS)}.")
    if selfix["initialization"] not in {"euler", "clone"}:
        raise ValueError("selfix.initialization must be 'euler' or 'clone'.")
    if int(selfix["num_steps"]) < 1:
        raise ValueError("selfix.num_steps must be >= 1.")
    if int(selfix["num_iterations"]) < 1:
        raise ValueError("selfix.num_iterations must be >= 1.")
    if int(selfix["window_size"]) < 1:
        raise ValueError("selfix.window_size must be >= 1.")
    momentum = float(selfix["momentum"])
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"selfix.momentum must satisfy 0 <= momentum < 1, got {momentum}")
    alpha1 = float(selfix["alpha1"])
    delta = float(selfix["delta"])
    if not 0.0 < alpha1 < 1.0:
        raise ValueError(f"selfix.alpha1 must satisfy 0 < alpha1 < 1, got {alpha1}")
    if delta <= 0.0:
        raise ValueError(f"selfix.delta must be > 0, got {delta}")


def resolve_output_root(config: dict, args: argparse.Namespace) -> dict[str, Path]:
    output_config = config["output"]
    root = Path(output_config["root"]).expanduser().resolve()

    result_dir = Path(output_config.get("result_dir", root / "results")).expanduser()
    image_dir = Path(output_config.get("image_dir", root / "images")).expanduser()
    feature_dir = Path(output_config.get("feature_dir", root / "features")).expanduser()

    if not result_dir.is_absolute():
        result_dir = (root / result_dir).resolve()
    else:
        result_dir = result_dir.resolve()
    if not image_dir.is_absolute():
        image_dir = (root / image_dir).resolve()
    else:
        image_dir = image_dir.resolve()
    if not feature_dir.is_absolute():
        feature_dir = (root / feature_dir).resolve()
    else:
        feature_dir = feature_dir.resolve()

    root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "root": root,
        "result_dir": result_dir,
        "image_dir": image_dir,
        "feature_dir": feature_dir,
    }

    if config["selfix"].get("save_trace", False) or getattr(args, "save_trace", False):
        trace_dir = root / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        output_paths["trace_dir"] = trace_dir

    return output_paths


def load_pie_samples(mapping_file: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with Path(mapping_file).open() as handle:
        mapping = json.load(handle)
    return [
        {
            "sample_id": sample_id,
            **item,
        }
        for sample_id, item in sorted(mapping.items(), key=lambda pair: pair[0])
    ]


def slice_samples(samples: list[dict[str, Any]], start_idx: int, end_idx: int | None) -> list[dict[str, Any]]:
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0, got {start_idx}")
    if end_idx is None:
        end_idx = len(samples) - 1
    if end_idx < start_idx:
        raise ValueError(f"end_idx must be >= start_idx, got start_idx={start_idx}, end_idx={end_idx}")
    if start_idx >= len(samples):
        raise ValueError(f"start_idx {start_idx} is out of range for {len(samples)} samples")
    if end_idx >= len(samples):
        raise ValueError(f"end_idx {end_idx} is out of range for {len(samples)} samples")
    return samples[start_idx : end_idx + 1]


def get_sample_paths(sample: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Path]:
    src_image_folder = Path(config["pie_bench"]["src_image_folder"]).expanduser().resolve()
    source_image_path = src_image_folder / sample["image_path"]
    image_path = output_paths["image_dir"] / f"{sample['sample_id']}.jpg"
    result_path = output_paths["result_dir"] / f"{sample['sample_id']}.json"
    feature_path = output_paths["feature_dir"] / sample["sample_id"]
    return {
        "source_image_path": source_image_path,
        "generated_image_path": image_path,
        "result_path": result_path,
        "feature_path": feature_path,
    }


def estimate_nfe(
    method: str,
    num_steps: int,
    inv_stop_ratio: float,
    num_iterations: int,
    initialization: str,
    renoise_enhance_editability: bool = False,
) -> int:
    effective_inv_steps, effective_gen_steps = get_effective_step_counts(num_steps, inv_stop_ratio)
    if method == "reflow":
        base_nfe = effective_inv_steps + effective_gen_steps
    elif method == "rf_solver":
        base_nfe = 2 * effective_inv_steps + 2 * effective_gen_steps
    elif method == "fireflow":
        base_nfe = (effective_inv_steps + 1) + (effective_gen_steps + 1)
    elif method == "aidi_e":
        base_nfe = effective_inv_steps * num_iterations + effective_gen_steps
    elif method in {"selfix", "fpi"}:
        if initialization == "clone":
            base_nfe = effective_inv_steps * num_iterations + effective_gen_steps
        else:
            base_nfe = effective_inv_steps * (1 + num_iterations) + effective_gen_steps
    elif method == "renoise":
        extra_reference = 1 if renoise_enhance_editability else 0
        base_nfe = effective_inv_steps * (extra_reference + num_iterations) + effective_gen_steps
    else:
        raise ValueError(f"Unsupported method for NFE estimate: {method}")
    return base_nfe


def build_config(
    sample: dict,
    config: dict,
    sample_paths: dict[str, Path],
    args: argparse.Namespace,
) -> SamplingConfig:
    selfix_config = config["selfix"]
    return SamplingConfig(
        name=selfix_config["name"],
        source_img_dir=str(sample_paths["source_image_path"]),
        feature_path=str(sample_paths["feature_path"]),
        guidance=selfix_config["guidance"],
        num_steps=selfix_config["num_steps"],
        inv_stop_ratio=selfix_config.get("inv_stop_ratio", 1.0),
        inject_steps=selfix_config["inject_steps"],
        start_layer=selfix_config["start_layer"],
        end_layer=selfix_config["end_layer"],
        output_dir=str(sample_paths["generated_image_path"].parent),
        output_prefix=sample["sample_id"],
        method=selfix_config["method"],
        offload=selfix_config.get("offload", False),
        reuse_v=selfix_config.get("reuse_v", 1),
        editing_strategy=selfix_config.get("editing_strategy", "replace_v"),
        qkv_ratio=selfix_config.get("qkv_ratio", "1.0,1.0,1.0"),
        seed=selfix_config.get("seed", 0),
        num_iterations=selfix_config["num_iterations"],
        initialization=selfix_config["initialization"],
        momentum=selfix_config["momentum"],
        window_size=selfix_config["window_size"],
        alpha1=selfix_config["alpha1"],
        delta=selfix_config["delta"],
        renoise_avg_mode=selfix_config.get("renoise_avg_mode", "uniform_all"),
        renoise_first_step_range=selfix_config.get("renoise_first_step_range", "0,3"),
        renoise_step_range=selfix_config.get("renoise_step_range", "7,9"),
        renoise_enhance_editability=selfix_config.get("renoise_enhance_editability", False),
        save_trace=selfix_config.get("save_trace", False) or getattr(args, "save_trace", False),
        save_latent_trajectory=selfix_config.get("save_latent_trajectory", False),
        device=config["evaluation"].get("device", "cuda"),
        add_sampling_metadata=False,
        enable_nsfw_filter=selfix_config.get("enable_nsfw_filter", False),
    )


def build_edit_config(
    sample: dict,
    config: dict,
    sample_paths: dict[str, Path],
    args: argparse.Namespace,
):
    source_prompt = sample["original_prompt"].replace("[", "").replace("]", "")
    editing_prompt = sample["editing_prompt"].replace("[", "").replace("]", "")

    edit_config = build_config(sample, config, sample_paths, args)
    edit_config.source_prompt = source_prompt
    edit_config.target_prompt = editing_prompt
    return edit_config


def build_reconstruction_config(
    sample: dict,
    config: dict,
    sample_paths: dict[str, Path],
    args: argparse.Namespace,
):
    source_prompt = sample["original_prompt"].replace("[", "").replace("]", "")

    recon_config = build_config(sample, config, sample_paths, args)
    recon_config.source_prompt = source_prompt
    recon_config.target_prompt = source_prompt
    return recon_config


def resolve_config(config: dict, args: argparse.Namespace) -> dict:
    selfix = config["selfix"]
    num_steps = int(selfix["num_steps"])
    inv_stop_ratio = float(selfix.get("inv_stop_ratio", 1.0))
    effective_inv_steps, effective_gen_steps = get_effective_step_counts(num_steps, inv_stop_ratio)
    return {
        "config_path": str(Path(args.config).resolve()),
        "selfix": {
            "method": selfix["method"],
            "num_steps": num_steps,
            "num_iterations": int(selfix["num_iterations"]),
            "window_size": int(selfix["window_size"]),
            "momentum": float(selfix["momentum"]),
            "alpha1": float(selfix["alpha1"]),
            "delta": float(selfix["delta"]),
            "initialization": selfix["initialization"],
            "guidance": float(selfix["guidance"]),
            "inject_steps": int(selfix["inject_steps"]),
            "start_layer": int(selfix["start_layer"]),
            "end_layer": int(selfix["end_layer"]),
            "reuse_v": selfix.get("reuse_v", 1),
            "editing_strategy": selfix.get("editing_strategy", "replace_v"),
            "qkv_ratio": selfix.get("qkv_ratio", "1.0,1.0,1.0"),
            "seed": selfix.get("seed", 0),
            "save_trace": selfix.get("save_trace", False) or getattr(args, "save_trace", False),
            "save_latent_trajectory": selfix.get("save_latent_trajectory", False),
        },
        "inv_stop_ratio": inv_stop_ratio,
        "effective_inv_steps": effective_inv_steps,
        "effective_gen_steps": effective_gen_steps,
        "partial_inversion_enabled": effective_inv_steps < num_steps,
        "estimated_nfe": estimate_nfe(
            selfix["method"],
            num_steps,
            inv_stop_ratio,
            int(selfix["num_iterations"]),
            selfix["initialization"],
            selfix.get("renoise_enhance_editability", False),
        ),
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
    }


def build_trace_payload(result, sample_id: str, args: argparse.Namespace, config: dict) -> dict | None:
    if result.trace is None:
        return None
    selfix = config["selfix"]
    return {
        "sample_id": sample_id,
        "method": selfix["method"],
        "initialization": selfix["initialization"],
        "num_iterations": selfix["num_iterations"],
        "momentum": selfix["momentum"],
        "window_size": selfix["window_size"],
        "alpha1": selfix["alpha1"],
        "delta": selfix["delta"],
        "num_steps": selfix["num_steps"],
        **result.trace,
    }


def atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def parse_number(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def compute_traj_straightness(traj_x, traj_ts, online: bool = False) -> float:
    traj_x = torch.as_tensor(traj_x, dtype=torch.float32)
    traj_ts = torch.as_tensor(traj_ts, dtype=torch.float32)
    assert traj_x.shape[1] == traj_ts.shape[0]
    assert traj_x.ndim == 3
    if traj_x.shape[1] < 2:
        raise ValueError("Trajectory must contain at least two timesteps to compute straightness.")

    dxs = traj_x[:, 1:] - traj_x[:, :-1]
    dts = traj_ts[1:] - traj_ts[:-1]
    if torch.any(dts == 0):
        raise ValueError("Trajectory timesteps must be strictly monotonic.")
    vels = dxs / dts[None, :, None]

    if online:
        cum_displacement = traj_x[:, 1:] - traj_x[:, 0:1]
        cum_dt = traj_ts[None, 1:, None] - traj_ts[None, 0:1, None]
        if torch.any(cum_dt == 0):
            raise ValueError("Trajectory timesteps must span a non-zero duration.")
        cum_vel = cum_displacement / cum_dt
        diff = ((vels - cum_vel) ** 2).mean(dim=-1)
    else:
        total_dt = traj_ts[-1] - traj_ts[0]
        if total_dt == 0:
            raise ValueError("Trajectory timesteps must span a non-zero duration.")
        global_vel = (traj_x[:, -1] - traj_x[:, 0]) / total_dt
        diff = ((vels - global_vel[:, None, :]) ** 2).mean(dim=-1)

    diff_dt = diff * dts[None, :].abs()
    return float(diff_dt.sum(dim=1).mean().item())


def _prepare_straightness_trajectory(direction_trajectory: dict[str, Any], direction: str) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = direction_trajectory.get("timesteps")
    latents = direction_trajectory.get("latents")
    if timesteps is None or latents is None:
        raise ValueError(f"Missing timesteps or latents for {direction} trajectory.")
    if len(timesteps) != len(latents):
        raise ValueError(
            f"Mismatched timestep/latent lengths for {direction} trajectory: "
            f"{len(timesteps)} vs {len(latents)}."
        )
    if len(timesteps) < 2:
        raise ValueError(f"{direction} trajectory must contain at least two timesteps.")

    flat_latents = []
    batch_size = None
    for latent in latents:
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        if latent_tensor.ndim < 1:
            raise ValueError(f"{direction} trajectory latent is missing a batch dimension.")
        if batch_size is None:
            batch_size = int(latent_tensor.shape[0])
        elif int(latent_tensor.shape[0]) != batch_size:
            raise ValueError(f"Inconsistent batch size in {direction} trajectory.")
        flat_latents.append(latent_tensor.reshape(batch_size, -1))

    traj_x = torch.stack(flat_latents, dim=1)
    traj_ts = torch.as_tensor(timesteps, dtype=torch.float32)
    return traj_x, traj_ts


def compute_reconstruction_trajectory_metrics(latent_trajectory: dict[str, Any]) -> dict[str, float]:
    if latent_trajectory is None:
        raise ValueError("Missing latent trajectory for reconstruction straightness metrics.")

    inverse_trajectory = latent_trajectory.get("inverse")
    generation_trajectory = latent_trajectory.get("generation")
    if inverse_trajectory is None or generation_trajectory is None:
        raise ValueError("Reconstruction straightness metrics require both inverse and generation trajectories.")

    inv_x, inv_ts = _prepare_straightness_trajectory(inverse_trajectory, "inverse")
    gen_x, gen_ts = _prepare_straightness_trajectory(generation_trajectory, "generation")
    return {
        "inv_rf_straightness": compute_traj_straightness(inv_x, inv_ts, online=False),
        "inv_rf_straightness_online": compute_traj_straightness(inv_x, inv_ts, online=True),
        "gen_rf_straightness": compute_traj_straightness(gen_x, gen_ts, online=False),
        "gen_rf_straightness_online": compute_traj_straightness(gen_x, gen_ts, online=True),
    }


def summarize_result_files(paths: list[Path]) -> tuple[dict[str, dict[str, float]], int]:
    totals: dict[str, dict[str, float]] = {}
    failed = 0
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "success":
            failed += 1
            continue
        for metric_name, metric_value in payload.get("metrics", {}).items():
            number = parse_number(metric_value)
            if number is None:
                continue
            metric_entry = totals.setdefault(metric_name, {"sum": 0.0, "count": 0.0})
            metric_entry["sum"] += number
            metric_entry["count"] += 1
    return totals, failed


def build_summary(totals: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    summary = []
    for metric_name in sorted(totals):
        count = int(totals[metric_name]["count"])
        mean = totals[metric_name]["sum"] / count if count else None
        summary.append({"metric": metric_name, "mean": "%.4f" % mean, "count": count})
    return summary


def config_snapshot(config: SamplingConfig) -> dict[str, Any]:
    return asdict(config)
