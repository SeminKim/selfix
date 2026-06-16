import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from tqdm import tqdm

import transformers.modeling_utils
import transformers.utils
import transformers.utils.import_utils
from edit import crop_image_to_multiple_of_16, load_components, run_image_edit
from pie_bench_metrics import DEFAULT_METRICS, MetricsCalculator, build_mask, compute_metrics, crop_for_evaluation
from pie_bench_utils import (
    apply_selfix_arg_overrides,
    atomic_write_json,
    build_edit_config,
    build_reconstruction_config,
    build_summary,
    build_trace_payload,
    compute_reconstruction_trajectory_metrics,
    get_default_template,
    get_sample_paths,
    load_pie_config,
    load_pie_samples,
    merge_with_default_config,
    resolve_config,
    resolve_output_root,
    slice_samples,
    summarize_result_files,
    validate_config,
)


def bypass_safety_check(*args, **kwargs):
    return None


transformers.utils.import_utils.check_torch_load_is_safe = bypass_safety_check
transformers.modeling_utils.check_torch_load_is_safe = bypass_safety_check
transformers.utils.check_torch_load_is_safe = bypass_safety_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public SelFix PIE-Bench experiments.")
    parser.add_argument("--config", required=True, type=str, help="Path to a .yaml or .json config file")
    parser.add_argument("--task", required=True, choices=("reconstruction", "editing"))
    parser.add_argument("--start_idx", type=int, default=0, help="Inclusive dataset start index")
    parser.add_argument("--end_idx", type=int, default=None, help="Inclusive dataset end index")
    parser.add_argument("--shard_index", type=int, default=0, help="Zero-based shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of modulo shards")
    parser.add_argument("--output_root", type=str, help="Optional override for output.root")
    parser.add_argument("--device", type=str, help="Optional override for evaluation.device")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples whose JSON result already exists")
    parser.add_argument("--save_trace", action="store_true", help="Save fixed-point trace JSON files")
    parser.add_argument(
        "--method",
        choices=("selfix", "fpi", "reflow", "rf_solver", "fireflow", "renoise", "aidi_e"),
    )
    parser.add_argument("--name", type=str)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--initialization", choices=("euler", "clone"), default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--alpha1", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--inject_steps", type=int, default=None)
    parser.add_argument("--start_layer", type=int, default=None)
    parser.add_argument("--end_layer", type=int, default=None)
    parser.add_argument("--reuse_v", type=int, default=None)
    parser.add_argument("--editing_strategy", type=str, default=None)
    parser.add_argument("--qkv_ratio", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--offload", action="store_true", default=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.output_root:
        config["output"]["root"] = args.output_root
        config["output"].pop("result_dir", None)
        config["output"].pop("image_dir", None)
        config["output"].pop("feature_dir", None)
    if args.device:
        config["evaluation"]["device"] = args.device
    if args.skip_existing:
        config["output"]["skip_existing"] = True
    if args.save_trace:
        config["selfix"]["save_trace"] = True
    return apply_selfix_arg_overrides(config, args)


def shard_samples(samples: list[dict], start_idx: int, shard_index: int, num_shards: int) -> list[tuple[int, dict]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}")
    return [
        (dataset_index, sample)
        for offset, sample in enumerate(samples)
        for dataset_index in [start_idx + offset]
        if dataset_index % num_shards == shard_index
    ]


def compute_reconstruction_metrics(metrics_calculator: MetricsCalculator, source_path: Path, generated_path: Path) -> dict:
    with Image.open(source_path) as src_handle:
        src_image = Image.fromarray(crop_image_to_multiple_of_16(src_handle))
    with Image.open(generated_path) as pred_handle:
        pred_image = pred_handle.convert("RGB")
    if src_image.size != pred_image.size:
        src_image = src_image.crop((0, 0, pred_image.size[0], pred_image.size[1]))
    return {
        "lpips": metrics_calculator.calculate_lpips(src_image, pred_image),
        "ssim": metrics_calculator.calculate_ssim(src_image, pred_image),
        "psnr": metrics_calculator.calculate_psnr(src_image, pred_image),
    }


def write_summary(output_paths: dict[str, Path]) -> None:
    result_paths = sorted(output_paths["result_dir"].glob("*.json"))
    totals, failed = summarize_result_files(result_paths)
    payload = {
        "schema_version": "pie_bench_summary_v1",
        "num_results": len(result_paths),
        "num_failed": failed,
        "metrics": build_summary(totals),
    }
    atomic_write_json(output_paths["root"] / "summary.json", payload)


def main() -> None:
    args = parse_args()
    config = load_pie_config(args.config)
    config = merge_with_default_config(config) if config else get_default_template()
    config = apply_overrides(config, args)
    validate_config(config)

    if args.task == "reconstruction":
        config["selfix"]["save_latent_trajectory"] = True

    output_paths = resolve_output_root(config, args)
    trajectory_dir = output_paths["root"] / "trajectories"
    if config["selfix"].get("save_latent_trajectory", False):
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = output_paths["root"] / "latents"
    if args.task == "reconstruction":
        latent_dir.mkdir(parents=True, exist_ok=True)

    samples = load_pie_samples(config["pie_bench"]["mapping_file"])
    selected_samples = slice_samples(samples, args.start_idx, args.end_idx)
    indexed_samples = shard_samples(selected_samples, args.start_idx, args.shard_index, args.num_shards)
    if not indexed_samples:
        raise ValueError("No PIE-Bench samples selected for this shard/range.")

    print(
        f"Loaded {len(samples)} PIE-Bench samples. Running {args.task} on "
        f"{len(indexed_samples)} selected samples."
    )

    config_factory = build_reconstruction_config if args.task == "reconstruction" else build_edit_config
    first_sample = indexed_samples[0][1]
    component_config = config_factory(first_sample, config, get_sample_paths(first_sample, config, output_paths), args)
    components = load_components(component_config)
    metrics_calculator = MetricsCalculator(config["evaluation"].get("device", "cuda"))
    metrics = config["evaluation"].get("metrics", DEFAULT_METRICS)

    resolved = resolve_config(config, args)
    resolved["task"] = args.task
    resolved["shard_index"] = args.shard_index
    resolved["num_shards"] = args.num_shards
    (output_paths["root"] / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))

    skip_existing = config["output"].get("skip_existing", False)
    for dataset_index, sample in tqdm(indexed_samples, desc=f"PIE {args.task}", unit="sample"):
        sample_paths = get_sample_paths(sample, config, output_paths)
        result_path = sample_paths["result_path"]
        if skip_existing and result_path.exists():
            tqdm.write(f"Skipping sample {sample['sample_id']} because {result_path} already exists.")
            continue

        edit_config = config_factory(sample, config, sample_paths, args)
        trajectory_path = trajectory_dir / f"{sample['sample_id']}.pt"
        trace_path = output_paths.get("trace_dir", output_paths["root"] / "trace") / f"{sample['sample_id']}.json"
        latent_path = latent_dir / f"{sample['sample_id']}.pt"
        start_time = perf_counter()
        trace_payload = None
        try:
            edit_result = run_image_edit(edit_config, components=components)
            if edit_result.latent_trajectory is not None:
                torch.save(edit_result.latent_trajectory, trajectory_path)
            if args.task == "reconstruction" and edit_result.inverted_latent is not None:
                torch.save(edit_result.inverted_latent, latent_path)
            trace_payload = build_trace_payload(edit_result, sample["sample_id"], args, config)

            payload = {
                "schema_version": f"pie_bench_{args.task}_v1",
                "task": args.task,
                "sample_id": sample["sample_id"],
                "index": dataset_index,
                "source_image_path": str(sample_paths["source_image_path"]),
                "source_prompt": edit_config.source_prompt,
                "target_prompt": edit_config.target_prompt,
                "generated_image_path": str(sample_paths["generated_image_path"]),
                "latent_trajectory_path": str(trajectory_path) if trajectory_path.exists() else None,
                "metrics": {},
                "edit_seconds": edit_result.elapsed_seconds,
                "status": "success",
                "error": None,
            }
            if args.task == "reconstruction":
                payload["inverted_latent_path"] = str(latent_path) if latent_path.exists() else None

            if edit_result.image is None:
                payload["status"] = "failed"
                payload["error"] = "NSFW filter blocked the generated image."
            else:
                edit_result.image.save(sample_paths["generated_image_path"], quality=95, subsampling=0)
                if args.task == "reconstruction":
                    metric_results = compute_reconstruction_metrics(
                        metrics_calculator,
                        sample_paths["source_image_path"],
                        sample_paths["generated_image_path"],
                    )
                    metric_results.update(compute_reconstruction_trajectory_metrics(edit_result.latent_trajectory))
                else:
                    with Image.open(sample_paths["source_image_path"]) as src_handle:
                        src_image = src_handle.convert("RGB")
                    with Image.open(sample_paths["generated_image_path"]) as tgt_handle:
                        tgt_image = crop_for_evaluation(tgt_handle.convert("RGB"))
                    metric_results = compute_metrics(
                        metrics_calculator,
                        metrics,
                        src_image,
                        tgt_image,
                        build_mask(sample["mask"]),
                        edit_config.source_prompt,
                        edit_config.target_prompt,
                    )
                payload["metrics"] = metric_results
        except Exception as exc:
            payload = {
                "schema_version": f"pie_bench_{args.task}_v1",
                "task": args.task,
                "sample_id": sample["sample_id"],
                "index": dataset_index,
                "source_image_path": str(sample_paths["source_image_path"]),
                "source_prompt": getattr(edit_config, "source_prompt", None),
                "target_prompt": getattr(edit_config, "target_prompt", None),
                "generated_image_path": str(sample_paths["generated_image_path"]),
                "latent_trajectory_path": str(trajectory_path) if trajectory_path.exists() else None,
                "metrics": {},
                "edit_seconds": perf_counter() - start_time,
                "status": "failed",
                "error": str(exc),
            }
            if args.task == "reconstruction":
                payload["inverted_latent_path"] = str(latent_path) if latent_path.exists() else None

        if trace_payload is not None:
            atomic_write_json(trace_path, trace_payload)
            payload["trace_path"] = str(trace_path)

        atomic_write_json(result_path, payload)
        tqdm.write(f"Finished sample {sample['sample_id']} with status={payload['status']}")

    write_summary(output_paths)


if __name__ == "__main__":
    main()
