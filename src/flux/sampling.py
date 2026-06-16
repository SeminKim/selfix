import math
from random import randrange
from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def ensure_trajectory_trace(info: dict) -> dict | None:
    if not bool(info.get("save_trace", False)):
        return None
    trace = info.setdefault("trace", {})
    trace.setdefault("trajectory_straightness", 0.0)
    trace.setdefault("generation_trajectory_straightness", 0.0)
    return trace


def update_trajectory_straightness(
    trace: dict | None,
    prev_velocity: Tensor | None,
    current_img: Tensor,
    next_img: Tensor,
    dt: float,
    inverse: bool,
) -> Tensor:
    velocity = (next_img - current_img) / dt
    if trace is not None and prev_velocity is not None:
        metric_name = "trajectory_straightness" if inverse else "generation_trajectory_straightness"
        trace[metric_name] += (velocity - prev_velocity).abs().mean().item()
    return velocity


def selfix_alpha(k: int, alpha1: float, delta: float) -> float:
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not 0.0 < alpha1 < 1.0:
        raise ValueError(f"alpha1 must satisfy 0 < alpha1 < 1, got {alpha1}")
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    return alpha1 * delta / (k + delta)


def build_selfix_anchor(
    dt: float,
    final_img_history: list[Tensor],
    final_dt_history: list[float],
    window_size: int,
) -> Tensor | None:
    if len(final_img_history) < 2:
        return None
    available_window_size = min(window_size + 1, len(final_img_history))
    recent_imgs = final_img_history[-available_window_size:]
    recent_dts = final_dt_history[-(available_window_size - 1):]
    velocities = [
        (curr_img - prev_img) / step_dt
        for prev_img, curr_img, step_dt in zip(recent_imgs[:-1], recent_imgs[1:], recent_dts)
        if step_dt not in (None, 0)
    ]
    if velocities:
        avg_velocity = torch.stack(velocities, dim=0).mean(dim=0)
        return recent_imgs[-1] + dt * avg_velocity
    return None


def record_latent_trajectory(info: dict, inverse: bool, timestep: float, latent: Tensor) -> None:
    if not bool(info.get("save_latent_trajectory", False)):
        return
    direction = "inverse" if inverse else "generation"
    trajectory = info.setdefault("latent_trajectory", {})
    direction_trajectory = trajectory.setdefault(direction, {"timesteps": [], "latents": []})
    direction_trajectory["timesteps"].append(float(timestep))
    direction_trajectory["latents"].append(latent.detach().cpu())


def _parse_renoise_range(value) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected renoise range formatted as start,end, got {value!r}")
        start, end = int(parts[0]), int(parts[1])
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        start, end = int(value[0]), int(value[1])
    else:
        raise ValueError(f"Unsupported renoise range value: {value!r}")
    if start < 0 or end < start:
        raise ValueError(f"Invalid renoise range: ({start}, {end})")
    return start, end


def _sequence_grid_shape(img_ids: Tensor) -> tuple[int, int]:
    img_ids_0 = img_ids[0]
    height = int(img_ids_0[:, 1].max().item()) + 1
    width = int(img_ids_0[:, 2].max().item()) + 1
    return height, width


def _sequence_to_spatial(x: Tensor, img_ids: Tensor) -> Tensor:
    height, width = _sequence_grid_shape(img_ids)
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


def _auto_corr_loss_spatial(x: Tensor, random_shift: bool = True) -> Tensor:
    b, c, _, _ = x.shape
    if b != 1:
        raise ValueError(f"auto-correlation regularization expects batch size 1, got {b}")
    x = x.squeeze(0)
    reg_loss = x.new_tensor(0.0)
    for ch_idx in range(c):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift and noise.shape[2] > 1:
                roll_amount = randrange(max(noise.shape[2] // 2, 1))
            else:
                roll_amount = 1
            reg_loss = reg_loss + (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean().square()
            reg_loss = reg_loss + (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean().square()
            if noise.shape[2] <= 8 or noise.shape[3] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss


def _latents_kl_divergence(x0: Tensor, x1: Tensor) -> Tensor:
    eps = 1e-7
    mu0 = x0.mean(dim=(2, 3))
    mu1 = x1.mean(dim=(2, 3))
    var0 = x0.var(dim=(2, 3), unbiased=False) + eps
    var1 = x1.var(dim=(2, 3), unbiased=False) + eps
    return 0.5 * (torch.log(var1 / var0) + (var0 + (mu0 - mu1).square()) / var1 - 1.0)


def _patchify_latents_kl_divergence(x0: Tensor, x1: Tensor, patch_size: int = 4) -> Tensor:
    _, _, h, w = x0.shape
    patch = max(1, min(patch_size, h, w))

    def patchify_tensor(input_tensor: Tensor) -> Tensor:
        patches = input_tensor.unfold(2, patch, patch).unfold(3, patch, patch)
        return patches.contiguous().view(-1, input_tensor.shape[1], patch, patch)

    return _latents_kl_divergence(patchify_tensor(x0), patchify_tensor(x1)).sum()


def _renoise_regularize_prediction(
    pred: Tensor,
    pred_target: Tensor | None,
    img_ids: Tensor,
    lambda_ac: float,
    lambda_kl: float,
) -> Tensor:
    if lambda_ac <= 0 and lambda_kl <= 0:
        return pred

    pred_reg = pred.detach().clone()
    img_ids = img_ids.detach().clone()
    pred_target_spatial = _sequence_to_spatial(pred_target.detach().clone(), img_ids) if pred_target is not None else None

    total_grad = None

    if lambda_kl > 0 and pred_target_spatial is not None:
        var = pred_reg.detach().clone().requires_grad_(True)
        loss_kl = _patchify_latents_kl_divergence(_sequence_to_spatial(var, img_ids), pred_target_spatial)
        loss_kl.backward()
        grad_kl = var.grad.detach()
        total_grad = lambda_kl * grad_kl

    if lambda_ac > 0:
        var = pred_reg.detach().clone().requires_grad_(True)
        loss_ac = _auto_corr_loss_spatial(_sequence_to_spatial(var, img_ids))
        loss_ac.backward()
        grad_ac = var.grad.detach()
        total_grad = lambda_ac * grad_ac if total_grad is None else total_grad + lambda_ac * grad_ac

    if total_grad is None:
        return pred_reg

    return (pred_reg - total_grad).detach()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    trace = ensure_trajectory_trace(info)
    prev_velocity = None

    step_list = []
    record_latent_trajectory(info, inverse, timesteps[0], img)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        
        next_img = img + (t_prev - t_curr) * pred
        prev_velocity = update_trajectory_straightness(
            trace,
            prev_velocity,
            img,
            next_img,
            t_prev - t_curr,
            inverse,
        )
        img = next_img
        record_latent_trajectory(info, inverse, t_prev, img)

    return img, info


def denoise_selfix(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0,
):
    inject_list = [True] * info["inject_step"] + [False] * (len(timesteps[:-1]) - info["inject_step"])
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    method = str(info.get("method", "selfix"))
    num_iterations = max(int(info.get("num_iterations", 3)), 1)
    initialization = str(info.get("initialization", "clone"))
    momentum = float(info.get("momentum", 0.0))
    window_size = max(int(info.get("window_size", 1)), 1)
    alpha1 = float(info.get("alpha1", 0.5))
    delta = float(info.get("delta", 1.0))
    use_selfix_anchor = method == "selfix"
    if initialization not in {"euler", "clone"}:
        raise ValueError(f"Unsupported fixed-point initialization: {initialization}")
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {momentum}")
    if method not in {"selfix", "fpi"}:
        raise ValueError(f"denoise_selfix only supports 'selfix' and 'fpi', got {method!r}")
    if method == "fpi":
        momentum = 0.0

    trace = ensure_trajectory_trace(info)
    if trace is not None:
        trace.setdefault("method", method)
        trace.setdefault("initialization", initialization)
        trace.setdefault("num_iterations", num_iterations)
        trace.setdefault("momentum", momentum)
        trace.setdefault("window_size", window_size)
        trace.setdefault("alpha1", alpha1)
        trace.setdefault("delta", delta)
        trace.setdefault("inverse_steps", [])

    final_img_history: list[Tensor] = []
    final_dt_history: list[float] = []
    prev_velocity = None
    record_latent_trajectory(info, inverse, timesteps[0], img)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        dt = t_prev - t_curr
        t_vec_curr = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info["t"] = t_prev if inverse else t_curr
        info["inverse"] = inverse
        info["second_order"] = False
        info["inject"] = inject_list[i]

        if not inverse:
            assert dt < 0, (t_curr, t_prev, dt)
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_curr,
                guidance=guidance_vec,
                info=info,
            )
            next_img = img + dt * pred
            velocity = update_trajectory_straightness(trace, prev_velocity, img, next_img, dt, inverse)
            img = next_img
            record_latent_trajectory(info, inverse, t_prev, img)
            prev_velocity = velocity
            continue

        assert dt > 0, (t_curr, t_prev, dt)

        if initialization == "euler":
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_curr,
                guidance=guidance_vec,
                info=info,
            )
            guess = img + dt * pred
        else:
            guess = img.clone()

        t_vec_next = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
        anchor = build_selfix_anchor(
            dt,
            final_img_history,
            final_dt_history,
            window_size,
        ) if use_selfix_anchor else None
        step_trace = None
        if trace is not None:
            step_trace = {
                "step_index": i,
                "t_curr": float(t_curr),
                "t_prev": float(t_prev),
                "used_anchor": anchor is not None,
                "deltas": [],
                "residuals": [],
                "alphas": [],
                "final_delta": None,
                "final_residual": None,
                "iterations_used": 0,
            }

        base_momentum_state = guess.clone()
        anchor_momentum_state = None
        for k in range(num_iterations):
            pred_next, info = model(
                img=guess,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_next,
                guidance=guidance_vec,
                info=info,
            )
            projected = img + dt * pred_next
            if anchor is None:
                alpha = 0.0
                anchor_update = torch.zeros_like(projected)
            else:
                alpha = selfix_alpha(k, alpha1, delta)
                anchor_update = anchor

            if anchor_momentum_state is None:
                anchor_momentum_state = anchor_update
            base_momentum_state = momentum * base_momentum_state + (1 - momentum) * projected
            anchor_momentum_state = momentum * anchor_momentum_state + (1 - momentum) * anchor_update
            updated_guess = (1 - alpha) * base_momentum_state + alpha * anchor_momentum_state
            step_delta = (updated_guess - guess).abs().mean().item()
            residual = (projected - img).abs().mean().item()
            if step_trace is not None:
                step_trace["deltas"].append(step_delta)
                step_trace["residuals"].append(residual)
                step_trace["alphas"].append(alpha)
            guess = updated_guess

        if step_trace is not None:
            step_trace["iterations_used"] = len(step_trace["deltas"])
            if step_trace["deltas"]:
                step_trace["final_delta"] = step_trace["deltas"][-1]
            if step_trace["residuals"]:
                step_trace["final_residual"] = step_trace["residuals"][-1]
            trace["inverse_steps"].append(step_trace)

        velocity = update_trajectory_straightness(trace, prev_velocity, img, guess, dt, inverse)
        img = guess
        record_latent_trajectory(info, inverse, t_prev, img)
        final_img_history.append(img)
        final_dt_history.append(dt)
        prev_velocity = velocity

    return img, info


def denoise_aidi_e(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0,
):
    inject_list = [True] * info["inject_step"] + [False] * (len(timesteps[:-1]) - info["inject_step"])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    aidi_iters = max(int(info.get("num_iterations", 3)), 1)

    trace = ensure_trajectory_trace(info)
    if trace is not None:
        trace.setdefault("method", "aidi_e")
        trace.setdefault("num_iterations", aidi_iters)
        trace.setdefault("inverse_steps", [])

    prev_velocity = None
    record_latent_trajectory(info, inverse, timesteps[0], img)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        dt = t_prev - t_curr
        t_vec_curr = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info["t"] = t_prev if inverse else t_curr
        info["inverse"] = inverse
        info["second_order"] = False
        info["inject"] = inject_list[i]

        if not inverse:
            assert dt < 0, (t_curr, t_prev, dt)
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_curr,
                guidance=guidance_vec,
                info=info,
            )
            next_img = img + dt * pred
            prev_velocity = update_trajectory_straightness(trace, prev_velocity, img, next_img, dt, inverse)
            img = next_img
            record_latent_trajectory(info, inverse, t_prev, img)
            continue

        assert dt > 0, (t_curr, t_prev, dt)

        guess = img.clone()
        t_vec_next = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
        projected_history: list[Tensor] = []
        step_trace = None
        if trace is not None:
            step_trace = {
                "step_index": i,
                "t_curr": float(t_curr),
                "t_prev": float(t_prev),
                "deltas": [],
                "residuals": [],
                "final_delta": None,
                "final_residual": None,
                "iterations_used": 0,
                "converged": False,
            }

        for _ in range(aidi_iters):
            pred_next, info = model(
                img=guess,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_next,
                guidance=guidance_vec,
                info=info,
            )
            projected = img + dt * pred_next
            projected_history.append(projected)
            if len(projected_history) >= 2:
                updated_guess = 0.5 * (projected_history[-2] + projected_history[-1])
            else:
                updated_guess = projected

            delta = (updated_guess - guess).abs().mean().item()
            residual = (projected - img).abs().mean().item()
            if step_trace is not None:
                step_trace["deltas"].append(delta)
                step_trace["residuals"].append(residual)

            guess = updated_guess
        if step_trace is not None:
            step_trace["iterations_used"] = len(step_trace["deltas"])
            if step_trace["deltas"]:
                step_trace["final_delta"] = step_trace["deltas"][-1]
            if step_trace["residuals"]:
                step_trace["final_residual"] = step_trace["residuals"][-1]
            trace["inverse_steps"].append(step_trace)

        prev_velocity = update_trajectory_straightness(trace, prev_velocity, img, guess, dt, inverse)
        img = guess
        record_latent_trajectory(info, inverse, t_prev, img)

    return img, info


def denoise_renoise(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0,
):
    inject_list = [True] * info["inject_step"] + [False] * (len(timesteps[:-1]) - info["inject_step"])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    renoise_iters = max(int(info.get("num_iterations", 10)), 1)
    renoise_avg_mode = str(info.get("renoise_avg_mode", "uniform_all"))
    renoise_first_step_range = _parse_renoise_range(info.get("renoise_first_step_range", "0,3"))
    renoise_step_range = _parse_renoise_range(info.get("renoise_step_range", "7,9"))
    renoise_enhance_editability = bool(info.get("renoise_enhance_editability", False))
    renoise_paper_t_threshold = 0.25
    renoise_lambda_ac = 10.0
    renoise_lambda_kl = 0.055
    trace = ensure_trajectory_trace(info)
    if trace is not None:
        trace.setdefault("method", "renoise")
        trace.setdefault("renoise_avg_mode", renoise_avg_mode)
        trace.setdefault("renoise_first_step_range", list(renoise_first_step_range))
        trace.setdefault("renoise_step_range", list(renoise_step_range))
        trace.setdefault("renoise_enhance_editability", renoise_enhance_editability)
        trace.setdefault("renoise_paper_t_threshold", renoise_paper_t_threshold)
        trace.setdefault("renoise_lambda_ac", renoise_lambda_ac)
        trace.setdefault("renoise_lambda_kl", renoise_lambda_kl)
        trace.setdefault("inverse_steps", [])

    if renoise_avg_mode not in {"uniform_all", "paper"}:
        raise ValueError(f"Unsupported ReNoise averaging mode: {renoise_avg_mode}")

    prev_velocity = None
    record_latent_trajectory(info, inverse, timesteps[0], img)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        dt = t_prev - t_curr
        t_vec_curr = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info["t"] = t_prev if inverse else t_curr
        info["inverse"] = inverse
        info["second_order"] = False
        info["inject"] = inject_list[i]

        if not inverse:
            assert dt < 0, (t_curr, t_prev, dt)
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_curr,
                guidance=guidance_vec,
                info=info,
            )
            next_img = img + dt * pred
            prev_velocity = update_trajectory_straightness(trace, prev_velocity, img, next_img, dt, inverse)
            img = next_img
            record_latent_trajectory(info, inverse, t_prev, img)
            continue

        assert dt > 0, (t_curr, t_prev, dt)
        t_vec_next = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
        guess = img.clone()
        proposals = []
        use_first_range = float(t_prev) < renoise_paper_t_threshold
        active_range = renoise_first_step_range if use_first_range else renoise_step_range
        regularization_start = 1 if renoise_avg_mode == "uniform_all" else active_range[0]
        pred_next_optimal = None
        if renoise_enhance_editability and (renoise_lambda_ac > 0 or renoise_lambda_kl > 0):
            with torch.inference_mode(False), torch.enable_grad():
                z1_ref = torch.randn_like(img)
                t_curr_value = float(t_curr)
                t_prev_value = float(t_prev)
                denom = max(1.0 - t_curr_value, 1e-6)
                z0_hat = (img - t_curr_value * z1_ref) / denom
                z_t_ref = (t_prev_value * z1_ref + (1.0 - t_prev_value) * z0_hat).detach().clone()
                grad_img_ids = img_ids.detach().clone()
                grad_txt = txt.detach().clone()
                grad_txt_ids = txt_ids.detach().clone()
                grad_vec = vec.detach().clone()
                grad_guidance_vec = guidance_vec.detach().clone()
                pred_next_optimal, _ = model(
                    img=z_t_ref,
                    img_ids=grad_img_ids,
                    txt=grad_txt,
                    txt_ids=grad_txt_ids,
                    y=grad_vec,
                    timesteps=t_vec_next,
                    guidance=grad_guidance_vec,
                    info=info,
                )
                pred_next_optimal = pred_next_optimal.detach()
        step_trace = None
        if trace is not None:
            step_trace = {
                "step_index": i,
                "t_curr": float(t_curr),
                "t_prev": float(t_prev),
                "avg_mode": renoise_avg_mode,
                "regularization_start": regularization_start,
                "deltas": [],
                "residuals": [],
                "iterations_used": 0,
                "final_delta": None,
                "final_residual": None,
            }

        for iter_idx in range(renoise_iters):
            pred_next, info = model(
                img=guess,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_next,
                guidance=guidance_vec,
                info=info,
            )
            proposal = img + dt * pred_next
            if pred_next_optimal is not None and iter_idx >= regularization_start:
                with torch.inference_mode(False), torch.enable_grad():
                    regularized_pred = _renoise_regularize_prediction(
                        pred_next.detach().clone(),
                        pred_next_optimal.detach().clone(),
                        img_ids.detach().clone(),
                        renoise_lambda_ac,
                        renoise_lambda_kl,
                    )
                proposal = img + dt * regularized_pred
            proposals.append(proposal)
            updated_guess = proposal
            delta = (updated_guess - guess).detach().float().abs().mean().item()
            residual = (proposal - img).detach().float().abs().mean().item()
            if step_trace is not None:
                step_trace["deltas"].append(delta)
                step_trace["residuals"].append(residual)
            guess = updated_guess

        if renoise_avg_mode == "uniform_all":
            selected_proposals = proposals
        else:
            start, end = active_range
            clamped_start = min(start, len(proposals) - 1)
            clamped_end = min(end, len(proposals) - 1)
            if clamped_end < clamped_start:
                clamped_start = clamped_end
            selected_proposals = proposals[clamped_start : clamped_end + 1]
            if step_trace is not None:
                step_trace["used_first_range"] = use_first_range
                step_trace["avg_range"] = [clamped_start, clamped_end]

        guess = torch.stack(selected_proposals, dim=0).mean(dim=0)

        if step_trace is not None:
            step_trace["iterations_used"] = len(step_trace["deltas"])
            if step_trace["deltas"]:
                step_trace["final_delta"] = step_trace["deltas"][-1]
            if step_trace["residuals"]:
                step_trace["final_residual"] = step_trace["residuals"][-1]
            trace["inverse_steps"].append(step_trace)

        prev_velocity = update_trajectory_straightness(trace, prev_velocity, img, guess, dt, inverse)
        img = guess
        record_latent_trajectory(info, inverse, t_prev, img)

    return img, info


def denoise_rf_solver(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    trace = ensure_trajectory_trace(info)
    prev_velocity = None

    step_list = []
    record_latent_trajectory(info, inverse, timesteps[0], img)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        next_img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
        prev_velocity = update_trajectory_straightness(
            trace,
            prev_velocity,
            img,
            next_img,
            t_prev - t_curr,
            inverse,
        )
        img = next_img
        record_latent_trajectory(info, inverse, t_prev, img)

    return img, info


def denoise_fireflow(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    trace = ensure_trajectory_trace(info)
    prev_velocity = None

    step_list = []
    next_step_velocity = None
    record_latent_trajectory(info, inverse, timesteps[0], img)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        if next_step_velocity is None:
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info
            )
        else:
            pred = next_step_velocity
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )
        next_step_velocity = pred_mid
        
        next_img = img + (t_prev - t_curr) * pred_mid
        prev_velocity = update_trajectory_straightness(
            trace,
            prev_velocity,
            img,
            next_img,
            t_prev - t_curr,
            inverse,
        )
        img = next_img
        record_latent_trajectory(info, inverse, t_prev, img)

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
