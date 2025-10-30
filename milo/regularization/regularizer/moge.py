import gc
from typing import Dict, Any, Optional

import torch
from tqdm import tqdm

def _get_lambda_from_schedule(iteration: int, config: Dict[str, Any]) -> float:
    lambda_value = config.get("weight_initial_value", 0.0)
    update_iters = config.get("weight_update_iters", [])
    update_values = config.get("weight_update_values", [])

    for idx, update_iter in enumerate(update_iters):
        if iteration == update_iter:
            print(
                f"[INFO] Updating MoGe supervision weight to "
                f"{update_values[idx]} at iteration {iteration}."
            )
        if iteration >= update_iter and idx < len(update_values):
            lambda_value = update_values[idx]

    return lambda_value


def initialize_moge_supervision(scene, config: Dict[str, Any], device: str = "cuda") -> Dict[str, list]:
    """
    Runs the MoGe model once on each training image to build supervision maps.
    Returns a dictionary with depth and normal lists stored on CPU.
    """
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise ImportError("MoGe package not found. Please ensure it is installed.") from exc

    checkpoint_path = config.get("moge_checkpoint_dir")
    if checkpoint_path is None:
        raise ValueError("`moge_checkpoint_dir` must be set in the MoGe config.")

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    moge_model = MoGeModel.from_pretrained(checkpoint_path).to(device_obj)
    moge_model.eval()

    train_cameras = scene.getTrainCameras().copy()
    depth_supervision = []
    depth_masks = []
    normal_supervision = []

    print("[INFO] Building MoGe supervision maps...")
    with torch.no_grad():
        for cam in tqdm(range(len(train_cameras)), desc="MoGe inference"):
            viewpoint = train_cameras[cam]
            image = viewpoint.original_image.to(device_obj)
            if image.ndim != 3:
                raise ValueError("Camera images are expected to be 3-channel tensors.")
            input_tensor = image.unsqueeze(0)
            output = moge_model.infer(input_tensor)

            depth = output["depth"].squeeze(0).to(torch.float32)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)

            mask = output.get("mask")
            if mask is not None:
                mask = mask.squeeze(0) if mask.ndim == 3 else mask
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                depth_mask = (mask > 0.5).to(torch.float32)
            else:
                depth_mask = torch.isfinite(depth).to(torch.float32)

            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = depth * depth_mask

            normal = output.get("normal")
            if normal is not None:
                normal = normal.squeeze(0)
                if normal.ndim == 3 and normal.shape[0] != 3:
                    normal = normal.permute(2, 0, 1)
                if normal.shape[0] != 3:
                    raise ValueError("MoGe normals are expected to have 3 channels.")
                normal = normal.to(torch.float32)
                normal = torch.where(
                    (depth_mask > 0.5),
                    normal,
                    torch.zeros_like(normal),
                )
            depth_supervision.append(depth.cpu())
            depth_masks.append(depth_mask.cpu())
            normal_supervision.append(normal.cpu() if normal is not None else None)

    del moge_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] MoGe supervision maps built.")

    return {"depth": depth_supervision, "depth_mask": depth_masks, "normal": normal_supervision}


def _resize_map(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if tensor.shape[-2:] == target_shape:
        return tensor
    return torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=target_shape,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)


def _resize_mask(mask: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    mask = mask.to(torch.float32)
    if mask.shape[-2:] == target_shape:
        return (mask > 0.5).to(mask.dtype)
    resized = torch.nn.functional.interpolate(
        mask.unsqueeze(0),
        size=target_shape,
        mode="nearest",
    ).squeeze(0)
    return (resized > 0.5).to(mask.dtype)


def _masked_depth_order_loss(
    rendered_depth: torch.Tensor,
    supervision_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    scene_extent: float,
    config: Dict[str, Any],
) -> Optional[torch.Tensor]:
    valid_indices = valid_mask.nonzero(as_tuple=False)
    if valid_indices.numel() == 0:
        return None

    rendered_vals = rendered_depth[valid_mask]
    supervision_vals = supervision_depth[valid_mask]
    n_valid = rendered_vals.numel()
    if n_valid < 2:
        return None

    sample_count = min(config.get("order_sample_count", 4096), n_valid)
    if sample_count < 2:
        return None

    idx1 = torch.randint(0, n_valid, (sample_count,), device=rendered_depth.device)
    idx2 = torch.randint(0, n_valid, (sample_count,), device=rendered_depth.device)
    diff = (rendered_vals[idx1] - rendered_vals[idx2]) / max(scene_extent, 1e-6)
    prior_diff = (supervision_vals[idx1] - supervision_vals[idx2]) / max(scene_extent, 1e-6)

    if config.get("normalize_loss", True):
        prior_diff = prior_diff / prior_diff.detach().abs().clamp(min=1e-8)

    depth_order_loss = - (diff * prior_diff).clamp(max=0)
    if config.get("log_space", False):
        depth_order_loss = torch.log1p(config.get("log_scale", 20.0) * depth_order_loss)

    reduction = config.get("reduction", "mean")
    if reduction == "mean":
        depth_order_loss = depth_order_loss.mean()
    elif reduction == "sum":
        depth_order_loss = depth_order_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return depth_order_loss


def compute_moge_regularization(
    iteration: int,
    render_pkg: Dict[str, torch.Tensor],
    viewpoint_idx: int,
    gaussians,
    config: Dict[str, Any],
    moge_supervision: Dict[str, list],
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Computes MoGe-based auxiliary supervision losses (relative depth, absolute depth, normals).
    """
    lambda_weight = _get_lambda_from_schedule(iteration, config)
    if lambda_weight <= 0:
        return {
            "total_loss": torch.tensor(0.0, device=render_pkg["render"].device),
            "depth_loss": None,
            "normal_loss": None,
            "supervision_depth": None,
            "supervision_normal": None,
            "supervision_mask": None,
            "lambda_value": lambda_weight,
        }

    device = render_pkg["render"].device
    depth_cfg = config.get("depth", "none")
    normal_enabled = config.get("normal", config.get("noraml", False))
    depth_loss = None
    normal_loss = None
    supervision_depth = None
    supervision_normal = None
    supervision_mask = None
    total_loss: Optional[torch.Tensor] = None

    if depth_cfg and depth_cfg.lower() != "none":
        supervision_depth = moge_supervision["depth"][viewpoint_idx].to(device)
        supervision_mask = moge_supervision["depth_mask"][viewpoint_idx].to(device)
        supervision_depth = _resize_map(
            supervision_depth,
            render_pkg["median_depth"].shape[-2:],
        )
        supervision_mask = _resize_mask(
            supervision_mask,
            render_pkg["median_depth"].shape[-2:],
        ).to(torch.bool)

        # Choose which rendered depth to use
        ratio = config.get("depth_ratio", 1.0)
        if ratio < 1.0 and "expected_depth" in render_pkg:
            rendered_depth = (
                (1.0 - ratio) * render_pkg["expected_depth"]
                + ratio * render_pkg["median_depth"]
            )
        else:
            rendered_depth = render_pkg["median_depth"]

        if depth_cfg.lower() == "order":
            depth_loss = _masked_depth_order_loss(
                rendered_depth=rendered_depth.squeeze(),
                supervision_depth=supervision_depth.squeeze(),
                valid_mask=supervision_mask.squeeze(),
                scene_extent=gaussians.spatial_lr_scale,
                config=config,
            )
        elif depth_cfg.lower() == "absolute":
            valid_mask = supervision_mask.squeeze()
            if valid_mask.any():
                normalize_abs = config.get("normalize_absolute", True)
                diff = (rendered_depth - supervision_depth).squeeze()[valid_mask]
                if normalize_abs:
                    diff = diff / max(gaussians.spatial_lr_scale, 1e-6)
                depth_loss = diff.abs().mean()
            else:
                depth_loss = None
        else:
            raise ValueError(f"Unknown MoGe depth supervision mode: {depth_cfg}")

        if depth_loss is not None:
            depth_loss = lambda_weight * depth_loss
            total_loss = depth_loss if total_loss is None else total_loss + depth_loss

    if normal_enabled:
        supervision_normal = moge_supervision["normal"][viewpoint_idx]
        if supervision_normal is not None:
            supervision_normal_map = supervision_normal.to(device)
            supervision_normal_map = _resize_map(
                supervision_normal_map,
                render_pkg["normal"].shape[-2:],
            )
            if supervision_mask is None:
                supervision_mask = torch.ones_like(supervision_normal_map[0:1], dtype=torch.bool, device=device)
            supervision_mask_bool = supervision_mask.squeeze()
            if supervision_mask_bool.any():
                normal_vals = supervision_normal_map.reshape(3, -1)
                rendered_normals = render_pkg["normal"].reshape(3, -1)
                mask_flat = supervision_mask_bool.reshape(-1)
                normal_vals = normal_vals[:, mask_flat]
                rendered_normals = rendered_normals[:, mask_flat]
                normal_vals = normal_vals / normal_vals.norm(
                    dim=0, keepdim=True
                ).clamp_min(1e-6)

                rendered_normals = rendered_normals / rendered_normals.norm(
                    dim=0, keepdim=True
                ).clamp_min(1e-6)
                cosine = (rendered_normals * normal_vals).sum(dim=0).clamp(-1.0, 1.0)

                normal_weight = config.get("normal_weight_multiplier", 1.0) * lambda_weight
                normal_loss = normal_weight * (1.0 - cosine).mean()
                total_loss = normal_loss if total_loss is None else total_loss + normal_loss
                supervision_normal = supervision_normal_map * supervision_mask_bool.unsqueeze(0)
            else:
                supervision_normal = None

    if total_loss is None:
        total_loss = torch.zeros(1, device=device, dtype=render_pkg["render"].dtype).squeeze()

    supervision_mask_to_log = None
    if supervision_mask is not None:
        supervision_mask_to_log = supervision_mask.to(torch.float32)

    return {
        "total_loss": total_loss,
        "depth_loss": depth_loss,
        "normal_loss": normal_loss,
        "supervision_depth": supervision_depth,
        "supervision_normal": supervision_normal,
        "supervision_mask": supervision_mask_to_log,
        "lambda_value": lambda_weight,
    }
