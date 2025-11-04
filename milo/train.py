import os
import sys
import gc
import yaml
from functools import partial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import torch
from random import randint
from utils.loss_utils import l1_loss, L1_loss_appearance
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui
from gaussian_renderer import render_imp, render_simp, render_depth, render_full
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, read_config
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

import numpy as np
import time
import torch.nn.functional as F
from contextlib import nullcontext
from torch.profiler import (
    profile as torch_profile,
    schedule as profiler_schedule,
    tensorboard_trace_handler,
    ProfilerActivity,
)

from utils.geometry_utils import depth_to_normal
from utils.log_utils import log_training_progress
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.mesh import (
    initialize_mesh_regularization,
    compute_mesh_regularization,
    reset_mesh_state_at_next_iteration,
)
from regularization.regularizer.moge import (
    initialize_moge_supervision,
    compute_moge_regularization,
)

from regularization.bilateral_grid.lib_bilagrid import total_variation_loss
from torchvision.utils import save_image
from regularization.sdf.learnable import set_sdf_asinh_scale, set_sdf_asinh_enabled


def _accumulate_loss(loss_terms: dict, key: str, value: torch.Tensor) -> None:
    if value is None:
        return
    if key in loss_terms:
        loss_terms[key] = loss_terms[key] + value
    else:
        loss_terms[key] = value


def _get_loss_weight(spec, iteration: int) -> float:
    if spec is None:
        return 0.0
    if isinstance(spec, (int, float)):
        return float(spec)

    start_iter = int(spec.get("start_iter", 0))
    if iteration < start_iter:
        return 0.0

    end_iter = spec.get("end_iter")
    if end_iter is not None and iteration >= int(end_iter):
        return 0.0

    base = float(spec.get("value", 1.0))
    target = spec.get("target")
    if target is None:
        return base

    ramp_iters = float(spec.get("ramp_iters", spec.get("ramp_length", 0)))
    if ramp_iters <= 0:
        return float(target)

    progress = min(1.0, max(0.0, (iteration - start_iter) / ramp_iters))
    schedule = spec.get("ramp", "linear").lower()
    if schedule == "exp":
        # Exponential interpolation in log space
        base_safe = max(1e-8, base)
        target_safe = max(1e-8, float(target))
        return float(base_safe * (target_safe / base_safe) ** progress)
    # Default linear ramp
    return float(base + (float(target) - base) * progress)


def _combine_losses(loss_terms: dict, weight_spec: dict, iteration: int):
    if not loss_terms:
        raise ValueError("No loss terms provided for combination.")

    weights_applied = {}
    total_loss = None
    for name, tensor in loss_terms.items():
        spec = weight_spec.get(name)
        if spec is None:
            # Default to 1.0 for photo, else 0.0 so that missing entries are explicit
            spec = 1.0 if name == "photo" else 0.0
        weight = _get_loss_weight(spec, iteration)
        weights_applied[name] = weight

        if total_loss is None:
            total_loss = tensor * weight
        else:
            total_loss = total_loss + tensor * weight

    if total_loss is None:
        # Fallback to zero tensor on same device as any existing term
        sample_tensor = next(iter(loss_terms.values()))
        total_loss = torch.zeros_like(sample_tensor)

    return total_loss, weights_applied

def training(
    dataset, opt, pipe, 
    testing_iterations, saving_iterations, 
    checkpoint_iterations, checkpoint, 
    debug_from, args, 
    depth_order_config, moge_config, mesh_config,
    log_interval,
):
    # ---Prepare logger--- 
    run = prepare_output_and_logger(dataset, args)
    
    # ---Initialize scene and Gaussians---
    first_iter = 0
    use_mip_filter = not args.disable_mip_filter
    gaussians = GaussianModel(
        sh_degree=0, 
        use_mip_filter=use_mip_filter, 
        learn_occupancy=args.mesh_regularization,
        use_appearance_network=args.decoupled_appearance,
        use_bilateral_grid=args.use_bilateral_grid
    )
    scene = Scene(dataset, gaussians, resolution_scales=[1,2])
    # scene = Scene(dataset, gaussians, resolution_scales=[1])

    if args.use_bilateral_grid:
        gaussians.build_bilateral_grid(len(scene.train_cameras[1]), args.grid_shape)

    gaussians.training_setup(opt)
    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")
    print(f"[INFO] Using learnable SDF: {gaussians.learn_occupancy}")
    print(f"[INFO] Using bilateral grid: {args.use_bilateral_grid}")
    if args.dense_gaussians:
        print("[INFO] Using dense Gaussians.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if args.mesh_regularization:
            if first_iter > mesh_config["start_iter"]:
                mesh_config["start_iter"] = first_iter + 1
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize culling stats
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras()))
    
    # Initialize 3D Mip filter
    if use_mip_filter:
        gaussians.compute_3D_filter(cameras=scene.getTrainCameras_warn_up(first_iter + 1, args.warn_until_iter, scale=1.0, scale2=2.0).copy())

    # Additional variables
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    postfix_dict = {}
    ema_loss_for_log = 0.0
    ema_depth_normal_loss_for_log = 0.0
    
    # ---Prepare Mesh-In-the-Loop Regularization---
    if args.mesh_regularization:
        print("[INFO] Using mesh regularization.")
        mesh_renderer, mesh_state = initialize_mesh_regularization(
            scene=scene,
            config=mesh_config,
        )
    ema_mesh_depth_loss_for_log = 0.0
    ema_mesh_normal_loss_for_log = 0.0
    ema_occupied_centers_loss_for_log = 0.0
    ema_occupancy_labels_loss_for_log = 0.0
    
    # ---Prepare Depth-Order Regularization---    
    if args.depth_order:
        print("[INFO] Using depth order regularization.")
        print(f"        > Using expected depth with depth_ratio {depth_order_config['depth_ratio']} for depth order regularization.")
        depth_priors = initialize_depth_order_supervision(
            scene=scene,
            config=depth_order_config,
            device='cuda',
        )
    ema_depth_order_loss_for_log = 0.0
    moge_supervision = None
    if args.moge or args.moge_mask_training:
        print("[INFO] Initializing MoGe inference.")
        if args.moge and moge_config:
            print(f"        > Depth mode: {moge_config.get('depth', 'none')}")
            print(f"        > Normal supervision: {moge_config.get('normal', moge_config.get('noraml', False))}")
        moge_supervision = initialize_moge_supervision(
            scene=scene,
            config=moge_config,
            device="cuda",
        )
    if args.moge_mask_training:
        print("[INFO] Applying MoGe mask to photometric losses.")
    ema_moge_depth_loss_for_log = 0.0
    ema_moge_normal_loss_for_log = 0.0
    ema_tv_loss_for_log = 0.0
    ema_scale_loss_for_log = 0.0
    # Loss weight schedule setup
    loss_weight_cfg = mesh_config.get("loss_weights", {})
    default_loss_weights = {
        "photo": {"value": 1.0},
        "vis": {"value": 0.0},
        "ray": {"value": 0.0},
        "shape": {"value": 1.0},
    }
    for key, default in default_loss_weights.items():
        if key not in loss_weight_cfg:
            loss_weight_cfg[key] = default
    mesh_config["loss_weights"] = loss_weight_cfg
    use_sdf_asinh = bool(mesh_config.get("use_sdf_asinh", True))
    mesh_config["use_sdf_asinh"] = use_sdf_asinh
    if use_sdf_asinh:
        sdf_scale = mesh_config.get("sdf_asinh_scale")
        if sdf_scale is None:
            ratio = float(mesh_config.get("sdf_asinh_scale_ratio", 0.02))
            # Approximate scene diagonal using camera extent radius
            scene_diag = float(scene.cameras_extent) * 2.0
            sdf_scale = max(scene_diag * ratio, 1e-4)
        sdf_scale = float(sdf_scale)
        mesh_config["sdf_asinh_scale"] = sdf_scale
        set_sdf_asinh_enabled(True)
        set_sdf_asinh_scale(sdf_scale)
    else:
        mesh_config["sdf_asinh_scale"] = None
        set_sdf_asinh_enabled(False)
        
    # ---Profiler setup---
    if args.use_profiler:
        profiler_output_dir = os.path.join(args.model_path, "profiler")
        os.makedirs(profiler_output_dir, exist_ok=True)
        profiler_activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            profiler_activities.append(ProfilerActivity.CUDA)
        profiler_cm = torch_profile(
            activities=profiler_activities,
            schedule=profiler_schedule(wait=5, warmup=5, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                profiler_output_dir,  # 不要加子目录
                worker_name="worker0"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
)
        print(f"[INFO] PyTorch profiler enabled. Traces stored in {profiler_output_dir}")
    else:
        profiler_cm = nullcontext()
    
    # ---Log optimizable param groups---
    print(f"[INFO] Found {len(gaussians.optimizer.param_groups)} optimizable param groups:")
    n_total_params = 0
    for param in gaussians.optimizer.param_groups:
        name = param['name']
        n_params = len(param['params'])
        print(f"\n========== {name} ==========")
        print(f"Total number of param groups: {n_params}")
        for param_i in param['params']:
            print(f"   > Shape {param_i.shape}")
            n_total_params = n_total_params + param_i.numel()
    if gaussians.learn_occupancy:
        print(f"\n========== base_occupancy ==========")
        print(f"   > Not learnable")
        print(f"   > Shape {gaussians._base_occupancy.shape}")
    print(f"\nTotal number of optimizable parameters: {n_total_params}\n")
    
    # ---Start optimization loop---    
    with profiler_cm as profiler:
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1

        for iteration in range(first_iter, opt.iterations + 1):   

            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            # ---Update SH degree---
            if iteration % 1000 == 0 and iteration>args.simp_iteration1:
                gaussians.oneupSHdegree()

            # ---Select random viewpoint---
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
                viewpoint_idx_stack = list(range(len(viewpoint_stack)))

            _random_view_idx = randint(0, len(viewpoint_stack)-1)
            viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
            viewpoint_cam = viewpoint_stack.pop(_random_view_idx)

            # ---Render scene---
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            reg_kick_on = iteration >= args.regularization_from_iter
            mesh_kick_on = args.mesh_regularization and (iteration >= mesh_config["start_iter"])
            depth_order_kick_on = args.depth_order
            moge_enabled = args.moge
            moge_depth_mode = (
                moge_config.get("depth", "none").lower() if (moge_enabled and moge_config) else "none"
            )
            moge_requires_expected_depth = (
                moge_enabled
                and moge_config
                and moge_depth_mode != "none"
                and moge_config.get("depth_ratio", 1.0) < 1.0
            )
        
            # If depth-normal regularization or mesh-in-the-loop regularization are active,
            # we use the rasterizer compatible with depth and normal rendering.
            if reg_kick_on or mesh_kick_on:
                render_pkg = render(
                    viewpoint_cam, gaussians, pipe, background,
                    require_coord=False, require_depth=True,
                )
            
            # Else, if depth-order or MoGe supervision is active, we use Mini-Splatting2 rasterizer 
            # but we render depth maps. This rasterizer is necessary for densification and simplification.
            elif depth_order_kick_on or moge_enabled:
                render_pkg = render_full(
                    viewpoint_cam, gaussians, pipe, background, 
                    culling=gaussians._culling[:,viewpoint_cam.uid],
                    compute_expected_normals=False,
                    compute_expected_depth=(
                        depth_order_kick_on or moge_requires_expected_depth
                    ),
                    compute_accurate_median_depth_gradient=(
                        depth_order_kick_on or (moge_enabled and moge_depth_mode != "none")
                    ),
                )
            
            # If no regularization is active, we just use the default Mini-Splatting2 rasterizer.
            else:
                render_pkg = render_imp(
                    viewpoint_cam, gaussians, pipe, background, 
                    culling=gaussians._culling[:,viewpoint_cam.uid],
                )

            # ---Compute losses---
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"], render_pkg["viewspace_points"], 
                render_pkg["visibility_filter"], render_pkg["radii"]
            )
            gt_image = viewpoint_cam.original_image.cuda()

            moge_training_mask = None
            if (
                args.moge
                and args.moge_mask_training
                and moge_supervision is not None
            ):
                mask_tensor = moge_supervision["depth_mask"][viewpoint_idx].to(image.device)
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                if mask_tensor.shape[-2:] != image.shape[-2:]:
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0), size=image.shape[-2:], mode="nearest"
                    ).squeeze(0)
                mask_tensor = (mask_tensor > 0.5).to(image.dtype)
                if mask_tensor.sum() > 0:
                    moge_training_mask = mask_tensor
                else:
                    moge_training_mask = None
        
            if moge_training_mask is not None:
                mask_rgb = moge_training_mask.expand_as(image)
            else:
                mask_rgb = None

            if gaussians.use_bilateral_grid:
                if iteration % 500 == 0:
                    image_before = image.clone().detach()

                    save_image(image_before, f"img/render_before_bilateral_{viewpoint_idx}_{iteration}.png")
                image = gaussians._apply_bilateral_grid(
                    image, viewpoint_idx, viewpoint_cam.image_height, viewpoint_cam.image_width
                )
                if iteration % 500 == 0:
                    image_after = image.clone().detach()

                    save_image(image_after, f"img/render_after_bilateral_{viewpoint_idx}_{iteration}.png")
            # Rendering loss
            if args.decoupled_appearance:
                if mask_rgb is not None:
                    transformed_image = L1_loss_appearance(
                        image, gt_image, gaussians, viewpoint_cam.uid, return_transformed_image=True
                    )
                    diff = torch.abs(transformed_image - gt_image) * mask_rgb
                    Ll1 = diff.sum() / mask_rgb.sum().clamp_min(1e-6)
                    image_for_ssim = image * mask_rgb
                    gt_for_ssim = gt_image * mask_rgb
                else:
                    Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
                    image_for_ssim = image
                    gt_for_ssim = gt_image
            else:
                if mask_rgb is not None:
                    diff = torch.abs(image - gt_image) * mask_rgb
                    Ll1 = diff.sum() / mask_rgb.sum().clamp_min(1e-6)
                    image_for_ssim = image * mask_rgb
                    gt_for_ssim = gt_image * mask_rgb
                else:
                    Ll1 = l1_loss(image, gt_image)
                    image_for_ssim = image
                    gt_for_ssim = gt_image

            ssim_value = fused_ssim(image_for_ssim.unsqueeze(0), gt_for_ssim.unsqueeze(0))
            photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            loss_terms = {"photo": photo_loss}
        
            # Depth-Normal Consistency Regularization
            if reg_kick_on:
                rendered_depth_to_normals: torch.Tensor = depth_to_normal(
                    viewpoint_cam, 
                    render_pkg["median_depth"],  # 1, H, W
                    render_pkg["expected_depth"],  # 1, H, W
                )  # 3, H, W or 2, 3, H, W
                rendered_normals: torch.Tensor = render_pkg["normal"]  # 3, H, W
            
                if rendered_depth_to_normals.ndim == 4:
                    # If shape is 2, 3, H, W
                    reg_depth_ratio = 0.6
                    normal_error_map = 1. - (rendered_normals[None] * rendered_depth_to_normals).sum(dim=1)  # 2, H, W
                    depth_normal_loss = args.lambda_depth_normal * (
                        (1. - reg_depth_ratio) * normal_error_map[0].mean() 
                        + reg_depth_ratio * normal_error_map[1].mean()
                    )
                else:
                    # If shape is 3, H, W
                    depth_normal_loss = args.lambda_depth_normal * (1 - (rendered_normals * rendered_depth_to_normals).sum(dim=0)).mean()
            
                _accumulate_loss(loss_terms, "shape", depth_normal_loss)
            
            # Depth Order Regularization
            # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
            # > In the paper, MILo does not rely on any learned prior. 
            if depth_order_kick_on:
                if depth_order_config["depth_ratio"] < 1.:
                    depth_for_depth_order = (
                        (1. - depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                        + depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                    )
                else:
                    depth_for_depth_order = render_pkg["median_depth"]
                
                depth_prior_loss, _, do_supervision_depth, lambda_depth_order = compute_depth_order_regularization(
                    iteration=iteration,
                    rendered_depth=depth_for_depth_order,
                    depth_priors=depth_priors,
                    viewpoint_idx=viewpoint_idx,
                    gaussians=gaussians,
                    config=depth_order_config,
                )
                
                _accumulate_loss(loss_terms, "shape", depth_prior_loss)
                depth_order_kick_on = lambda_depth_order > 0
        
            moge_depth_loss = None
            moge_normal_loss = None
            moge_supervision_depth = None
            moge_supervision_normal = None
            moge_lambda = 0.0
            moge_supervision_mask = moge_training_mask
            if moge_enabled and moge_supervision is not None:
                moge_pkg = compute_moge_regularization(
                    iteration=iteration,
                    render_pkg=render_pkg,
                    viewpoint_idx=viewpoint_idx,
                    gaussians=gaussians,
                    config=moge_config,
                    moge_supervision=moge_supervision,
                )
                _accumulate_loss(loss_terms, "shape", moge_pkg["total_loss"])
                moge_depth_loss = moge_pkg["depth_loss"]
                moge_normal_loss = moge_pkg["normal_loss"]
                moge_supervision_depth = moge_pkg["supervision_depth"]
                moge_supervision_normal = moge_pkg["supervision_normal"]
                if moge_pkg["supervision_mask"] is not None:
                    moge_supervision_mask = moge_pkg["supervision_mask"]
                moge_lambda = moge_pkg["lambda_value"]
            moge_kick_on = (
                moge_enabled
                and (moge_lambda > 0)
                and (
                    (moge_depth_loss is not None)
                    or (moge_normal_loss is not None)
                )
            )
            moge_logging_active = moge_kick_on or (moge_training_mask is not None)

            # Mesh-In-the-Loop Regularization
            if mesh_kick_on:
                if args.detach_gaussian_rendering:
                    detached_render_pkg = {
                        "render": render_pkg["render"].detach(),
                        "median_depth": render_pkg["median_depth"].detach(),
                        "expected_depth": render_pkg["expected_depth"].detach(),
                        "normal": render_pkg["normal"].detach(),
                    }
            
                mesh_regularization_pkg = compute_mesh_regularization(
                    iteration=iteration,
                    render_pkg=detached_render_pkg if args.detach_gaussian_rendering else render_pkg,
                    viewpoint_cam=viewpoint_cam,
                    viewpoint_idx=viewpoint_idx,
                    gaussians=gaussians,
                    scene=scene,
                    pipe=pipe,
                    background=background,
                    kernel_size=0.0,
                    config=mesh_config,
                    mesh_renderer=mesh_renderer,
                    mesh_state=mesh_state,
                    render_func=partial(render, require_coord=False, require_depth=True),
                    weight_adjustment=100. / opt.iterations,
                    args=args,
                    integrate_func=integrate,
                )
                mesh_loss = mesh_regularization_pkg["mesh_loss"]
                mesh_depth_loss = mesh_regularization_pkg["mesh_depth_loss"]
                mesh_normal_loss = mesh_regularization_pkg["mesh_normal_loss"]
                occupied_centers_loss = mesh_regularization_pkg["occupied_centers_loss"]
                occupancy_labels_loss = mesh_regularization_pkg["occupancy_labels_loss"]
                mesh_state = mesh_regularization_pkg["updated_state"]
                mesh_render_pkg = mesh_regularization_pkg["mesh_render_pkg"]
            
                _accumulate_loss(loss_terms, "shape", mesh_loss)
        
            if gaussians.use_bilateral_grid:
                tv_loss = 10 * total_variation_loss(gaussians.bil_grids.grids)
                _accumulate_loss(loss_terms, "shape", tv_loss)
            else:
                tv_loss = None

            scale_loss = None
            if args.scale_regularization:
                scales = gaussians.get_scaling
                effective_radius = torch.linalg.vector_norm(scales, dim=-1)
                scale_loss = effective_radius.pow(3).mean()
                scale_loss = args.scale_regularization_weight * scale_loss
                _accumulate_loss(loss_terms, "shape", scale_loss)

            for required_loss_key in loss_weight_cfg.keys():
                if required_loss_key != "photo" and required_loss_key not in loss_terms:
                    loss_terms[required_loss_key] = torch.zeros_like(photo_loss)

            total_loss, weights_used = _combine_losses(loss_terms, loss_weight_cfg, iteration)
            for weight_name, weight_value in weights_used.items():
                postfix_dict[f"lambda_{weight_name}"] = weight_value

            # ---Backward pass---
            total_loss.backward()

            iter_end.record()

            with torch.no_grad():
                # ---Logging---
                (
                    postfix_dict,
                    ema_loss_for_log, 
                    ema_depth_normal_loss_for_log, 
                    ema_mesh_depth_loss_for_log, ema_mesh_normal_loss_for_log, 
                    ema_occupied_centers_loss_for_log, ema_occupancy_labels_loss_for_log, 
                    ema_depth_order_loss_for_log, ema_scale_loss_for_log, ema_moge_depth_loss_for_log, ema_moge_normal_loss_for_log
                ) = log_training_progress(
                    args, iteration, log_interval, progress_bar, run,
                    scene, gaussians, pipe, opt, background,
                    viewpoint_idx, viewpoint_cam, render_pkg, 
                    mesh_render_pkg if mesh_kick_on else None, 
                    do_supervision_depth if depth_order_kick_on else None,
                    reg_kick_on, mesh_kick_on, depth_order_kick_on,
                    total_loss, depth_normal_loss if reg_kick_on else None, 
                    mesh_depth_loss if mesh_kick_on else None, mesh_normal_loss if mesh_kick_on else None, 
                    occupied_centers_loss if mesh_kick_on else None, occupancy_labels_loss if mesh_kick_on else None, 
                    depth_prior_loss if depth_order_kick_on else None,
                    tv_loss,
                    scale_loss if args.scale_regularization else None,
                    mesh_config if mesh_kick_on else None, 
                    postfix_dict, ema_loss_for_log, ema_depth_normal_loss_for_log, ema_mesh_depth_loss_for_log, 
                    ema_mesh_normal_loss_for_log, ema_occupied_centers_loss_for_log, ema_occupancy_labels_loss_for_log,
                    ema_depth_order_loss_for_log, ema_scale_loss_for_log, ema_tv_loss_for_log, testing_iterations, saving_iterations, render_imp,
                    moge_logging_active,
                    moge_depth_loss if moge_kick_on else None,
                    moge_normal_loss if moge_kick_on else None,
                    ema_moge_depth_loss_for_log,
                    ema_moge_normal_loss_for_log,
                    moge_supervision_depth if moge_kick_on else None,
                    moge_supervision_normal if moge_kick_on else None,
                    moge_supervision_mask,
                )

                # ---Densification---
                gaussians_have_changed = False
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                    if gaussians._culling[:,viewpoint_cam.uid].sum()==0:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    else:
                        # normalize xy gradient after culling
                        gaussians.add_densification_stats_culling(viewspace_point_tensor, visibility_filter, gaussians.factor_culling)

                    area_max = render_pkg["area_max"]
                    mask_blur = torch.logical_or(mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != args.depth_reinit_iter:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune_mask(opt.densify_grad_threshold, 
                                                        0.005, scene.cameras_extent, 
                                                        size_threshold, mask_blur)
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                        gaussians_have_changed = True
                        if use_mip_filter:
                            gaussians.compute_3D_filter(
                                cameras=scene.getTrainCameras_warn_up(
                                    iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                                ).copy()
                            )
                    
                    if iteration == args.depth_reinit_iter:

                        num_depth = gaussians._xyz.shape[0]*args.num_depth_factor

                        # interesction_preserving for better point cloud reconstruction result at the early stage, not affect rendering quality
                        gaussians.interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                        if use_mip_filter:
                            gaussians.compute_3D_filter(
                                cameras=scene.getTrainCameras_warn_up(
                                    iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                                ).copy()
                            )
                        
                        pts, rgb = gaussians.depth_reinit(scene, render_depth, iteration, num_depth, args, pipe, background)

                        gaussians.reinitial_pts(pts, rgb)

                        gaussians.training_setup(opt)
                        gaussians.init_culling(len(scene.getTrainCameras()))
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                        torch.cuda.empty_cache()
                        gaussians_have_changed = True
                        if use_mip_filter:
                            gaussians.compute_3D_filter(
                                cameras=scene.getTrainCameras_warn_up(
                                    iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                                ).copy()
                            )

                    if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration!=args.depth_reinit_iter:
                        gaussians.culling_with_clone(scene, render_simp, iteration, args, pipe, background)
                        torch.cuda.empty_cache()
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                        gaussians_have_changed = True
                        if use_mip_filter:
                            gaussians.compute_3D_filter(
                                cameras=scene.getTrainCameras_warn_up(
                                    iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                                ).copy()
                            )

                # ---Pruning and simplification---
                if iteration == args.simp_iteration1:
                    if args.dense_gaussians:
                        gaussians.culling_with_importance_pruning(scene, render_simp, iteration, args, pipe, background)
                    else:
                        gaussians.culling_with_interesction_sampling(scene, render_simp, iteration, args, pipe, background)
                    gaussians.max_sh_degree=dataset.sh_degree
                    gaussians.extend_features_rest()

                    gaussians.training_setup(opt)
                    torch.cuda.empty_cache()
                    gaussians_have_changed = True
                    if use_mip_filter:
                            gaussians.compute_3D_filter(
                                cameras=scene.getTrainCameras_warn_up(
                                    iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                                ).copy()
                            )
                
                if iteration == args.simp_iteration2:
                    if args.dense_gaussians:
                        gaussians.culling_with_importance_pruning(scene, render_simp, iteration, args, pipe, background)
                    else:
                        gaussians.culling_with_interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                    torch.cuda.empty_cache()
                    gaussians_have_changed = True
                    if use_mip_filter:
                            gaussians.compute_3D_filter(
                                cameras=scene.getTrainCameras_warn_up(
                                    iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                                ).copy()
                            )

                if iteration == (args.simp_iteration2+opt.iterations)//2:
                    gaussians.init_culling(len(scene.getTrainCameras()))

                # ---Reset mesh state if Gaussians have changed---
                if mesh_kick_on and gaussians_have_changed:
                    mesh_state = reset_mesh_state_at_next_iteration(mesh_state)
                
                # ---Update 3D Mip Filter---
                if use_mip_filter and (
                    (iteration == args.warn_until_iter)
                    or (iteration % args.update_mip_filter_every == 0)
                ):
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy())

                # ---Optimizer step---
                if iteration < opt.iterations:
                    if gaussians.use_appearance_network or gaussians.use_bilateral_grid:
                        gaussians.optimizer.step()
                    else:
                        visible = radii>0
                        gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # ---Save checkpoint---
                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  
                
            if iteration % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            if profiler is not None:
                profiler.step()
    print('Num of Gaussians: %d'%(gaussians._xyz.shape[0]))
    
    if WANDB_FOUND:
        run.finish()
    
    return 


def prepare_output_and_logger(dataset, args):    
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create WandB run       
    global WANDB_FOUND
    WANDB_FOUND = (
        WANDB_FOUND
        and (args.wandb_project is not None)
        and (args.wandb_entity is not None)
    )
    if WANDB_FOUND:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
        )
    else:
        run=None
        print("[INFO] WandB not found, skipping logging.")
    return run


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ----- Usual arguments -----
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[8000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # ----- Rasterization technique -----
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "gof"])
    
    # ----- Mesh-In-the-Loop Regularization -----
    parser.add_argument("--no_mesh_regularization", action="store_true")
    parser.add_argument("--mesh_config", type=str, default="default")
    # Gaussians management
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--detach_gaussian_rendering", action="store_true")

    # ----- Densification and Simplification -----
    # > Inspired by Mini-Splatting2.
    # > Used for pruning, densification and Gaussian pivots selection.
    parser.add_argument("--imp_metric", required=True, type=str, choices=["outdoor", "indoor"])
    parser.add_argument("--config_path", type=str, default="./configs/fast")
    # Aggressive Cloning
    parser.add_argument("--aggressive_clone_from_iter", type=int, default = 500)
    parser.add_argument("--aggressive_clone_interval", type=int, default = 250)
    # Depth Reinitialization
    parser.add_argument("--warn_until_iter", type=int, default = 3000)
    parser.add_argument("--depth_reinit_iter", type=int, default=2_000)
    parser.add_argument("--num_depth_factor", type=float, default=1)
    # Simplification
    parser.add_argument("--simp_iteration1", type=int, default = 3_000)
    parser.add_argument("--simp_iteration2", type=int, default = 8_000)
    parser.add_argument("--sampling_factor", type=float, default = 0.6)
    
    # ----- Depth-Normal consistency Regularization -----
    # > Inspired by 2DGS, GOF, RaDe-GS...
    parser.add_argument("--regularization_from_iter", type=int, default = 3_000)
    parser.add_argument("--lambda_depth_normal", type=float, default = 0.05)
    
    # ----- Scale Regularization -----
    parser.add_argument("--scale_regularization", action="store_true",
                        help="Enable radius-based scale regularization on Gaussians.")
    parser.add_argument("--scale_regularization_weight", type=float, default=1e-3,
                        help="Weight applied to the scale regularization loss.")
    
    # ----- Depth Order Regularization (Learned Prior) -----
    # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
    # > In the paper, MILo does not rely on any learned prior.
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--depth_order_config", type=str, default="default")

    parser.add_argument("--moge", action="store_true")
    parser.add_argument("--moge_config", type=str, default="default")
    parser.add_argument("--moge_mask_training", action="store_true",
                        help="Apply MoGe mask to primary photometric losses.")

    # ----- 3D Mip Filter -----
    # > Inspired by Mip-Splatting.
    parser.add_argument("--disable_mip_filter", action="store_true", default=False)
    parser.add_argument("--update_mip_filter_every", type=int, default=100)

    # ----- Appearance Network for Exposure-aware loss -----
    # > Inspired by GOF.
    parser.add_argument("--decoupled_appearance", action="store_true")

    # ---- bilateral_grid
    parser.add_argument("--use_bilateral_grid", action="store_true")
    # ----- Logging -----
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_profiler", action="store_true", help="Enable PyTorch profiler with TensorBoard trace output.")
    
    args = parser.parse_args(sys.argv[1:])

    args = read_config(parser)
    args.save_iterations.append(args.iterations)
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.mesh_regularization = not args.no_mesh_regularization
    
    if args.port == -1:
        args.port = np.random.randint(5000, 9000)
        print(f"Using random port: {args.port}")
    
    # Load depth order regularization config (not used in MILo paper)
    if args.depth_order:
        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            depth_order_config = yaml.safe_load(f)
    else:
        depth_order_config = None
        
    if args.moge or args.moge_mask_training:
        moge_config_file = os.path.join(BASE_DIR, "configs", "moge_supervise", f"{args.moge_config}.yaml")
        with open(moge_config_file, "r") as f:
            moge_config = yaml.safe_load(f)
        if "normal" not in moge_config and "noraml" in moge_config:
            moge_config["normal"] = moge_config["noraml"]
    else:
        moge_config = None

    # Load mesh-in-the-loop regularization config
    if args.mesh_regularization:
        # Get mesh regularization config file
        mesh_config_file = os.path.join(BASE_DIR, "configs", "mesh", f"{args.mesh_config}.yaml")
        with open(mesh_config_file, "r") as f:
            mesh_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh regularization with config: {args.mesh_config}")
    else:
        mesh_config = None
    
    # Message for imp_metric
    print(f"[INFO] Using importance metric: {args.imp_metric}.")
    
    # Message for detach_gaussian_rendering
    if args.detach_gaussian_rendering:
        print(f"[INFO] Detaching Gaussian rendering for mesh regularization.")
    
    # Import rendering function
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
    elif args.rasterizer == "gof":
        from gaussian_renderer.gof import render_gof as render
        from gaussian_renderer.gof import integrate_gof as integrate
        
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start=time.time()
    
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,
        depth_order_config,
        moge_config,
        mesh_config,
        args.log_interval,
    )

    torch.cuda.synchronize()
    time_end=time.time()
    time_total=time_end-time_start
    print('time: %fs'%(time_total))

    time_txt_path=os.path.join(args.model_path, r'time.txt')
    with open(time_txt_path, 'w') as f:  
        f.write(str(time_total)) 

    # All done
    print("\nTraining complete.")
