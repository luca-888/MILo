import numpy as np
import torch
import os
import argparse
from PIL import Image
from blender.loading import (
    load_blender_package,
    load_scene_state_from_blender_package,
)
from blender.editing import (
    get_edited_gaussians_from_scene_state,
    EditedScenePipelineParams,
)


if __name__ == "__main__":
    print_every_n_frames = 5
    
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to render Frosting scenes edited or animated with Blender.')
    
    parser.add_argument('-p', '--package_path',
                        type=str, 
                        help='(Required) path to the Blender data package to use for rendering.')
    
    parser.add_argument('-o', '--output_path',
                        type=str, 
                        default=None,
                        help='Path to the output folder where to save the rendered images. \
                        If None, images will be saved in ./blender/renders/{package_name}.')
    
    parser.add_argument('--rasterizer', type=str, default='radegs', choices=['radegs', 'gof'],
                        help='Rasterizer to use for rendering.')
    
    parser.add_argument('--binding_mode', type=str, default='triangles', choices=['triangles', 'vertices'],
                        help='Mode to use for binding Gaussians to the mesh. '
                        'If "triangles", Gaussians will be bound to the closest triangle and deformed with triangles axes.'
                        'If "vertices", Gaussians will be bound to the closest vertices and deformed with them.'
                        )
    
    parser.add_argument('--n_verts_per_gaussian', type=int, default=8, 
                        help='Number of vertices per Gaussian to use for deformation. Only used if binding_mode is vertices.')
    
    parser.add_argument('--filter_big_gaussians_with_th', type=float, default=2., 
                        help='Threshold for filtering big Gaussians. '
                        'A Gaussian is considered too big if its scale increases by a ratio greater than this threshold.'
                        'The big Gaussians will not be rendered.')
    
    parser.add_argument('--clamp_big_gaussians_with_th', type=float, default=2., 
                        help='Threshold for clamping big Gaussians. '
                        'A Gaussian is considered too big if its scale increases by a ratio greater than this threshold.'
                        'The big Gaussians will be clamped to this threshold.')
    
    parser.add_argument('--filter_distant_gaussians_with_th', type=float, default=2., 
                        help='Threshold for filtering distant Gaussians. '
                        'A Gaussian is considered too distant if its relative distance to its bound mesh is greater than this threshold.'
                        'The distant Gaussians will not be rendered.')
    
    parser.add_argument('--export_frame_as_ply', type=int, default=-1, 
                        help='Export the edited 3DGS representation of the scene at the specified frame as a PLY file. '
                        'If -1, no PLY file will be exported and all frames will be rendered.')

    parser.add_argument('--white_background', action='store_true', help='Use a white background instead of black.')
    
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = os.path.join('blender', 'renders', os.path.splitext(os.path.basename(args.package_path))[0])
    os.makedirs(args.output_path, exist_ok=True)
    print(f"[INFO] Results will be saved in {args.output_path}")
    print(f"[INFO] Binding mode: {args.binding_mode}")
    if args.binding_mode == 'vertices':
        print(f"[INFO] Number of vertices per Gaussian: {args.n_verts_per_gaussian}")
    
    # ----- Load Blender package -----
    pkg = load_blender_package(args.package_path)
    
    # ----- Build scene state -----
    scene_state = load_scene_state_from_blender_package(
        pkg, 
        n_verts_per_gaussian=args.n_verts_per_gaussian, 
        bind_to_triangles=args.binding_mode == 'triangles'
    )
    print(f"Loaded scene state.")
    
    # ----- Get render cameras -----
    render_cameras = scene_state['render_cameras']
    print(f"Created {len(render_cameras)} render cameras.")
    
    # ----- Create pipeline -----
    pipe = EditedScenePipelineParams()
    if args.white_background:
        background = torch.ones(3, dtype=torch.float32, device='cuda')
    else:
        background = torch.zeros(3, dtype=torch.float32, device='cuda')
    
    # ----- Render -----
    if args.rasterizer == 'radegs':
        from gaussian_renderer import render_radegs as render
    elif args.rasterizer == 'gof':
        from gaussian_renderer import render_gof as render
    else:
        raise NotImplementedError(f"Rasterizer {args.rasterizer} not implemented.")
    
    with torch.no_grad():
        if args.export_frame_as_ply == -1:
            for i_frame, render_camera in enumerate(render_cameras):
                if True:
                    # Render Gaussians
                    gaussians = get_edited_gaussians_from_scene_state(
                        scene_state=scene_state,
                        i_frame=i_frame,
                        package=pkg,
                        camera=render_cameras[i_frame],
                        filter_big_gaussians_with_th=args.filter_big_gaussians_with_th,
                        clamp_big_gaussians_with_th=args.clamp_big_gaussians_with_th,
                        filter_distant_gaussians_with_th=args.filter_distant_gaussians_with_th,
                    )
                    
                    rgb_render = render(
                        viewpoint_camera=render_cameras[i_frame], 
                        pc=gaussians, 
                        pipe=pipe, 
                        bg_color=background, 
                        kernel_size=0.0, 
                        scaling_modifier = 1.0, 
                        require_coord=False, 
                        require_depth=True,
                    )["render"].nan_to_num().clamp(min=0, max=1).permute(1, 2, 0)
                    
                    # Save image
                    try:
                        save_path = os.path.join(args.output_path, f"{i_frame+1:04d}.png")
                        img = Image.fromarray((rgb_render.cpu().numpy() * 255).astype(np.uint8))
                        img.save(save_path)
                    except:
                        print(f"Error saving frame {i_frame+1} to {save_path}")
                    
                    # Info
                    if i_frame % print_every_n_frames == 0:
                        print(f"Saved frame {i_frame+1} to {save_path}")
                        
                    torch.cuda.empty_cache()
        else:
            raise NotImplementedError("Exporting frame as PLY file is not implemented yet.")
            ply_save_path = os.path.join(args.output_path, f"{args.export_frame_as_ply+1:04d}.ply")
            print(f"Exported PLY file of frame {args.export_frame_as_ply+1} to {args.ply_save_path}")

    print("Rendering completed.")