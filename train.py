import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from scene.temporal_model import TemporalModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, TemporalParams, get_combined_args
from utils.temporal_utils import generate_deformation_sequence, compute_temporal_loss

def train_deformablegs(dataset, opt, pipe, testing_iterations, saving_iterations):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations))
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    for iteration in range(1, opt.iterations + 1):
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            time_interval = 1 / len(scene.getTrainCameras())
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image = render_pkg_re["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii
            radii = render_pkg_re["radii"]
            gaussians.max_radii2D[render_pkg_re["visibility_filter"]] = torch.max(
                gaussians.max_radii2D[render_pkg_re["visibility_filter"]],
                radii[render_pkg_re["visibility_filter"]]
            )

            # Save checkpoints
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, render_pkg_re["visibility_filter"])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

def train_temporal(dataset, temporal_params, opt):

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.load_weights(dataset.model_path)

    train_cams = scene.getTrainCameras()
    print(f"Training cameras: {len(train_cams)}")

    # Initialize temporal model
    xyz = gaussians.get_xyz
    N = xyz.shape[0]
    flatten_dim = N * (3 + 4 + 3)

    temporal = TemporalModel(
        input_dim=flatten_dim,
        hidden_dim=temporal_params.hidden_dim,
        num_layers=temporal_params.num_layers,
        output_dim=flatten_dim * temporal_params.output_frame,
        lr=temporal_params.temporal_lr_init
    )
    temporal.train_setting(opt)
    temporal.load_weights(dataset.model_path, iteration=-1)

    print(f"Temporal model: input={flatten_dim}, hidden={temporal_params.hidden_dim}")
    print(f"Training for {temporal_params.temporal_iterations} iterations...")

    # Training loop
    progress_bar = tqdm(range(temporal_params.temporal_iterations))
    cam_index = 0
    num_cams = len(train_cams)

    all_losses = []

    for iter_idx in progress_bar:
        cam = train_cams[cam_index]
        cam_index = (cam_index + 1) % (num_cams - temporal_params.seq_len - temporal_params.output_frame + 1)

        if dataset.load2gpu_on_the_fly:
            cam.load2device()

        # Generate time steps
        time_steps = [train_cams[cam_index + i].fid.item()
                     for i in range(temporal_params.seq_len + temporal_params.output_frame)]

        # Generate deformation sequence
        sequence = generate_deformation_sequence(deform, xyz, time_steps, device='cuda')

        # First seq_len frames as input, last output_frame frames as target
        input_seq = sequence[:temporal_params.seq_len].unsqueeze(0)
        target = sequence[temporal_params.seq_len:].unsqueeze(0)

        # Forward pass
        temporal.train_mode()
        pred = temporal.step(input_seq)
        pred = pred.view(1, temporal_params.output_frame, flatten_dim)

        # Compute loss
        loss_weights = {'xyz': 0.0, 'rot': 0.0, 'scale': 0.0}
        loss = 0
        for i in range(temporal_params.output_frame):
            loss += compute_temporal_loss(pred[:, i, :], target[:, i, :], N, loss_weights)
        loss /= temporal_params.output_frame

        # Backward pass
        temporal.optimizer.zero_grad()
        loss.backward()
        temporal.optimizer.step()
        current_loss = loss.item()
        all_losses.append(current_loss)
        progress_bar.set_postfix({"loss": f"{current_loss:.6f}"})

        if dataset.load2gpu_on_the_fly:
            cam.load2device('cpu')

    avg_loss = sum(all_losses) / len(all_losses)
    print(f"Average temporal loss: {avg_loss:.6f}")

    temporal.save_weights(dataset.model_path, temporal_params.temporal_iterations)
    print(f"Temporal model saved at iteration {temporal_params.temporal_iterations}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    return None


if __name__ == "__main__":
    parser = ArgumentParser(description="FutureGS Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    temporal = TemporalParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)

    dataset = lp.extract(args)

    train_deformablegs(dataset, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)
    train_temporal(dataset, temporal.extract(args), op.extract(args))