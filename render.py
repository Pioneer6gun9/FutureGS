import torch
from tqdm import tqdm
from torchvision.utils import save_image
from pathlib import Path

from scene import Scene, DeformModel
from scene.temporal_model import TemporalModel
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, TemporalParams, get_combined_args
from utils.temporal_utils import generate_deformation_sequence, predict_with_sliding_window, parse_deformation
from utils.rigid_utils import build_knn_graph


def render_temporal_predictions(dataset, iteration, pipeline, background, temporal_params):
    """
    Render temporal prediction results (chain prediction)
    """
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.load_weights(dataset.model_path)
    xyz = scene.gaussians.get_xyz
    N = xyz.shape[0]
    flatten_dim = N * (3 + 4 + 3)

    temporal = TemporalModel(
        input_dim=flatten_dim,
        hidden_dim=temporal_params.hidden_dim,
        num_layers=temporal_params.num_layers,
        output_dim=flatten_dim * temporal_params.output_frame
    )
    temporal.load_weights(dataset.model_path, iteration=-1)

    # Build KNN graph
    knn_indices, knn_weights = build_knn_graph(
        xyz,
        k_neighbors=temporal_params.k_neighbors,
        theta_w=temporal_params.theta_w,
        device='cuda'
    )

    # Save directories
    base_save_dir = Path(dataset.model_path) / "temporal" / "predictions"
    renders_dir = base_save_dir / "renders"
    gt_dir = base_save_dir / "gt"
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Get cameras
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    print(f"Test cameras: {len(test_cams)}")
    print(f"Seq len: {temporal_params.seq_len}, Output frame: {temporal_params.output_frame}")

    # === First frame: Generate initial sequence using deform ===
    first_cam = test_cams[0]
    if dataset.load2gpu_on_the_fly:
        first_cam.load2device()

    # Use last seq_len+output_frame frames from training set to generate initial sequence
    start_idx = len(train_cams) - temporal_params.seq_len - temporal_params.output_frame
    time_steps = [train_cams[start_idx + i].fid.item()
                 for i in range(temporal_params.seq_len + temporal_params.output_frame)]

    print(f"Initial time steps: {time_steps}")

    # Generate sequence
    sequence = generate_deformation_sequence(deform, xyz, time_steps, device='cuda')
    input_seq = sequence[:temporal_params.seq_len]

    # Predict first frame
    pred_flat = predict_with_sliding_window(
        temporal, input_seq, temporal_params.seq_len,
        temporal_params.output_frame, flatten_dim,
        knn_indices, knn_weights, N
    )

    # Parse and render
    d_xyz, rotation, scaling = parse_deformation(pred_flat, N)

    with torch.no_grad():
        render_out = render(
            first_cam, gaussians, pipeline, background,
            d_xyz, rotation, scaling, dataset.is_6dof
        )
        pred_img = torch.clamp(render_out["render"], 0.0, 1.0)

    gt_img = torch.clamp(first_cam.original_image.to("cuda"), 0.0, 1.0)

    save_image(pred_img.cpu(), renders_dir / "pred_cam0.png")
    save_image(gt_img.cpu(), gt_dir / "gt_cam0.png")

    # Update sliding window
    predicted_seq = torch.cat([sequence[1:], pred_flat.unsqueeze(0)], dim=0)

    if dataset.load2gpu_on_the_fly:
        first_cam.load2device('cpu')

    # === Subsequent cameras: Chain prediction ===
    for idx, cam in enumerate(tqdm(test_cams[1:], desc="Chain Prediction"), start=1):
        if dataset.load2gpu_on_the_fly:
            cam.load2device()

        pred_flat = predict_with_sliding_window(
            temporal, predicted_seq, temporal_params.seq_len,
            temporal_params.output_frame, flatten_dim,
            knn_indices, knn_weights, N
        )

        d_xyz, rotation, scaling = parse_deformation(pred_flat, N)

        with torch.no_grad():
            render_out = render(
                cam, gaussians, pipeline, background,
                d_xyz, rotation, scaling, dataset.is_6dof
            )
            pred_img = torch.clamp(render_out["render"], 0.0, 1.0)

        gt_img = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)

        save_image(pred_img.cpu(), renders_dir / f"pred_cam{idx}.png")
        save_image(gt_img.cpu(), gt_dir / f"gt_cam{idx}.png")

        predicted_seq = torch.cat([predicted_seq[1:], pred_flat.unsqueeze(0)], dim=0)

        if dataset.load2gpu_on_the_fly:
            cam.load2device('cpu')

    print(f"\nRendering complete! Results saved to {base_save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render temporal predictions")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    temporal = TemporalParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    bg_color = [0, 0, 0] if not model.extract(args).white_background else [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_temporal_predictions(
        model.extract(args), args.iteration, pipeline.extract(args),
        background, temporal.extract(args)
    )
