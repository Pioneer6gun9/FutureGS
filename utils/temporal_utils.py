import torch
from typing import List, Tuple
from utils.rigid_utils import calculate_weights, compute_rigid_loss


def generate_deformation_sequence(deform, xyz, time_steps, device='cuda'):
    N = xyz.shape[0]
    sequence = []

    for t in time_steps:
        t_tensor = torch.tensor([t], device=device)
        time_input = t_tensor.unsqueeze(0).expand(N, -1)

        with torch.no_grad():
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

        # Flatten and concatenate
        d_xyz_flat = d_xyz.view(-1)
        d_rotation_flat = d_rotation.view(-1)
        d_scaling_flat = d_scaling.view(-1)
        combined = torch.cat([d_xyz_flat, d_rotation_flat, d_scaling_flat], dim=-1)

        sequence.append(combined.unsqueeze(0))

    sequence = torch.cat(sequence, dim=0)  # [T, flatten_dim]
    return sequence


def predict_with_sliding_window(temporal_model, sequence, seq_len, output_frame, flatten_dim, knn_indices=None, knn_weights=None, N=None):
    pred_flats = []
    knn_losses = []

    temporal_model.eval_mode()
    with torch.no_grad():
        for i in range(output_frame):
            # Sliding window prediction
            input_seq = sequence[i:i+seq_len].unsqueeze(0)
            pred_result = temporal_model.step(input_seq)
            pred_flats.append(pred_result.squeeze(0))

            # Calculate rigidity constraint loss
            if knn_indices is not None and knn_weights is not None and N is not None:
                pred_d_xyz = pred_result[:, :N*3].view(N, 3)
                rigid_loss = compute_rigid_loss(pred_d_xyz, knn_indices, knn_weights)
                knn_losses.append(rigid_loss.item())

    if knn_losses:
        knn_weight = calculate_weights(knn_losses)
        pred_flats = [pred_flats[i] * knn_weight[i] for i in range(output_frame)]
        pred_flat = torch.stack(pred_flats, dim=0).sum(dim=0)
    else:
        pred_flat = torch.stack(pred_flats, dim=0).mean(dim=0)

    return pred_flat


def parse_deformation(pred_flat, N):
    d_xyz = pred_flat[:N*3].view(N, 3)
    rotation = pred_flat[N*3:N*3+N*4].view(N, 4)
    scaling = pred_flat[N*3+N*4:].view(N, 3)

    return d_xyz, rotation, scaling


def compute_temporal_loss(pred, target, N, loss_weights=None):
    if loss_weights is None:
        loss_weights = {'xyz': 0.0, 'rot': 0.0, 'scale': 0.0}

    loss_fn = torch.nn.MSELoss()

    # Decompose prediction and target
    d_xyz_pred = pred[:, :N*3]
    d_xyz_target = target[:, :N*3]

    rot_pred = pred[:, N*3:N*(3+4)]
    rot_target = target[:, N*3:N*(3+4)]

    scale_pred = pred[:, N*(3+4):]
    scale_target = target[:, N*(3+4):]

    # Calculate component-wise losses
    base_loss = loss_fn(pred, target)
    xyz_loss = torch.nn.functional.mse_loss(d_xyz_pred, d_xyz_target)
    rot_loss = torch.nn.functional.mse_loss(rot_pred, rot_target)
    scale_loss = torch.nn.functional.mse_loss(scale_pred, scale_target)

    # Weighted sum
    total_loss = (base_loss +
                 loss_weights['xyz'] * xyz_loss +
                 loss_weights['rot'] * rot_loss +
                 loss_weights['scale'] * scale_loss)

    return total_loss
