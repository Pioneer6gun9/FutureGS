import torch
import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors


def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
      w: (N, 3) A 3-vector

    Returns:
      W: (N, 3, 3) A skew matrix such that W @ v == w x v
    """
    zeros = torch.zeros(w.shape[0], device=w.device)
    w_skew_list = [zeros, -w[:, 2], w[:, 1],
                   w[:, 2], zeros, -w[:, 0],
                   -w[:, 1], w[:, 0], zeros]
    w_skew = torch.stack(w_skew_list, dim=-1).reshape(-1, 3, 3)
    return w_skew


def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.

    Args:
      R: (3, 3) An orthonormal rotation matrix.
      p: (3,) A 3-vector representing an offset.

    Returns:
      X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
    transform = torch.cat([torch.cat([R, p], dim=-1), bottom_row], dim=1)

    return transform


def exp_so3(w: torch.Tensor, theta: float) -> torch.Tensor:
    W = skew(w)
    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)  # batch matrix multiplication
    R = identity + torch.sin(theta.unsqueeze(-1)) * W + (1.0 - torch.cos(theta.unsqueeze(-1))) * W_sqr
    return R


def exp_se3(S: torch.Tensor, theta: float) -> torch.Tensor:
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)

    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)
    theta = theta.view(-1, 1, 1)

    p = torch.bmm((theta * identity + (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * W_sqr),
                  v.unsqueeze(-1))
    return rp_to_se3(R, p)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    return v[..., :3] / v[..., -1:]

def get_knn(pos: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Compute K nearest neighbors"""
    nn_model = NearestNeighbors(n_neighbors=k+1).fit(pos)
    distances, indices = nn_model.kneighbors(pos)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.int32)


def calculate_weights(values, max_k=10):
    if not values:
        return []
    min_val, max_val = min(values), max(values)
    normalized = [1 if max_val == min_val else
                  1 + (v - min_val) * (max_k - 1) / (max_val - min_val) for v in values]
    reciprocals = [1 / v for v in normalized]
    return [r / sum(reciprocals) for r in reciprocals]


def compute_rigid_loss(pred_d_xyz, knn_indices, knn_weights):
    knn_pos = pred_d_xyz[knn_indices]
    pos_diff = knn_pos - pred_d_xyz.unsqueeze(1)
    return (knn_weights * torch.norm(pos_diff, dim=-1)).mean()


def build_knn_graph(xyz, k_neighbors=20, theta_w=100000, device='cuda'):
    with torch.no_grad():
        base_pos = xyz.detach().cpu().numpy()
        knn_distances, knn_indices = get_knn(base_pos, k=k_neighbors)
        knn_indices = torch.from_numpy(knn_indices).long().to(device)
        knn_weights = torch.exp(-theta_w * torch.from_numpy(knn_distances).float().to(device)**2)
    return knn_indices, knn_weights
