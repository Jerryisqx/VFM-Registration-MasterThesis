

import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix


def pose_to_tmat(target_pose):
    """
    Convert 18D pose vector to 4x4 transformation matrices for femur and tibia.

    Args:
        target_pose (torch.Tensor): A tensor of shape (18,).

    Returns:
        tmat_femur (torch.Tensor): A 4x4 transformation matrix for the femur.
        tmat_tibia (torch.Tensor): A 4x4 transformation matrix for the tibia.
    """
    device = target_pose.device

    # Extract translation vectors
    femur_translation = target_pose[:3].view(3).to(device)
    tibia_translation = target_pose[9:12].view(3).to(device)

    # Extract rotation in 6D
    femur_rotation_6d = target_pose[3:9].view(1, 6).to(device)
    tibia_rotation_6d = target_pose[12:18].view(1, 6).to(device)

    # Convert 6D rotation representation to 3x3 rotation matrices
    femur_rotation_matrix = rotation_6d_to_matrix(femur_rotation_6d)
    tibia_rotation_matrix = rotation_6d_to_matrix(tibia_rotation_6d)

    # Create 4x4 transformation matrices using torch.stack to keep grad tracking
    tmat_femur = torch.stack([
        torch.cat([femur_rotation_matrix[0][0], femur_translation[0].view(1)]),
        torch.cat([femur_rotation_matrix[0][1], femur_translation[1].view(1)]),
        torch.cat([femur_rotation_matrix[0][2], femur_translation[2].view(1)]),
        torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
    ])

    tmat_tibia = torch.stack([
        torch.cat([tibia_rotation_matrix[0][0], tibia_translation[0].view(1)]),
        torch.cat([tibia_rotation_matrix[0][1], tibia_translation[1].view(1)]),
        torch.cat([tibia_rotation_matrix[0][2], tibia_translation[2].view(1)]),
        torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
    ])

    # Ensure matrices are of shape [1, 4, 4]
    tmat_femur = tmat_femur.unsqueeze(0)
    tmat_tibia = tmat_tibia.unsqueeze(0)

    return tmat_femur, tmat_tibia
