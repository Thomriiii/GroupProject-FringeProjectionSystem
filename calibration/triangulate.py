#!/usr/bin/env python3
"""
calibration/triangulate.py

Triangulation helper to compute 3D points from camera/projector pairs.
Uses closest-point between two rays.
"""

from __future__ import annotations

import numpy as np


def triangulate_points(K_cam, K_proj, R, t, cam_xy, proj_xy):
    """
    Triangulate 3D points from corresponding camera and projector pixels.

    Parameters
    ----------
    K_cam : (3x3) np.ndarray
        Camera intrinsic matrix.
    K_proj : (3x3) np.ndarray
        Projector intrinsic matrix.
    R : (3x3) np.ndarray
        Rotation from camera frame to projector frame.
    t : (3,) np.ndarray
        Translation from camera frame to projector frame.
    cam_xy : (N x 2) np.ndarray
        Camera pixel coordinates.
    proj_xy : (N x 2) np.ndarray
        Projector pixel coordinates.

    Returns
    -------
    pts_cam : (N x 3) np.ndarray
        Triangulated 3D points in camera coordinates.
    """
    Kc_inv = np.linalg.inv(K_cam)
    Kp_inv = np.linalg.inv(K_proj)

    N = cam_xy.shape[0]
    pts_out = np.zeros((N, 3), dtype=np.float64)

    # Projector origin in camera frame
    o1 = np.zeros(3)
    o2 = t.reshape(3)

    for i in range(N):
        u, v = cam_xy[i]
        x, y = proj_xy[i]

        d1 = Kc_inv @ np.array([u, v, 1.0])
        d1 = d1 / np.linalg.norm(d1)

        d2_proj = Kp_inv @ np.array([x, y, 1.0])
        d2 = R.T @ (d2_proj / np.linalg.norm(d2_proj))

        A = np.stack([d1, -d2], axis=1)
        b = o2 - o1

        # Solve for scalars s, t that minimize ||o1 + s*d1 - (o2 + t*d2)||
        ATA = A.T @ A
        ATb = A.T @ b
        sol = np.linalg.solve(ATA, ATb)
        s = sol[0]

        p_cam = o1 + s * d1
        pts_out[i] = p_cam

    return pts_out
