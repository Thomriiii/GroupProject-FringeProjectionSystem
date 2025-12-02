"""
projector_intrinsics.py

Solve projector intrinsics + camera-projector extrinsics
using a PSP-generated dataset.

Input:
    dataset:
        cam_points : list of Nx2 arrays  (camera pixels)
        proj_points: list of Nx2 arrays  (projector pixels)
        proj_w, proj_h

Output:
    Saves:
      projector_intrinsics_psp.npz
      stereo_params_psp.npz
"""

from __future__ import annotations
import os
import numpy as np
import cv2


def zhang_solve_intrinsics(proj_pts_list, cam_pts_list):
    """
    Solve projector intrinsics Kp using homographies (Zhang method).
    proj_pts_list : list of Nx2 projector coords per pose
    cam_pts_list  : list of Nx2 camera coords per pose
    """

    print("[INTR] Solving projector intrinsics using Zhang method...")

    V = []

    for cam_pts, proj_pts in zip(cam_pts_list, proj_pts_list):
        H, _ = cv2.findHomography(proj_pts, cam_pts, method=cv2.RANSAC)
        if H is None:
            continue

        h1 = H[:, 0]
        h2 = H[:, 1]

        v12 = np.array([
            h1[0]*h2[0],
            h1[0]*h2[1] + h1[1]*h2[0],
            h1[1]*h2[1],
            h1[2]*h2[0] + h1[0]*h2[2],
            h1[2]*h2[1] + h1[1]*h2[2],
            h1[2]*h2[2]
        ])

        v11 = np.array([
            h1[0]*h1[0],
            h1[0]*h1[1] + h1[1]*h1[0],
            h1[1]*h1[1],
            h1[2]*h1[0] + h1[0]*h1[2],
            h1[2]*h1[1] + h1[1]*h1[2],
            h1[2]*h1[2]
        ])

        v22 = np.array([
            h2[0]*h2[0],
            h2[0]*h2[1] + h2[1]*h2[0],
            h2[1]*h2[1],
            h2[2]*h2[0] + h2[0]*h2[2],
            h2[2]*h2[1] + h2[1]*h2[2],
            h2[2]*h2[2]
        ])

        V.append(v12)
        V.append(v11 - v22)

    V = np.vstack(V)

    # Solve Vb=0
    _, _, vh = np.linalg.svd(V)
    b = vh[-1]

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    # Recover intrinsics
    cy = (B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2]) / (B[0, 0]*B[1, 1] - B[0, 1]**2)
    lam = B[0, 2] - (B[0, 1]*cy)
    fx = np.sqrt(lam / B[0, 0])
    fy = np.sqrt((lam*B[0, 0]) / (B[0, 0]*B[1, 1] - B[0, 1]**2))
    cx = -B[0, 2] / B[0, 0]
    cy = cy

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0,  1]
    ])

    print("[INTR] Initial projector intrinsics:")
    print(K)

    return K


def refine_intrinsics(K_init, proj_pts_list, cam_pts_list):
    print("[INTR] Refining intrinsics via calibrateCamera()...")

    objpoints = []
    imgpoints = []

    for proj, cam in zip(proj_pts_list, cam_pts_list):
        objpoints.append(proj.reshape(-1, 1, 2).astype(np.float32))
        imgpoints.append(cam.reshape(-1, 1, 2).astype(np.float32))

    # Get camera size
    if os.path.exists("camera_intrinsics.npz"):
        cam_data = np.load("camera_intrinsics.npz")
        W, H = cam_data["size"]  # [1280, 720]
    else:
        H, W = 720, 1280

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (int(W), int(H)),
        K_init,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    print("[INTR] Refinement RMS:", ret)
    print("[INTR] Final K:")
    print(K)
    print("[INTR] Distortion:")
    print(dist)

    return K, dist, rvecs, tvecs, ret


def solve_stereo(Kp, proj_pts_list, cam_pts_list):
    """
    Solve camera–projector extrinsics using cv2.solvePnP().
    """

    print("[STEREO] Solving camera→projector pose...")

    # Use the largest pose
    i = np.argmax([pts.shape[0] for pts in proj_pts_list])

    proj_pts = proj_pts_list[i]
    cam_pts = cam_pts_list[i]

    # We treat projector coords as 3D points on z=0 plane
    obj = np.column_stack([proj_pts, np.zeros(len(proj_pts))]).astype(np.float32)
    img = cam_pts.astype(np.float32)

    # Solve Pose
    ret, rvec, tvec = cv2.solvePnP(obj, img, Kp, None)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    print("[STEREO] R:")
    print(R)
    print("[STEREO] t:")
    print(t)

    return R, t


def solve_from_dataset(dataset):
    cam_points = dataset["cam_points"]
    proj_points = dataset["proj_points"]

    K_init = zhang_solve_intrinsics(proj_points, cam_points)

    K_refined, dist, rvecs, tvecs, rms = refine_intrinsics(K_init, proj_points, cam_points)

    R, t = solve_stereo(K_refined, proj_points, cam_points)

    np.savez("projector_intrinsics_psp.npz", K=K_refined, dist=dist)
    np.savez("stereo_params_psp.npz", R=R, t=t)

    print("[INTR] Saved projector_intrinsics_psp.npz")
    print("[STEREO] Saved stereo_params_psp.npz")

    return {
        "K": K_refined,
        "dist": dist,
        "R": R,
        "t": t,
        "rms": rms
    }
