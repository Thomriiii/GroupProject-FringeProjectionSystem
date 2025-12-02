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


def _fallback_intrinsics(proj_pts_list):
    all_proj = np.vstack(proj_pts_list)
    cx = np.median(all_proj[:, 0])
    cy = np.median(all_proj[:, 1])
    span_x = all_proj[:, 0].max() - all_proj[:, 0].min()
    span_y = all_proj[:, 1].max() - all_proj[:, 1].min()
    f = max(span_x, span_y)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    print("[INTR] Warning: Zhang solve degenerate; using fallback guess from projector extents.")
    return K


def zhang_solve_intrinsics(proj_pts_list, cam_pts_list, max_pts: int = 8000):
    """
    Solve projector intrinsics Kp using homographies (Zhang method).
    proj_pts_list : list of Nx2 projector coords per pose
    cam_pts_list  : list of Nx2 camera coords per pose
    max_pts : cap correspondences per pose to keep RANSAC tractable
    """

    print("[INTR] Solving projector intrinsics using Zhang method...")

    V = []
    rng = np.random.default_rng(0)
    poses_used = 0

    for cam_pts, proj_pts in zip(cam_pts_list, proj_pts_list):
        if len(cam_pts) > max_pts:
            idx = rng.choice(len(cam_pts), size=max_pts, replace=False)
            cam_use = cam_pts[idx]
            proj_use = proj_pts[idx]
        else:
            cam_use = cam_pts
            proj_use = proj_pts

        H, _ = cv2.findHomography(proj_use, cam_use, method=cv2.RANSAC)
        if H is None:
            continue
        poses_used += 1

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

    if len(V) < 2:
        print("[INTR] Warning: insufficient valid poses for Zhang; need at least 1.")
        return _fallback_intrinsics(proj_pts_list)

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
    denom = (B[0, 0]*B[1, 1] - B[0, 1]**2)
    cy = (B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2]) / denom
    lam = B[0, 2] - (B[0, 1]*cy)
    if B[0, 0] <= 0 or denom <= 0 or lam <= 0:
        return _fallback_intrinsics(proj_pts_list)
    fx = np.sqrt(lam / B[0, 0])
    fy = np.sqrt((lam*B[0, 0]) / denom)
    cx = -B[0, 2] / B[0, 0]

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
        # Treat projector coordinates as lying on z=0 plane for PnP.
        obj = np.column_stack([proj, np.zeros(len(proj))]).astype(np.float32)
        objpoints.append(obj.reshape(-1, 1, 3))
        imgpoints.append(cam.reshape(-1, 1, 2).astype(np.float32))

    if not objpoints:
        raise RuntimeError("No projector/camera correspondences available for refinement.")

    # Get camera size
    if os.path.exists("camera_intrinsics.npz"):
        cam_data = np.load("camera_intrinsics.npz")
        W, H = cam_data["size"]  # [1280, 720]
    else:
        H, W = 720, 1280

    # Ensure K_init is finite; fall back to a simple guess if Zhang failed.
    if not np.isfinite(K_init).all():
        cx, cy = W * 0.5, H * 0.5
        f = max(W, H)
        print("[INTR] Warning: invalid Zhang init; falling back to centre guess.")
        K_init = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

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
