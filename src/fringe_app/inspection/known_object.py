"""Known-object defect detection pipeline.

This module is additive and runs after reconstruction. It does not modify
capture/phase/unwrap/UV/triangulation algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from fringe_app.recon.io import save_ply


@dataclass(slots=True)
class InspectionConfig:
    normal_knn: int = 30
    mesh_sample_points: int = 120000
    icp_max_iters: int = 30
    icp_tolerance: float = 1e-6
    icp_sample_size: int = 50000
    cluster_radius_mm: float = 1.5
    min_cluster_points: int = 25
    sample_seed: int = 0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "InspectionConfig":
        c = dict(cfg or {})
        return cls(
            normal_knn=int(c.get("normal_knn", 30)),
            mesh_sample_points=int(c.get("mesh_sample_points", 120000)),
            icp_max_iters=int(c.get("icp_max_iters", 30)),
            icp_tolerance=float(c.get("icp_tolerance", 1e-6)),
            icp_sample_size=int(c.get("icp_sample_size", 50000)),
            cluster_radius_mm=float(c.get("cluster_radius_mm", 1.5)),
            min_cluster_points=int(c.get("min_cluster_points", 25)),
            sample_seed=int(c.get("sample_seed", 0)),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    out = np.zeros_like(arr)
    valid = norms > 1e-12
    if np.any(valid):
        out[valid] = arr[valid] / norms[valid, None]
    return out


def _read_ascii_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("rb") as f:
        line = f.readline().decode("utf-8", errors="ignore").strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = None
        n_vertices = None
        in_vertex_element = False
        properties: list[tuple[str, str]] = []

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PLY header missing end_header: {path}")
            s = raw.decode("utf-8", errors="ignore").strip()
            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element "):
                parts = s.split()
                in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex_element:
                    n_vertices = int(parts[2])
            elif s.startswith("property ") and in_vertex_element:
                parts = s.split()
                if len(parts) >= 3 and parts[1] != "list":
                    properties.append((parts[1], parts[2]))
            elif s == "end_header":
                break

        if n_vertices is None or n_vertices <= 0:
            raise ValueError(f"PLY has no vertices: {path}")
        prop_names = [name for _, name in properties]
        if "x" not in prop_names or "y" not in prop_names or "z" not in prop_names:
            raise ValueError(f"PLY vertex properties missing x/y/z: {path}")

        i_x = prop_names.index("x")
        i_y = prop_names.index("y")
        i_z = prop_names.index("z")
        i_nx = prop_names.index("nx") if "nx" in prop_names else None
        i_ny = prop_names.index("ny") if "ny" in prop_names else None
        i_nz = prop_names.index("nz") if "nz" in prop_names else None
        has_normals = i_nx is not None and i_ny is not None and i_nz is not None

        points = np.zeros((n_vertices, 3), dtype=np.float64)
        normals = np.zeros((n_vertices, 3), dtype=np.float64) if has_normals else None

        if fmt == "ascii":
            for i in range(n_vertices):
                row = f.readline().decode("utf-8", errors="ignore")
                if not row:
                    raise ValueError(f"PLY ended early while reading vertices: {path}")
                parts = row.strip().split()
                if len(parts) < len(prop_names):
                    raise ValueError(f"Invalid PLY vertex row {i} in {path}")
                points[i, 0] = float(parts[i_x])
                points[i, 1] = float(parts[i_y])
                points[i, 2] = float(parts[i_z])
                if normals is not None:
                    normals[i, 0] = float(parts[i_nx])
                    normals[i, 1] = float(parts[i_ny])
                    normals[i, 2] = float(parts[i_nz])
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            ply_type_map = {
                "char": "i1",
                "uchar": "u1",
                "int8": "i1",
                "uint8": "u1",
                "short": "i2",
                "ushort": "u2",
                "int16": "i2",
                "uint16": "u2",
                "int": "i4",
                "uint": "u4",
                "int32": "i4",
                "uint32": "u4",
                "float": "f4",
                "float32": "f4",
                "double": "f8",
                "float64": "f8",
            }
            dtype_fields = []
            for t, name in properties:
                if t not in ply_type_map:
                    raise ValueError(f"Unsupported PLY property type '{t}' in {path}")
                dtype_fields.append((name, endian + ply_type_map[t]))
            dt = np.dtype(dtype_fields)
            arr = np.fromfile(f, dtype=dt, count=n_vertices)
            if arr.shape[0] != n_vertices:
                raise ValueError(f"PLY ended early while reading binary vertices: {path}")
            points[:, 0] = arr[prop_names[i_x]].astype(np.float64)
            points[:, 1] = arr[prop_names[i_y]].astype(np.float64)
            points[:, 2] = arr[prop_names[i_z]].astype(np.float64)
            if normals is not None:
                normals[:, 0] = arr[prop_names[i_nx]].astype(np.float64)
                normals[:, 1] = arr[prop_names[i_ny]].astype(np.float64)
                normals[:, 2] = arr[prop_names[i_nz]].astype(np.float64)
        else:
            raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")

    if normals is not None:
        normals = _normalize_rows(normals)
    return points.astype(np.float64), normals


def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("v "):
            parts = s.split()
            if len(parts) < 4:
                continue
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif s.startswith("f "):
            parts = s.split()[1:]
            idxs: list[int] = []
            for p in parts:
                token = p.split("/")[0]
                if not token:
                    continue
                vi = int(token)
                if vi < 0:
                    vi = len(verts) + vi + 1
                idxs.append(vi - 1)
            if len(idxs) >= 3:
                for k in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[k], idxs[k + 1]])

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError(f"OBJ contains no usable mesh geometry: {path}")

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if np.any(f < 0) or np.any(f >= v.shape[0]):
        raise ValueError(f"OBJ has invalid face indices: {path}")
    return v, f


def _load_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Invalid STL file: {path}")

    tri_count = int.from_bytes(data[80:84], byteorder="little", signed=False)
    expected = 84 + tri_count * 50

    triangles: list[np.ndarray] = []
    if expected == len(data):
        # Binary STL
        off = 84
        for _ in range(tri_count):
            # normal ignored
            off += 12
            v0 = np.frombuffer(data, dtype="<f4", count=3, offset=off).astype(np.float64)
            off += 12
            v1 = np.frombuffer(data, dtype="<f4", count=3, offset=off).astype(np.float64)
            off += 12
            v2 = np.frombuffer(data, dtype="<f4", count=3, offset=off).astype(np.float64)
            off += 12
            off += 2  # attribute byte count
            triangles.append(np.stack([v0, v1, v2], axis=0))
    else:
        # ASCII STL fallback
        lines = data.decode("utf-8", errors="ignore").splitlines()
        current: list[np.ndarray] = []
        for line in lines:
            s = line.strip().lower()
            if s.startswith("vertex "):
                parts = s.split()
                if len(parts) >= 4:
                    current.append(np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64))
                    if len(current) == 3:
                        triangles.append(np.stack(current, axis=0))
                        current = []

    if len(triangles) == 0:
        raise ValueError(f"STL contains no triangles: {path}")

    tri = np.asarray(triangles, dtype=np.float64)
    verts = tri.reshape(-1, 3)
    faces = np.arange(verts.shape[0], dtype=np.int64).reshape(-1, 3)
    return verts, faces


def _sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be [M,3]")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("Mesh is empty")

    tri = vertices[faces]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross, axis=1)
    valid = area2 > 1e-12
    if not np.any(valid):
        raise ValueError("Mesh triangles have zero area")

    tri = tri[valid]
    cross = cross[valid]
    area = 0.5 * area2[valid]
    prob = area / float(np.sum(area))

    rng = np.random.default_rng(int(seed))
    n = max(1000, int(n_samples))
    tri_idx = rng.choice(np.arange(tri.shape[0]), size=n, replace=True, p=prob)
    picked = tri[tri_idx]
    nrm = _normalize_rows(cross[tri_idx])

    r1 = rng.random(n)
    r2 = rng.random(n)
    sr1 = np.sqrt(r1)
    w0 = 1.0 - sr1
    w1 = sr1 * (1.0 - r2)
    w2 = sr1 * r2

    points = (
        w0[:, None] * picked[:, 0, :]
        + w1[:, None] * picked[:, 1, :]
        + w2[:, None] * picked[:, 2, :]
    )
    return points.astype(np.float64), nrm.astype(np.float64)


def _estimate_normals_open3d(points: np.ndarray, k: int) -> np.ndarray | None:
    try:
        import open3d as o3d
    except Exception:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(5, int(k))))
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape != points.shape:
        return None

    # Deterministic orientation: roughly away from centroid.
    center = np.mean(points, axis=0)
    outward = points - center[None, :]
    flip = np.sum(normals * outward, axis=1) < 0
    if np.any(flip):
        normals[flip] *= -1.0
    return _normalize_rows(normals)


def _estimate_normals_knn_pca(points: np.ndarray, k: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    n = int(pts.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 points to estimate normals")

    knn = max(5, min(int(k), n - 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=knn + 1)

    center = np.mean(pts, axis=0)
    normals = np.zeros_like(pts)
    for i in range(n):
        nbr = pts[idx[i, 1:], :]
        mu = np.mean(nbr, axis=0)
        q = nbr - mu[None, :]
        cov = q.T @ q
        _, vecs = np.linalg.eigh(cov)
        normal = vecs[:, 0]
        if float(np.dot(normal, pts[i, :] - center)) < 0.0:
            normal = -normal
        normals[i, :] = normal
    return _normalize_rows(normals)


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    normals = _estimate_normals_open3d(points, k=k)
    if normals is not None:
        return normals
    return _estimate_normals_knn_pca(points, k=k)


def load_reference_geometry(path: str | Path, sample_points: int = 120000, seed: int = 0) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Reference geometry not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".ply":
        points, normals = _read_ascii_ply_points(p)
        if normals is None:
            normals = estimate_normals(points, k=30)
        return {
            "source_type": "pointcloud",
            "path": str(p),
            "points": points.astype(np.float64),
            "normals": normals.astype(np.float64),
            "n_points": int(points.shape[0]),
        }

    if suffix == ".obj":
        vertices, faces = _load_obj_mesh(p)
    elif suffix == ".stl":
        vertices, faces = _load_stl_mesh(p)
    else:
        raise ValueError("Reference model must be .ply, .stl, or .obj")

    pts, nrm = _sample_mesh_surface(vertices, faces, n_samples=sample_points, seed=seed)
    return {
        "source_type": "mesh",
        "path": str(p),
        "points": pts.astype(np.float64),
        "normals": nrm.astype(np.float64),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "n_points": int(pts.shape[0]),
    }


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    r = transform[:3, :3]
    t = transform[:3, 3]
    return (r @ points.T).T + t[None, :]


def _rot_from_vec(omega: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation

        return Rotation.from_rotvec(omega).as_matrix()
    except Exception:
        th = float(np.linalg.norm(omega))
        if th < 1e-12:
            return np.eye(3, dtype=np.float64)
        k = omega / th
        kx = np.array(
            [
                [0.0, -k[2], k[1]],
                [k[2], 0.0, -k[0]],
                [-k[1], k[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _coarse_alignment(scan_points: np.ndarray, ref_points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    c_scan = np.mean(scan_points, axis=0)
    c_ref = np.mean(ref_points, axis=0)

    scan_scale = float(np.sqrt(np.mean(np.sum((scan_points - c_scan[None, :]) ** 2, axis=1))))
    ref_scale = float(np.sqrt(np.mean(np.sum((ref_points - c_ref[None, :]) ** 2, axis=1))))
    ratio = ref_scale / max(scan_scale, 1e-12)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = c_ref - c_scan
    stats = {
        "scan_scale": scan_scale,
        "ref_scale": ref_scale,
        "scale_ratio": ratio,
    }
    return transform, stats


def _icp_point_to_plane(
    scan_points: np.ndarray,
    ref_points: np.ndarray,
    ref_normals: np.ndarray,
    init_transform: np.ndarray,
    max_iters: int,
    tolerance: float,
    sample_size: int,
) -> tuple[np.ndarray, float, int]:
    if scan_points.shape[0] == 0 or ref_points.shape[0] == 0:
        raise ValueError("ICP requires non-empty point sets")

    tree = cKDTree(ref_points)
    tform = np.asarray(init_transform, dtype=np.float64).copy()
    n_scan = int(scan_points.shape[0])

    if n_scan > int(sample_size) > 0:
        # Deterministic subset
        sample_idx = np.linspace(0, n_scan - 1, int(sample_size), dtype=np.int64)
    else:
        sample_idx = np.arange(n_scan, dtype=np.int64)

    prev_rmse = None
    rmse = float("inf")
    iters_done = 0

    for it in range(max(1, int(max_iters))):
        iters_done = it + 1
        pts_all = _transform_points(scan_points, tform)
        pts = pts_all[sample_idx, :]

        dists, idx = tree.query(pts, k=1)
        valid = np.isfinite(dists)
        if int(np.count_nonzero(valid)) < 6:
            break

        p = pts[valid, :]
        q = ref_points[idx[valid], :]
        n = ref_normals[idx[valid], :]
        finite_n = np.isfinite(n).all(axis=1)
        p = p[finite_n, :]
        q = q[finite_n, :]
        n = n[finite_n, :]
        if p.shape[0] < 6:
            break

        residual = np.sum((p - q) * n, axis=1)
        a_rot = np.cross(p, n)
        a = np.hstack([a_rot, n])

        x, *_ = np.linalg.lstsq(a, -residual, rcond=None)
        omega = x[:3]
        trans = x[3:]

        r_delta = _rot_from_vec(omega)
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = r_delta
        delta[:3, 3] = trans
        tform = delta @ tform

        rmse = float(np.sqrt(np.mean(np.square(residual))))
        step_norm = float(np.linalg.norm(x))

        if prev_rmse is not None:
            improve = abs(prev_rmse - rmse)
            if improve < float(tolerance) and step_norm < 1e-7:
                break
        prev_rmse = rmse

    return tform, float(rmse), int(iters_done)


def _diverging_colors(values_mm: np.ndarray, clip_mm: float) -> np.ndarray:
    v = np.asarray(values_mm, dtype=np.float64)
    c = max(float(clip_mm), 1e-6)
    t = np.clip(v / c, -1.0, 1.0)

    out = np.zeros((v.shape[0], 3), dtype=np.float32)
    pos = t >= 0.0
    neg = ~pos

    # Positive (bump): white -> red
    out[pos, 0] = 255.0
    out[pos, 1] = 255.0 * (1.0 - t[pos])
    out[pos, 2] = 255.0 * (1.0 - t[pos])

    # Negative (dent): white -> blue
    out[neg, 0] = 255.0 * (1.0 + t[neg])
    out[neg, 1] = 255.0 * (1.0 + t[neg])
    out[neg, 2] = 255.0
    return out


def _overlay_colors(values_mm: np.ndarray, tolerance_mm: float) -> np.ndarray:
    v = np.asarray(values_mm, dtype=np.float64)
    tol = max(float(tolerance_mm), 1e-9)
    ok = np.abs(v) <= tol
    out = np.zeros((v.shape[0], 3), dtype=np.float32)
    out[ok, :] = np.array([40.0, 185.0, 60.0], dtype=np.float32)
    out[~ok, :] = np.array([230.0, 45.0, 40.0], dtype=np.float32)
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        ra = int(self.rank[pa])
        rb = int(self.rank[pb])
        if ra < rb:
            self.parent[pa] = pb
        elif ra > rb:
            self.parent[pb] = pa
        else:
            self.parent[pb] = pa
            self.rank[pa] = ra + 1


def _estimate_spacing(points: np.ndarray) -> float:
    n = int(points.shape[0])
    if n < 3:
        return 0.0
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2)
    nn = d[:, 1]
    valid = np.isfinite(nn) & (nn > 0)
    if not np.any(valid):
        return 0.0
    return float(np.median(nn[valid]))


def _segment_defects(
    aligned_points: np.ndarray,
    deviation_mm: np.ndarray,
    tolerance_mm: float,
    cluster_radius_mm: float,
    min_cluster_points: int,
) -> list[dict[str, Any]]:
    defect_mask = np.abs(deviation_mm) > float(tolerance_mm)
    idx_all = np.where(defect_mask)[0]
    if idx_all.size == 0:
        return []

    defect_points = aligned_points[idx_all, :]
    n = int(defect_points.shape[0])
    if n == 0:
        return []

    radius_m = max(float(cluster_radius_mm), 1e-6) / 1000.0
    tree = cKDTree(defect_points)
    pairs = tree.query_pairs(radius_m, output_type="ndarray")

    uf = _UnionFind(n)
    if pairs.size > 0:
        for row in pairs:
            uf.union(int(row[0]), int(row[1]))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    spacing_m = _estimate_spacing(aligned_points)
    spacing_mm = spacing_m * 1000.0

    clusters: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < int(min_cluster_points):
            continue
        local_idx = np.asarray(members, dtype=np.int64)
        global_idx = idx_all[local_idx]
        pts = aligned_points[global_idx, :]
        dev = deviation_mm[global_idx]

        max_abs = float(np.max(np.abs(dev)))
        mean_dev = float(np.mean(dev))
        centroid_m = np.mean(pts, axis=0)
        bbox_min = np.min(pts, axis=0)
        bbox_max = np.max(pts, axis=0)

        area_mm2 = float(len(members) * (spacing_mm ** 2)) if spacing_mm > 0 else 0.0
        if max_abs >= 3.0 * float(tolerance_mm):
            severity = "high"
        elif max_abs >= 2.0 * float(tolerance_mm):
            severity = "medium"
        else:
            severity = "low"

        clusters.append(
            {
                "point_indices": global_idx.tolist(),
                "count": int(len(members)),
                "area_mm2": area_mm2,
                "max_deviation_mm": max_abs,
                "mean_deviation_mm": mean_dev,
                "centroid_m": [float(v) for v in centroid_m],
                "centroid_mm": [float(v * 1000.0) for v in centroid_m],
                "bounding_box_m": {
                    "min": [float(v) for v in bbox_min],
                    "max": [float(v) for v in bbox_max],
                },
                "severity": severity,
            }
        )

    clusters.sort(key=lambda d: (-float(d.get("max_deviation_mm", 0.0)), -int(d.get("count", 0))))
    for i, c in enumerate(clusters, start=1):
        c["id"] = int(i)
    return clusters


def _load_scan_points(run_dir: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    recon_dir = run_dir / "reconstruction"
    xyz_path = recon_dir / "xyz.npy"
    mask_path = recon_dir / "masks" / "mask_recon.npy"
    pcd_path = recon_dir / "pointcloud.ply"

    if xyz_path.exists() and mask_path.exists():
        xyz = np.asarray(np.load(xyz_path), dtype=np.float64)
        mask = np.asarray(np.load(mask_path), dtype=bool)
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError(f"Invalid xyz.npy shape: {xyz.shape}")
        if mask.shape != xyz.shape[:2]:
            raise ValueError(f"mask_recon shape {mask.shape} does not match xyz {xyz.shape}")
        finite = np.isfinite(xyz[:, :, 0]) & np.isfinite(xyz[:, :, 1]) & np.isfinite(xyz[:, :, 2])
        valid = mask & finite
        ys, xs = np.where(valid)
        if ys.size == 0:
            raise ValueError("No valid 3D points in reconstruction masks")
        points = xyz[ys, xs, :]
        return points.astype(np.float64), valid.astype(bool), {
            "source": "reconstruction/xyz.npy + masks/mask_recon.npy",
            "point_count": int(points.shape[0]),
            "grid_shape": [int(xyz.shape[0]), int(xyz.shape[1])],
        }

    if pcd_path.exists():
        points, _ = _read_ascii_ply_points(pcd_path)
        if points.shape[0] == 0:
            raise ValueError("reconstruction pointcloud.ply has zero points")
        return points.astype(np.float64), None, {
            "source": "reconstruction/pointcloud.ply",
            "point_count": int(points.shape[0]),
            "grid_shape": None,
        }

    raise FileNotFoundError(
        f"Reconstruction outputs not found for run '{run_dir.name}'. "
        "Expected reconstruction/xyz.npy + masks/mask_recon.npy or reconstruction/pointcloud.ply"
    )


def _load_or_compute_normals(
    inspection_dir: Path,
    recon_dir: Path,
    points: np.ndarray,
    cfg: InspectionConfig,
    recompute: bool,
) -> tuple[np.ndarray, str]:
    candidates = [inspection_dir / "normals.npy", recon_dir / "normals.npy"]
    if not recompute:
        for p in candidates:
            if not p.exists():
                continue
            arr = np.asarray(np.load(p), dtype=np.float64)
            if arr.shape == points.shape:
                return _normalize_rows(arr), str(p)

    normals = estimate_normals(points, k=cfg.normal_knn)
    out_path = inspection_dir / "normals.npy"
    np.save(out_path, normals.astype(np.float32))
    return normals, str(out_path)


def _compute_deviation(
    aligned_scan_points: np.ndarray,
    ref_points: np.ndarray,
    ref_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(ref_points)
    dists, idx = tree.query(aligned_scan_points, k=1)
    nearest = ref_points[idx, :]
    nearest_normals = ref_normals[idx, :]

    signed_m = np.sum((aligned_scan_points - nearest) * nearest_normals, axis=1)
    signed_mm = signed_m * 1000.0
    return signed_mm.astype(np.float64), dists.astype(np.float64), idx.astype(np.int64)


def run_known_object_inspection(
    *,
    run_root: Path,
    run_id: str,
    reference_model: str | Path,
    tolerance_mm: float,
    cfg: dict[str, Any] | None = None,
    recompute: bool = False,
) -> dict[str, Any]:
    config = InspectionConfig.from_dict(cfg)
    run_dir = Path(run_root) / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    inspection_dir = run_dir / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = run_dir / "reconstruction"

    reference_abs = str(Path(reference_model).expanduser().resolve())
    tolerance_mm = max(_safe_float(tolerance_mm, 0.5), 0.0)

    meta_path = inspection_dir / "inspection_meta.json"
    existing_meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text())
        except Exception:
            existing_meta = {}

    same_ref = str(existing_meta.get("reference_model", "")) == reference_abs
    same_tol = abs(_safe_float(existing_meta.get("tolerance_mm", -1.0), -1.0) - tolerance_mm) < 1e-9

    outputs = {
        "registration_transform": inspection_dir / "registration_transform.json",
        "deviation_map": inspection_dir / "deviation_map.npy",
        "deviation_colored": inspection_dir / "deviation_map_colored.ply",
        "defects": inspection_dir / "defects.json",
        "defect_report": inspection_dir / "defect_report.json",
        "defect_overlay": inspection_dir / "defect_overlay.ply",
    }

    scan_points, confidence_mask, scan_info = _load_scan_points(run_dir)
    normals, normals_source = _load_or_compute_normals(
        inspection_dir=inspection_dir,
        recon_dir=recon_dir,
        points=scan_points,
        cfg=config,
        recompute=bool(recompute),
    )

    ref = load_reference_geometry(
        reference_abs,
        sample_points=config.mesh_sample_points,
        seed=config.sample_seed,
    )
    ref_points = np.asarray(ref["points"], dtype=np.float64)
    ref_normals = np.asarray(ref["normals"], dtype=np.float64)

    need_registration = bool(recompute) or (not same_ref) or (not outputs["registration_transform"].exists())

    if need_registration:
        init_t, coarse_stats = _coarse_alignment(scan_points, ref_points)
        tform, rmse, iters = _icp_point_to_plane(
            scan_points=scan_points,
            ref_points=ref_points,
            ref_normals=ref_normals,
            init_transform=init_t,
            max_iters=config.icp_max_iters,
            tolerance=config.icp_tolerance,
            sample_size=config.icp_sample_size,
        )
        reg_payload = {
            "run_id": str(run_id),
            "reference_model": reference_abs,
            "coarse": coarse_stats,
            "icp": {
                "rmse_m": float(rmse),
                "rmse_mm": float(rmse * 1000.0),
                "iterations": int(iters),
                "max_iters": int(config.icp_max_iters),
                "sample_size": int(config.icp_sample_size),
                "tolerance": float(config.icp_tolerance),
            },
            "transform": tform.tolist(),
            "created_at": _now_iso(),
        }
        outputs["registration_transform"].write_text(json.dumps(_json_safe(reg_payload), indent=2))
    else:
        reg_payload = json.loads(outputs["registration_transform"].read_text())
        tform = np.asarray(reg_payload.get("transform"), dtype=np.float64)
        if tform.shape != (4, 4):
            raise ValueError("registration_transform.json has invalid transform matrix")

    aligned_points = _transform_points(scan_points, tform)

    need_deviation = bool(recompute) or (not same_ref) or (not outputs["deviation_map"].exists())
    if need_deviation:
        deviation_mm, nn_dists_m, nn_idx = _compute_deviation(
            aligned_scan_points=aligned_points,
            ref_points=ref_points,
            ref_normals=ref_normals,
        )
        np.save(outputs["deviation_map"], deviation_mm.astype(np.float32))
        nn_path = inspection_dir / "nearest_ref_index.npy"
        np.save(nn_path, nn_idx.astype(np.int64))
        np.save(inspection_dir / "nearest_ref_distance_m.npy", nn_dists_m.astype(np.float32))
    else:
        deviation_mm = np.asarray(np.load(outputs["deviation_map"]), dtype=np.float64)
        if deviation_mm.ndim != 1 or deviation_mm.shape[0] != aligned_points.shape[0]:
            raise ValueError("deviation_map.npy shape does not match scan point count")

    need_colored = bool(recompute) or need_deviation or (not outputs["deviation_colored"].exists())
    if need_colored:
        col = _diverging_colors(deviation_mm, clip_mm=max(3.0 * tolerance_mm, 0.1))
        save_ply(aligned_points.astype(np.float32), col.astype(np.float32), outputs["deviation_colored"])

    need_defects = (
        bool(recompute)
        or need_deviation
        or (not same_ref)
        or (not same_tol)
        or (not outputs["defects"].exists())
        or (not outputs["defect_report"].exists())
        or (not outputs["defect_overlay"].exists())
    )

    if need_defects:
        clusters = _segment_defects(
            aligned_points=aligned_points,
            deviation_mm=deviation_mm,
            tolerance_mm=tolerance_mm,
            cluster_radius_mm=config.cluster_radius_mm,
            min_cluster_points=config.min_cluster_points,
        )
        outputs["defects"].write_text(json.dumps(_json_safe(clusters), indent=2))

        overlay_col = _overlay_colors(deviation_mm, tolerance_mm=tolerance_mm)
        save_ply(aligned_points.astype(np.float32), overlay_col.astype(np.float32), outputs["defect_overlay"])

        defects_report = []
        for c in clusters:
            defects_report.append(
                {
                    "id": int(c["id"]),
                    "centroid": c.get("centroid_mm"),
                    "max_deviation": float(c.get("max_deviation_mm", 0.0)),
                    "area": float(c.get("area_mm2", 0.0)),
                    "severity": str(c.get("severity", "low")),
                }
            )

        report = {
            "scan_id": str(run_id),
            "reference_model": reference_abs,
            "tolerance_mm": float(tolerance_mm),
            "defects_detected": int(len(defects_report)),
            "pass": bool(len(defects_report) == 0),
            "defects": defects_report,
            "stats": {
                "deviation_mm_mean_abs": float(np.mean(np.abs(deviation_mm))) if deviation_mm.size > 0 else 0.0,
                "deviation_mm_p95_abs": float(np.percentile(np.abs(deviation_mm), 95.0)) if deviation_mm.size > 0 else 0.0,
                "deviation_mm_max_abs": float(np.max(np.abs(deviation_mm))) if deviation_mm.size > 0 else 0.0,
                "point_count": int(aligned_points.shape[0]),
            },
            "generated_at": _now_iso(),
        }
        outputs["defect_report"].write_text(json.dumps(_json_safe(report), indent=2))
    else:
        report = json.loads(outputs["defect_report"].read_text())
        try:
            clusters = json.loads(outputs["defects"].read_text())
        except Exception:
            clusters = []

    meta = {
        "scan_id": str(run_id),
        "reference_model": reference_abs,
        "tolerance_mm": float(tolerance_mm),
        "recompute": bool(recompute),
        "config": {
            "normal_knn": int(config.normal_knn),
            "mesh_sample_points": int(config.mesh_sample_points),
            "icp_max_iters": int(config.icp_max_iters),
            "icp_tolerance": float(config.icp_tolerance),
            "icp_sample_size": int(config.icp_sample_size),
            "cluster_radius_mm": float(config.cluster_radius_mm),
            "min_cluster_points": int(config.min_cluster_points),
            "sample_seed": int(config.sample_seed),
        },
        "scan": scan_info,
        "reference": {
            "source_type": ref.get("source_type"),
            "path": ref.get("path"),
            "n_points": int(ref_points.shape[0]),
            "mesh_vertices": ref.get("mesh_vertices"),
            "mesh_faces": ref.get("mesh_faces"),
        },
        "registration": {
            "path": str(outputs["registration_transform"]),
            "rmse_mm": _safe_float((reg_payload.get("icp", {}) or {}).get("rmse_mm"), 0.0),
        },
        "reused": {
            "registration": bool((not need_registration) and outputs["registration_transform"].exists()),
            "deviation_map": bool((not need_deviation) and outputs["deviation_map"].exists()),
            "defect_outputs": bool((not need_defects) and outputs["defect_report"].exists()),
            "normals": bool(Path(normals_source).exists()),
        },
        "files": {k: str(v) for k, v in outputs.items()},
        "normals_path": str(normals_source),
        "generated_at": _now_iso(),
    }
    meta_path.write_text(json.dumps(_json_safe(meta), indent=2))

    report = json.loads(outputs["defect_report"].read_text())

    return {
        "run_id": str(run_id),
        "inspection_dir": str(inspection_dir),
        "reference_model": reference_abs,
        "tolerance_mm": float(tolerance_mm),
        "pass": bool(report.get("pass", False)),
        "defects_detected": int(report.get("defects_detected", 0)),
        "defect_report": str(outputs["defect_report"]),
        "overlay_visualization": str(outputs["defect_overlay"]),
        "deviation_visualization": str(outputs["deviation_colored"]),
        "reused": dict(meta.get("reused", {})),
        "report": report,
        "files": {k: str(v) for k, v in outputs.items()},
    }
