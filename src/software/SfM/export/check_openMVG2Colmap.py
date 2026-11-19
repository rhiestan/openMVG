#!/usr/bin/env python
"""Check openMVG_main_openMVG2Colmap on a synthetic scene.

Builds an sfm_data with one camera per supported OpenMVG model (plus an
unsupported spherical one and an unposed view), converts it to a COLMAP model
(text and binary) and checks that
  - the text and binary models are identical,
  - every observation reprojected with COLMAP's camera formulas, in COLMAP
    pixel coordinates, gives the same residuals as OpenMVG's formulas on the
    sfm_data (RMS equal to ~1e-9 px),
  - the per-point ERROR equals OpenMVG's mean reprojection error,
  - unsupported and unposed views are left out, tracks reference valid 2D
    points, colors are sampled from the images (resolved relative to the
    sfm_data file), and missing images give gray points instead of a failure.

Usage: check_openMVG2Colmap.py <bin dir with openMVG_main_openMVG2Colmap and
                                 openMVG_main_ConvertSfM_DataFormat> <work dir>
"""

import json
import math
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

RNG = np.random.default_rng(42)
W, H = 640, 480


# --- OpenMVG camera models (normalized point -> distorted normalized point) --

def omvg_disto(model, p, x):
    r2 = x[0] ** 2 + x[1] ** 2
    if model == "pinhole":
        return x
    if model == "pinhole_radial_k1":
        return x * (1 + p[0] * r2)
    if model == "pinhole_radial_k3":
        return x * (1 + p[0] * r2 + p[1] * r2 ** 2 + p[2] * r2 ** 3)
    if model == "pinhole_brown_t2":
        k1, k2, k3, t1, t2 = p
        k = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        tx = t2 * (r2 + 2 * x[0] ** 2) + 2 * t1 * x[0] * x[1]
        ty = t1 * (r2 + 2 * x[1] ** 2) + 2 * t2 * x[0] * x[1]
        return x + np.array([x[0] * k + tx, x[1] * k + ty])
    if model == "fisheye":
        r = math.hypot(x[0], x[1])
        th = math.atan(r)
        thd = th + p[0] * th ** 3 + p[1] * th ** 5 + p[2] * th ** 7 + p[3] * th ** 9
        return x * (thd / r if r > 1e-8 else 1.0)
    raise ValueError(model)


def omvg_project(cam, R, C, X):
    Xc = R @ (X - C)
    x = omvg_disto(cam["model"], cam["disto"], Xc[:2] / Xc[2])
    return cam["f"] * x + np.array(cam["pp"])


# --- COLMAP camera models (colmap/sensor/models/*.h) ------------------------

def colmap_project(model, params, Xc):
    u, v = Xc[0] / Xc[2], Xc[1] / Xc[2]
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return np.array([fx * u + cx, fy * v + cy])
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        r2 = u * u + v * v
        du, dv = u * k * r2, v * k * r2
        return np.array([f * (u + du) + cx, f * (v + dv) + cy])
    if model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params
        u2, uv, v2 = u * u, u * v, v * v
        r2 = u2 + v2
        r4, r6 = r2 * r2, r2 * r2 * r2
        radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        return np.array([fx * (u + du) + cx, fy * (v + dv) + cy])
    if model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        uu, vv = u, v
        r = math.sqrt(u * u + v * v)
        if r > np.finfo(float).eps:
            th = math.atan(r)
            uu, vv = u * th / r, v * th / r
        t2 = uu * uu + vv * vv
        radial = k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4
        return np.array([fx * (uu + uu * radial) + cx, fy * (vv + vv * radial) + cy])
    raise ValueError(model)


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = (math.cos(ax), math.sin(ax), math.cos(ay),
                              math.sin(ay), math.cos(az), math.sin(az))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --- Synthetic scene --------------------------------------------------------

CAMERAS = [
    dict(model="pinhole", f=600.0, pp=[319.1217, 241.7], disto=[]),
    dict(model="pinhole_radial_k1", f=580.0, pp=[322.3, 236.9], disto=[-0.12]),
    dict(model="pinhole_radial_k3", f=610.0, pp=[319.1217, 239.4], disto=[-0.21, 0.13, -0.04]),
    dict(model="pinhole_brown_t2", f=590.0, pp=[317.6, 243.2],
         disto=[-0.18, 0.09, -0.02, 0.0031, -0.0017]),
    dict(model="fisheye", f=300.0, pp=[321.4, 238.8], disto=[0.02, -0.011, 0.004, -0.0012]),
    dict(model="spherical"),
]


def make_scene():
    points = RNG.uniform([-2, -1.5, -1], [2, 1.5, 1], size=(300, 3))
    views, poses = [], {}
    view_id = 0
    for cam_id, cam in enumerate(CAMERAS):
        for k in range(2):
            C = np.array([-0.6 + 1.2 * k + 0.1 * cam_id, 0.05 * cam_id, -6.0])
            R = rot(0.03 * cam_id, -0.1 + 0.2 * k, 0.02 * k)
            views.append(dict(id=view_id, cam=cam_id, name="imgs/view_%02d.ppm" % view_id))
            poses[view_id] = (R, C)
            view_id += 1
    # a view without a pose
    views.append(dict(id=view_id, cam=0, name="imgs/view_%02d.ppm" % view_id))
    structure = []
    for pid, X in enumerate(points):
        obs = {}
        for v in views:
            if v["id"] not in poses or CAMERAS[v["cam"]]["model"] == "spherical":
                continue
            R, C = poses[v["id"]]
            x = omvg_project(CAMERAS[v["cam"]], R, C, X) + RNG.normal(0, 0.3, 2)
            if 0 <= x[0] < W - 1 and 0 <= x[1] < H - 1:
                obs[v["id"]] = (pid * 10 + v["id"], x)
        # an observation in the spherical view (must be dropped)
        if pid % 7 == 0:
            obs[10] = (pid, np.array([100.0, 100.0]))
        if len(obs) >= 2:
            structure.append((pid + 1000, X, obs))
    return views, poses, structure


def sfm_data_json(views, poses, structure):
    ptr_id = [2147483649]
    type_ids = {}

    def new_ptr():
        ptr_id[0] += 1
        return ptr_id[0] - 1

    j_views = []
    for v in views:
        j_views.append({"key": v["id"], "value": {
            "polymorphic_id": 1073741824,
            "ptr_wrapper": {"id": new_ptr(), "data": {
                "local_path": "", "filename": v["name"], "width": W, "height": H,
                "id_view": v["id"], "id_intrinsic": v["cam"],
                "id_pose": v["id"] if v["id"] in poses else 4294967295}}}})
    j_intr = []
    for cam_id, cam in enumerate(CAMERAS):
        data = {"width": W, "height": H}
        if cam["model"] == "spherical":
            data = {"value0": data}  # cereal::base_class<IntrinsicBase>
        else:
            data.update(focal_length=cam["f"], principal_point=cam["pp"])
            key = {"pinhole_radial_k1": "disto_k1", "pinhole_radial_k3": "disto_k3",
                   "pinhole_brown_t2": "disto_t2", "fisheye": "fisheye"}.get(cam["model"])
            if key:
                data[key] = cam["disto"]
        value = {}
        if cam["model"] in type_ids:
            value["polymorphic_id"] = type_ids[cam["model"]]
        else:
            type_ids[cam["model"]] = 2147483649 + len(type_ids)
            value["polymorphic_id"] = type_ids[cam["model"]]
            value["polymorphic_name"] = cam["model"]
        value["ptr_wrapper"] = {"id": new_ptr(), "data": data}
        j_intr.append({"key": cam_id, "value": value})
    j_ext = [{"key": k, "value": {"rotation": R.tolist(), "center": C.tolist()}}
             for k, (R, C) in poses.items()]
    j_struct = [{"key": pid, "value": {"X": X.tolist(), "observations": [
        {"key": vid, "value": {"id_feat": feat, "x": x.tolist()}}
        for vid, (feat, x) in sorted(obs.items())]}} for pid, X, obs in structure]
    return {"sfm_data_version": "0.3", "root_path": ".", "views": j_views,
            "intrinsics": j_intr, "extrinsics": j_ext, "structure": j_struct,
            "control_points": []}


def image_color(view_id, x, y):
    return ((x * 3) % 256, (y * 5) % 256, (view_id * 37 + 11) % 256)


def write_images(folder, views):
    os.makedirs(folder, exist_ok=True)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    for v in views:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        r, g, b = image_color(v["id"], xs, ys)
        img[..., 0], img[..., 1], img[..., 2] = r, g, b
        with open(os.path.join(folder, os.path.basename(v["name"])), "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (W, H))
            f.write(img.tobytes())


# --- COLMAP readers ---------------------------------------------------------

NUM_PARAMS = {"PINHOLE": 4, "SIMPLE_RADIAL": 4, "FULL_OPENCV": 12, "OPENCV_FISHEYE": 8}
MODEL_IDS = {1: "PINHOLE", 2: "SIMPLE_RADIAL", 6: "FULL_OPENCV", 5: "OPENCV_FISHEYE"}


def data_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if not l.startswith("#")]


def read_text(d):
    cams, imgs, pts = {}, {}, {}
    for l in data_lines(os.path.join(d, "cameras.txt")):
        e = l.split()
        cams[int(e[0])] = (e[1], int(e[2]), int(e[3]), [float(v) for v in e[4:]])
    lines = data_lines(os.path.join(d, "images.txt"))
    for a, b in zip(lines[0::2], lines[1::2]):
        e = a.split()
        e2 = b.split()
        p2d = [(float(e2[i]), float(e2[i + 1]), int(e2[i + 2])) for i in range(0, len(e2), 3)]
        imgs[int(e[0])] = ([float(v) for v in e[1:8]], int(e[8]), e[9], p2d)
    for l in data_lines(os.path.join(d, "points3D.txt")):
        e = l.split()
        track = [(int(e[i]), int(e[i + 1])) for i in range(8, len(e), 2)]
        pts[int(e[0])] = ([float(v) for v in e[1:4]], tuple(int(v) for v in e[4:7]),
                          float(e[7]), track)
    return cams, imgs, pts


class Reader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.b = f.read()
        self.o = 0

    def get(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def done(self):
        return self.o == len(self.b)


def read_binary(d):
    cams, imgs, pts = {}, {}, {}
    r = Reader(os.path.join(d, "cameras.bin"))
    for _ in range(r.get("Q")):
        cid, mid, w, h = r.get("I"), r.get("i"), r.get("Q"), r.get("Q")
        model = MODEL_IDS[mid]
        cams[cid] = (model, w, h, list(r.get("d" * NUM_PARAMS[model])))
    assert r.done()
    r = Reader(os.path.join(d, "images.bin"))
    for _ in range(r.get("Q")):
        iid = r.get("I")
        qt = list(r.get("7d"))
        cid = r.get("I")
        end = r.b.index(b"\0", r.o)
        name = r.b[r.o:end].decode("utf-8")
        r.o = end + 1
        p2d = [r.get("ddQ") for _ in range(r.get("Q"))]
        imgs[iid] = (qt, cid, name, [tuple(p) for p in p2d])
    assert r.done()
    r = Reader(os.path.join(d, "points3D.bin"))
    for _ in range(r.get("Q")):
        pid = r.get("Q")
        xyz = list(r.get("3d"))
        rgb = r.get("3B")
        err = r.get("d")
        track = [r.get("II") for _ in range(r.get("Q"))]
        pts[pid] = (xyz, tuple(rgb), err, [tuple(t) for t in track])
    assert r.done()
    return cams, imgs, pts


# --- Main -------------------------------------------------------------------

def run(cmd, cwd):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        sys.exit("command failed")


def check(cond, msg):
    if not cond:
        sys.exit("FAILED: " + msg)
    print("ok:", msg)


def main():
    bin_dir, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    exe = lambda n: os.path.join(bin_dir, n + (".exe" if os.name == "nt" else ""))
    scene_dir = os.path.join(work, "scene")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(scene_dir)

    views, poses, structure = make_scene()
    with open(os.path.join(scene_dir, "sfm_data.json"), "w") as f:
        json.dump(sfm_data_json(views, poses, structure), f, indent=1)
    write_images(os.path.join(scene_dir, "imgs"), views)

    # Convert from another current directory: the images must be found
    # relative to the sfm_data file.
    sfm_bin = os.path.join(scene_dir, "sfm_data.bin")
    run([exe("openMVG_main_ConvertSfM_DataFormat"), "-i",
         os.path.join(scene_dir, "sfm_data.json"), "-o", sfm_bin], work)
    out = os.path.join(work, "colmap")
    run([exe("openMVG_main_openMVG2Colmap"), "-i", sfm_bin, "-o", out, "-f", "both"], work)

    tcams, timgs, tpts = read_text(out)
    bcams, bimgs, bpts = read_binary(out)
    check(tcams == bcams, "text and binary cameras are identical")
    check(timgs == bimgs, "text and binary images are identical (bit-exact doubles)")
    check(tpts == bpts, "text and binary points are identical (bit-exact doubles)")

    exported_views = {v["id"] for v in views
                      if v["id"] in poses and CAMERAS[v["cam"]]["model"] != "spherical"}
    check(set(bimgs) == exported_views, "only posed views with a supported camera are exported")
    check(sorted(bcams) == [0, 1, 2, 3, 4], "spherical camera is not exported")
    check(set(bpts) == {pid for pid, _, _ in structure}, "all points are exported")
    check(all(name == "imgs/view_%02d.ppm" % iid for iid, (_, _, name, _) in bimgs.items()),
          "image names are relative to root_path, with forward slashes")

    # Consistency of tracks and 2D points
    nobs = 0
    for pid, (_, _, _, track) in bpts.items():
        for iid, idx in track:
            assert bimgs[iid][3][idx][2] == pid
            nobs += 1
    check(nobs == sum(len(i[3]) for i in bimgs.values()), "tracks and 2D points reference each other")

    # Reprojection: COLMAP formulas on the COLMAP model vs OpenMVG formulas
    # on the sfm_data
    sfm_obs = {(pid + 0, vid): (x, X) for pid, X, obs in structure
               for vid, (_, x) in obs.items()}
    res_colmap, res_omvg, err_ok = [], [], True
    for pid, (xyz, _, err, track) in bpts.items():
        X = np.array(xyz)
        errs = []
        for iid, idx in track:
            qt, cid, _, p2d = bimgs[iid]
            model, _, _, params = bcams[cid]
            Xc = quat_to_R(*qt[:4]) @ X + np.array(qt[4:])
            rc = np.array(p2d[idx][:2]) - colmap_project(model, params, Xc)
            v = next(v for v in views if v["id"] == iid)
            x, _ = sfm_obs[(pid, iid)]
            R, C = poses[iid]
            ro = x - omvg_project(CAMERAS[v["cam"]], R, C, X)
            res_colmap.append(rc)
            res_omvg.append(ro)
            errs.append(np.linalg.norm(ro))
        err_ok &= abs(np.mean(errs) - err) < 1e-9
    res_colmap, res_omvg = np.array(res_colmap), np.array(res_omvg)
    rms = lambda r: math.sqrt(np.mean(np.sum(r * r, axis=1)))
    print("RMS OpenMVG %.12f px, COLMAP %.12f px, max residual difference %.3g px"
          % (rms(res_omvg), rms(res_colmap), np.abs(res_colmap - res_omvg).max()))
    check(abs(rms(res_colmap) - rms(res_omvg)) < 1e-9, "COLMAP reprojection RMS equals OpenMVG's")
    check(np.abs(res_colmap - res_omvg).max() < 1e-8, "every residual matches")
    check(err_ok, "per-point ERROR equals OpenMVG's mean reprojection error")

    # Colors are sampled in the first image of the track that sees the point
    colors_ok = True
    for pid, (_, rgb, _, track) in bpts.items():
        iid, idx = min(track)
        x = np.array(bimgs[iid][3][idx][:2]) - 0.5
        expected = image_color(iid, int(round(x[0])), int(round(x[1])))
        colors_ok &= tuple(rgb) == tuple(expected)
    check(colors_ok, "point colors are sampled from the images")

    # Missing images: still a complete model, with gray points
    shutil.rmtree(os.path.join(scene_dir, "imgs"))
    out2 = os.path.join(work, "colmap_noimages")
    run([exe("openMVG_main_openMVG2Colmap"), "-i", sfm_bin, "-o", out2, "-f", "bin"], work)
    _, _, pts2 = read_binary(out2)
    check(set(pts2) == set(bpts) and all(p[1] == (128, 128, 128) for p in pts2.values()),
          "without images all points are written, in gray")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
