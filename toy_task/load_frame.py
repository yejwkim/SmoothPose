"""Minimal loader for the toy task data (numpy + opencv only).

    import load_frame as L
    seq = L.load_sequence('01__07')          # npz arrays + n_frames
    f   = L.load_frame('01__07', 100)        # points, image, mask, intrinsics
    uv  = L.project(f['points'], f['fx'], f['fy'], f['cx'], f['cy'])
    V0, F = L.load_template(seq['template'])

Run it directly for a self-check on one frame:
    python load_frame.py 01__07 100
"""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_sequence(seq, root=ROOT):
    """The sequence's inputs.npz as a dict, plus `n_frames`, `dir` and the
    path of its object `template`."""
    d = os.path.join(root, 'sequences', seq)
    z = np.load(os.path.join(d, 'inputs.npz'), allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out['n_frames'] = len(out['points_offsets']) - 1
    out['dir'] = d
    out['template'] = os.path.join(root, 'templates',
                                   f"{out['object_name']}.obj")
    return out


def points(seq, t):
    """Frame `t`'s object point cloud, (n_t, 3) float32, camera frame.
    `seq` is a load_sequence dict (or the sequence id)."""
    if isinstance(seq, str):
        seq = load_sequence(seq)
    off = seq['points_offsets']
    return seq['points_flat'][off[t]:off[t + 1]]


def load_frame(seq, t, root=ROOT, with_image=True):
    """Everything for one frame: `points` (n,3) camera frame, `mask` (H,W)
    bool object mask, `image` (H,W,3) BGR video frame (None if with_image is
    False), and the intrinsics `fx, fy, cx, cy`."""
    if isinstance(seq, str):
        seq = load_sequence(seq, root)
    mask = cv2.imread(os.path.join(seq['dir'], 'object_masks', f'{t:05d}.png'),
                      cv2.IMREAD_GRAYSCALE) > 127
    img = None
    if with_image:
        cap = cv2.VideoCapture(os.path.join(seq['dir'], 'video.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'could not read video frame {t}')
    fx, fy = seq['focal_length'][t]
    cx, cy = seq['principal_point'][t]
    return {'points': points(seq, t), 'mask': mask, 'image': img,
            'fx': float(fx), 'fy': float(fy),
            'cx': float(cx), 'cy': float(cy)}


def project(pts, fx, fy, cx, cy):
    """Camera-frame points (n,3) -> pixels (n,2), OpenCV pinhole."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([fx * pts[:, 0] / pts[:, 2] + cx,
                     fy * pts[:, 1] / pts[:, 2] + cy], axis=1)


def load_template(path):
    """(V (N,3) float64, F (M,3) int64) from a template OBJ. The face lines
    are 'f v/vt v/vt v/vt', so only the first index of each triple is used."""
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                F.append([int(c.split('/')[0]) - 1 for c in line.split()[1:4]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


if __name__ == '__main__':
    import sys

    seq_id = sys.argv[1] if len(sys.argv) > 1 else '01__07'
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seq = load_sequence(seq_id)
    f = load_frame(seq, t)
    uv = project(f['points'], f['fx'], f['fy'], f['cx'], f['cy'])
    H, W = f['mask'].shape
    uvi = np.round(uv).astype(int)
    ok = ((uvi[:, 0] >= 0) & (uvi[:, 0] < W)
          & (uvi[:, 1] >= 0) & (uvi[:, 1] < H))
    V0, F0 = load_template(seq['template'])
    print(f"{seq_id} ({seq['object_label']}), {seq['n_frames']} frames, "
          f"frame {t}")
    print(f"  points {f['points'].shape}  centroid "
          f"{f['points'].mean(0).round(3)}")
    print(f"  image {f['image'].shape}  mask {f['mask'].shape} "
          f"({int(f['mask'].sum())} px)")
    print(f"  template {V0.shape[0]} verts / {F0.shape[0]} faces")
    print(f"  projected points inside the object mask: "
          f"{f['mask'][uvi[ok, 1], uvi[ok, 0]].mean():.3f}")
