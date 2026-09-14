# Assignment — fitting a rigid object template through a video, smoothly

## Goal

Four videos show a person handling an object. For each video, recover the **pose of that object in every frame**, as a trajectory that both fits the per-frame observations and moves smoothly in time.

Concretely, for each sequence produce a per-frame rigid placement of the object's template mesh in the camera frame:

$$
\mathbf{v}^{(i)}_t \;=\; s\,\mathbf{R}_t\,\mathbf{v}^{(i)}_0 \;+\; \mathbf{t}_t ,
\qquad
\mathbf{R}_t \in SO(3),\quad \mathbf{t}_t \in \mathbb{R}^3,\quad s > 0
$$

where $\mathbf{v}^{(i)}_0$ is the $i$-th template vertex in its own template coordinate frame and $\mathbf{v}^{(i)}_t$ is that vertex in the camera frame at time $t$. 

The object is rigid and the template is metric.

Two requirements hold at once:

1. **Per-frame fit.** The posed template must explain that frame: it should agree with the frame's object point cloud, and its silhouette (projected with the given intrinsics) should agree with the object mask.
2. **Temporal smoothness.** The trajectory $\{(\mathbf{R}_t, \mathbf{t}_t)\}_{t=1}^{T}$ must be smooth. The real motion is continuous; fitting frames independently produces jitter, flips, and 90-180 degree orientation jumps on near-symmetric objects.

Smoothness must not be bought by walking away significantly from the data: a smooth but wrong trajectory is a failure, not a solution.

**The task is deliberately open-ended — any approach is fine.** There is no reference implementation to match, no required loss, and no required method.

## What you are given

```
toy_task/
  README.md
  load_frame.py                           # minimal loader (numpy + opencv)
  templates/
    01.obj  03.obj  04.obj  07.obj        # object templates, TEMPLATE coords
  sequences/
    01__01/  01__03/  01__04/  01__07/
      video.mp4                           # 1920x1080, 30 fps, one frame per timestep
      object_masks/00000.png ...          # binary (0/255) object mask per frame
      inputs.npz                          # per-frame geometry + intrinsics
```

| sequence | frames | object | template | template extent (m) |
|---|---|---|---|---|
| `01__01` | 321 | suitcase | `templates/01.obj` (591 v / 1188 f) | 0.97 x 0.45 x 0.29 |
| `01__03` | 531 | ball     | `templates/03.obj` (998 v / 1992 f) | 0.22 x 0.22 x 0.22 |
| `01__04` | 181 | umbrella | `templates/04.obj` (1267 v / 2153 f) | 0.99 x 0.99 x 0.86 |
| `01__07` | 371 | chair    | `templates/07.obj` (342 v / 678 f)  | 0.94 x 0.66 x 0.70 |

Frame index `t` means the same thing everywhere: `video.mp4` frame `t`, `object_masks/{t:05d}.png`, and row `t` of the per-frame arrays in `inputs.npz`. Frames are contiguous, none are missing, and every frame has all three inputs.

There is **no ground truth** in this folder — the object poses are what you are being asked to produce.

### `inputs.npz`

| key | shape / dtype | meaning |
|---|---|---|
| `points_flat` | (M, 3) float32 | **object point clouds in the camera frame**, all frames concatenated — the per-frame 3D observation |
| `points_offsets` | (T+1,) int64 | frame `t`'s points are `points_flat[off[t]:off[t+1]]` (10000 per frame, fewer where the source geometry itself has fewer vertices — only `01__03`, min 7485) |
| `focal_length` | (T, 2) float32 | `(fx, fy)` in pixels |
| `principal_point` | (T, 2) float32 | `(cx, cy)` in pixels |
| `fps` | int64 | 30 |
| `sequence_id`, `object_name`, `object_label` | str | e.g. `01__04`, `04`, `umbrella` |

Intrinsics are in fact constant across all frames and sequences (f≈918.45, c≈(956.97, 555.94)), but they are stored per frame so you never have to assume it. Image size comes from the video or the masks (1080 x 1920).

### Loading a frame

The point clouds are ragged, so they are stored concatenated and sliced by `points_offsets`:

```python
import numpy as np
z   = np.load('sequences/01__07/inputs.npz', allow_pickle=True)
off = z['points_offsets']
t   = 100
pts = z['points_flat'][off[t]:off[t+1]]      # (n_t, 3) float32, camera frame
fx, fy = z['focal_length'][t]
cx, cy = z['principal_point'][t]
```

`load_frame.py` wraps that and pulls the matching video frame and mask:

```python
import load_frame as L

seq   = L.load_sequence('01__07')                # npz arrays + n_frames + template path
f     = L.load_frame(seq, 100)                   # or L.load_frame('01__07', 100)
f['points']                                      # (n_t, 3) float32, camera frame
f['mask']                                        # (1080, 1920) bool, object mask
f['image']                                       # (1080, 1920, 3) BGR video frame
f['fx'], f['fy'], f['cx'], f['cy']               # intrinsics

uv    = L.project(f['points'], f['fx'], f['fy'], f['cx'], f['cy'])   # (n_t, 2) pixels
V0, F = L.load_template(seq['template'])         # template verts / faces
pts   = L.points(seq, 100)                       # just the point cloud
```

It needs only numpy and opencv. Run directly it self-checks one frame (`python load_frame.py 01__07 100`), printing shapes and the fraction of projected points landing inside the object mask. `with_image=False` skips the video decode.

## What to hand in

1. **The trajectories** — one file per sequence, however you like as long as it is self-describing. A natural form is an npz with `R` (T,3,3) float, `t` (T,3) float and `s` (T,), following the equation above, with `T` matching that sequence's frame count.
2. **The code** that produced them, runnable against this folder.
3. **A short write-up** — what you did, what you tried that did not work, where your result is weakest and why.
4. **An overlay video** per sequence — your posed template drawn over the video frames. It is the fastest way for both of us to see what the numbers mean.

## Know your inputs

The point cloud is a per-frame monocular reconstruction, not a clean scan.
Expect:

- **Partial, noisy geometry.** It is the object-labelled part of a single-image 3D reconstruction, cut out by a segmentation step, recomputed independently for every frame. Thin parts (chair legs, umbrella shaft) can be missing or fused to the person.
- **A residual along-ray depth/scale bias** of a few percent, roughly constant within a sequence.
- **Occlusion.** The person handles the object, so on many frames the mask is a partial silhouette and the point cloud a partial surface. Some frames are far worse than their neighbours; recovering those from temporal context is part of the task.
- **Near-symmetric objects.** The ball is a sphere (orientation is unobservable); the suitcase and chair have strong discrete symmetries. Pose is ambiguous per frame and is resolved, if at all, by the trajectory.

## Checking yourself

With no ground truth, judge your own result by:

- **Reprojection overlay.** Rasterise the posed template with the intrinsics above and compare against `object_masks/{t}.png` (IoU per frame). Watch the overlay video; failures are usually obvious to the eye.
- **Fit residual.** Chamfer distance between the posed template and that frame's points.
- **Smoothness.** Inter-frame rotation geodesic (degrees) and translation delta (cm). A good trajectory has no isolated spikes.
- **The classic failure to look for:** low per-frame residual together with large frame-to-frame rotation jumps between symmetry-equivalent poses.

Report these per sequence rather than as one pooled number — the four sequences
fail in different ways.

## Where the data comes from

Derived from the **InterCap** dataset ([project page](https://intercap.is.tue.mpg.de/)) — subject 01, camera 1, colour stream; the `templates/` meshes are InterCap's own object scans. Use is subject to InterCap's original licence and terms.

> Yinghao Huang, Omid Taheri, Michael J. Black, Dimitrios Tzionas.
> *InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction
> from Multi-view RGB-D Images.* International Journal of Computer Vision
> (IJCV), 2024. [doi:10.1007/s11263-024-01984-1](https://doi.org/10.1007/s11263-024-01984-1)

> Yinghao Huang, Omid Taheri, Michael J. Black, Dimitrios Tzionas.
> *InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction.*
> German Conference on Pattern Recognition (GCPR), LNCS 13485, pp. 281-299,
> Springer, 2022.
