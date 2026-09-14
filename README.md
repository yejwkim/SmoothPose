# SmoothPose

This project estimates a smooth object pose trajectory from each video's
per-frame point cloud and mask. The final method is **Approach 3S**: robust
sequential ICP initialized from the preceding frame, followed by uniform
five-frame temporal smoothing of translation and rotation.

## Contents

- `solution/`: the four submitted trajectories and overlay videos. Files
  `01`, `03`, `04`, and `07` correspond to sequences `01__01`, `01__03`,
  `01__04`, and `01__07`. Each NPZ contains `R`, `t`, `s`, and
  `frame_indices`.
- `approach_0.py`–`approach_5.py`: experimental methods, progressing from
  centroid/PCA baselines and independent ICP through filtering, sequential
  ICP, temporal candidate selection, and joint optimization.
- `approach_3_smoothed.py`: final temporal smoothing and evaluation stage.
  (Approach 3 followed by Approach 2-style smoothing)
- `evaluation.py`, `experiment_utils.py`, `pose_utils.py`: shared evaluation,
  output, and pose utilities.
- `toy_task/`: supplied sequences, masks, point clouds, and template meshes.

## Reproducing final trajectories

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt

for sequence in 01__01 01__03 01__04 01__07; do
  python3 approach_3.py "$sequence" full --variant robust --direction forward
done
python3 approach_3_smoothed.py --all
```

Generated trajectories, metrics, and overlays are written to:

```text
results/<sequence>/full/approach_3_smoothed/window_05/
```

## Create interval videos

Generate a slowed inspection clip from a center frame and optional radius:

```bash
python3 test_frames.py 01__01 84 10
```

This keeps every source frame in `[74, 95)`, writes it at 10 FPS under
`outputs/test_frames/01__01/`, and highlights the mask boundary. After checking
the clip, copy it into `stretches/<sequence>/` with its category name and record
the same half-open frame interval in `stretches/SELECTION.md`:

```bash
mkdir -p stretches/01__01
cp outputs/test_frames/01__01/center_00084_radius_10.mp4 \
  stretches/01__01/clear.mp4
```

Run `python3 inspect_sequences.py` first to check if automatic candidate 
intervals for `clear`, `motion`, and `occlusion` would be useful before
manual selection. Interval names can be customized if necessary.

## Test each approach

Use a named interval (e.g. `clear`, `motion`, or `occlusion`) from
`stretches/SELECTION.md`. The examples below use the suitcase's `clear`
interval. Run them in order because some approaches consume earlier results.

```bash
# 0: centroid and PCA diagnostic baselines
python3 approach_0.py 01__01 clear --variant both

# 1: independent single- and multi-initialization ICP
python3 approach_1.py 01__01 clear --variant both

# 2: filter the Approach 1 trajectory (5-frame window)
python3 approach_2.py 01__01 clear --window short

# 3: ordinary and robust sequential ICP, tracked forward
python3 approach_3.py 01__01 clear --variant both --direction forward

# 4: independent versus temporal pose-candidate selection
python3 approach_4.py 01__01 clear --variant both

# 5: joint geometry, temporal, and mask refinement
python3 approach_5.py 01__01 clear --variant all --scale-policy fixed

# 3S: smooth an existing full robust-forward Approach 3 trajectory
python3 approach_3_smoothed.py 01__01 full
```

Every command evaluates its output by default and writes the trajectory,
metrics, configuration, and overlay under `results/<sequence>/<interval>/`.
Use `python3 approach_<number>.py --self-check` for a quick logic check or
`--skip-evaluation` when an overlay and metrics are not needed.
