"""Shared trajectory and experiment-output conventions.

Every experiment is stored as:

    results/<sequence_id>/<run_name>/<approach>/<variant>/
        config.json
        trajectory.npz
        metrics.csv
        summary.json
        overlay.mp4
"""

import json
from pathlib import Path
import numpy as np
from pose_utils import prepare_scales, validate_trajectory

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = ROOT / "results"
EXPERIMENT_SCHEMA_VERSION = 1

def _json_default(value):
    """Convert common NumPy configuration values to ordinary JSON values"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")

def _safe_path_component(value, name):
    """Validate a user-readable name before using it as one path component"""
    value = str(value)
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{name} must be one nonempty path component, got {value!r}")
    return value

def experiment_output_dir(approach, variant, sequence_id, run_name,
                          results_root=DEFAULT_RESULTS_ROOT):
    """Return an experiment directory grouped for visual comparison"""
    parts = [
        _safe_path_component(sequence_id, "sequence_id"),
        _safe_path_component(run_name, "run_name"),
        _safe_path_component(approach, "approach"),
        _safe_path_component(variant, "variant"),
    ]
    return Path(results_root).joinpath(*parts)

def prepare_frame_indices(frame_indices, n_poses, sequence_n_frames=None):
    """Validate a contiguous source-frame interval for a pose array"""
    if n_poses <= 0:
        raise ValueError("a trajectory must contain at least one pose")

    if frame_indices is None:
        indices = np.arange(n_poses, dtype=np.int64)
    else:
        raw = np.asarray(frame_indices)
        if raw.shape != (n_poses,):
            raise ValueError(f"frame_indices must have shape ({n_poses},), got {raw.shape}")
        if not np.issubdtype(raw.dtype, np.integer):
            if not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
                raise ValueError("frame_indices must contain integers")
        indices = raw.astype(np.int64)

    if np.any(indices < 0):
        raise ValueError("frame_indices cannot contain negative values")
    if len(indices) > 1 and not np.all(np.diff(indices) == 1):
        raise ValueError(
            "frame_indices must be strictly increasing and contiguous so "
            "frame-to-frame smoothness metrics remain meaningful"
        )
    if sequence_n_frames is not None and indices[-1] >= sequence_n_frames:
        raise ValueError(
            f"frame index {indices[-1]} is outside a sequence with "
            f"{sequence_n_frames} frames"
        )
    return indices

def save_experiment(approach, variant, sequence_id, run_name, rotations,
                    translations, scales, frame_indices=None, parameters=None,
                    results_root=DEFAULT_RESULTS_ROOT):
    """Save a trajectory and reproducibility metadata in the standard layout
    Returns experiment directory
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    scales = prepare_scales(scales, len(rotations))
    validation = validate_trajectory(rotations, translations, scales)
    indices = prepare_frame_indices(frame_indices, len(rotations))

    output_dir = experiment_output_dir(approach, variant, sequence_id, run_name, results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "trajectory.npz"
    np.savez_compressed(trajectory_path, R=rotations, t=translations, s=scales,
                        frame_indices=indices)

    config = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "approach": str(approach),
        "variant": str(variant),
        "sequence_id": str(sequence_id),
        "run_name": str(run_name),
        "frame_range": {
            "start": int(indices[0]),
            "end_exclusive": int(indices[-1] + 1),
            "count": int(len(indices)),
        },
        "trajectory_file": trajectory_path.name,
        "scale_policy": (
            "constant" if np.allclose(scales, scales[0]) else "per_frame"
        ),
        "parameters": {} if parameters is None else parameters,
        "trajectory_validation": validation,
    }
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True, default=_json_default)
        file.write("\n")

    return output_dir
