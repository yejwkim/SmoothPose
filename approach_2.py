"""Approach 2: temporally filter saved independent ICP trajectories.

Translation is filtered with a Savitzky-Golay polynomial filter. Rotation is
filtered with a local, triangularly weighted Markley quaternion average, which
keeps every output matrix in SO(3).

Default source trajectories:
    ball (01__03): translation_only
    other objects: multi_initialization

Default windows over the selected 21-frame stretches:
    short=5 frames, medium=9 frames, long=15 frames

Approach 2 does not refit the point clouds. It smooths Approach 1 poses and
then evaluates how much continuity improves and how much observation fit is
lost.

Usage:
    python3 approach_2.py
    python3 approach_2.py 01__01 clear --window short
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from approach_0 import load_stretches
from approach_1 import BALL_SEQUENCE_ID
from evaluation import (DEFAULT_SAMPLE_COUNT, DEFAULT_SAMPLE_SEED,
                        evaluate_trajectory)
from experiment_utils import (DEFAULT_RESULTS_ROOT, experiment_output_dir,
                              prepare_frame_indices, save_experiment)
from pose_utils import (prepare_scales, rotation_geodesic,
                        validate_trajectory)
from toy_task.load_frame import load_sequence

WINDOWS = {
    "short": 5,
    "medium": 9,
    "long": 15,
}
SOURCE_VARIANTS = (
    "single_initialization",
    "multi_initialization",
    "translation_only",
)
DEFAULT_POLYNOMIAL_ORDER = 2

def default_source_variant(sequence_id):
    """Choose the physically meaningful Approach 1 source by object"""
    if sequence_id == BALL_SEQUENCE_ID:
        return "translation_only"
    return "multi_initialization"

def load_source_trajectory(sequence_id, run_name, source_variant,
                           results_root=DEFAULT_RESULTS_ROOT):
    """Load and validate an Approach 1 trajectory from the standard layout"""
    source_dir = experiment_output_dir(
        "approach_1", source_variant, sequence_id, run_name, results_root
    )
    trajectory_path = source_dir / "trajectory.npz"
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"missing source trajectory: {trajectory_path}\n"
            "Run the corresponding Approach 1 experiment first."
        )

    sequence = load_sequence(sequence_id)
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        required = {"R", "t", "s", "frame_indices"}
        missing = required.difference(trajectory.files)
        if missing:
            raise ValueError(f"{trajectory_path} is missing keys: {', '.join(sorted(missing))}")
        rotations = np.asarray(trajectory["R"], dtype=np.float64)
        translations = np.asarray(trajectory["t"], dtype=np.float64)
        scales = prepare_scales(trajectory["s"], len(rotations))
        frame_indices = prepare_frame_indices(
            trajectory["frame_indices"],
            len(rotations),
            sequence_n_frames=sequence["n_frames"],
        )

    validate_trajectory(rotations, translations, scales)
    return {
        "R": rotations,
        "t": translations,
        "s": scales,
        "frame_indices": frame_indices,
        "path": trajectory_path,
    }

def validate_window(window_size, n_frames, polynomial_order):
    """Validate a symmetric odd smoothing window"""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window size must be odd and at least 3, got {window_size}")
    if window_size > n_frames:
        raise ValueError(
            f"window size {window_size} exceeds trajectory length {n_frames}"
        )
    if polynomial_order < 0 or polynomial_order >= window_size:
        raise ValueError(
            f"polynomial order must satisfy 0 <= order < {window_size}, "
            f"got {polynomial_order}"
        )

def smooth_translations(translations, window_size,
                        polynomial_order=DEFAULT_POLYNOMIAL_ORDER):
    """Apply a Savitzky-Golay filter independently to x, y, and z"""
    translations = np.asarray(translations, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(
            f"translations must have shape (T, 3), got {translations.shape}"
        )
    validate_window(window_size, len(translations), polynomial_order)
    return savgol_filter(
        translations,
        window_length=window_size,
        polyorder=polynomial_order,
        axis=0,
        mode="interp",
    )

def markley_quaternion_mean(quaternions, weights):
    """Return the sign-invariant weighted chordal mean of unit quaternions"""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4), got {quaternions.shape}")
    if weights.shape != (len(quaternions),):
        raise ValueError(f"weights must have shape ({len(quaternions)},), got {weights.shape}")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and not all zero")

    weights = weights / weights.sum()
    accumulator = np.einsum("n,ni,nj->ij", weights, quaternions, quaternions)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean = eigenvectors[:, np.argmax(eigenvalues)]
    return mean / np.linalg.norm(mean)

def smooth_rotations(rotations, window_size, description="smooth rotations"):
    """Average local rotations without averaging matrices or Euler angles"""
    rotations = np.asarray(rotations, dtype=np.float64)
    validate_trajectory(rotations, np.zeros((len(rotations), 3)), np.ones(len(rotations)))
    validate_window(window_size, len(rotations), polynomial_order=0)

    quaternions = Rotation.from_matrix(rotations).as_quat()
    radius = window_size // 2
    smoothed = []
    for frame_index in tqdm(range(len(rotations)), desc=description, unit="frame"):
        start = max(0, frame_index - radius)
        end = min(len(rotations), frame_index + radius + 1)
        offsets = np.arange(start, end) - frame_index
        weights = radius + 1 - np.abs(offsets)
        mean_quaternion = markley_quaternion_mean(quaternions[start:end], weights)
        smoothed.append(Rotation.from_quat(mean_quaternion).as_matrix())
    return np.asarray(smoothed, dtype=np.float64)

def smooth_trajectory(rotations, translations, scales, frame_indices,
                      window_size,
                      polynomial_order=DEFAULT_POLYNOMIAL_ORDER,
                      description="smooth rotations"):
    """Return a temporally filtered trajectory and correction diagnostics"""
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    scales = prepare_scales(scales, len(rotations))
    frame_indices = prepare_frame_indices(frame_indices, len(rotations))
    validate_trajectory(rotations, translations, scales)

    smoothed_translations = smooth_translations(translations, window_size, polynomial_order)
    smoothed_rotations = smooth_rotations(rotations, window_size, description=description)
    validate_trajectory(smoothed_rotations, smoothed_translations, scales)

    diagnostics = []
    for index, frame_index in enumerate(frame_indices):
        diagnostics.append({
            "frame": int(frame_index),
            "rotation_correction_deg": rotation_geodesic(
                rotations[index], smoothed_rotations[index]
            ),
            "translation_correction_cm": 100.0 * float(np.linalg.norm(
                translations[index] - smoothed_translations[index]
            )),
        })

    return {
        "R": smoothed_rotations,
        "t": smoothed_translations,
        "s": scales.copy(),
        "frame_indices": frame_indices,
        "diagnostics": diagnostics,
    }

def save_smoothing_diagnostics(output_dir, diagnostics):
    """Save how far smoothing moved each original Approach 1 pose"""
    path = Path(output_dir) / "smoothing_diagnostics.csv"
    columns = [
        "frame",
        "rotation_correction_deg",
        "translation_correction_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(diagnostics)
    return path

def run_experiment(sequence_id, run_name, source_variant, window_name,
                   results_root=DEFAULT_RESULTS_ROOT,
                   polynomial_order=DEFAULT_POLYNOMIAL_ORDER,
                   evaluate=True, evaluation_sample_count=DEFAULT_SAMPLE_COUNT,
                   evaluation_sample_seed=DEFAULT_SAMPLE_SEED):
    """Load, smooth, save, and optionally evaluate one Approach 2 run"""
    window_size = WINDOWS[window_name]
    source = load_source_trajectory(sequence_id, run_name, source_variant, results_root)
    output_variant = f"{source_variant}_window_{window_size:02d}"
    trajectory = smooth_trajectory(
        source["R"],
        source["t"],
        source["s"],
        source["frame_indices"],
        window_size,
        polynomial_order=polynomial_order,
        description=f"smooth {sequence_id} {run_name} w={window_size}",
    )

    output_dir = save_experiment(
        "approach_2",
        output_variant,
        sequence_id,
        run_name,
        trajectory["R"],
        trajectory["t"],
        trajectory["s"],
        trajectory["frame_indices"],
        parameters={
            "source_approach": "approach_1",
            "source_variant": source_variant,
            "source_trajectory": str(source["path"].resolve()),
            "translation_filter": "Savitzky-Golay",
            "translation_polynomial_order": int(polynomial_order),
            "rotation_filter": "triangularly weighted Markley quaternion mean",
            "window_name": window_name,
            "window_size_frames": int(window_size),
            "refits_observations": False,
        },
        results_root=results_root,
    )
    diagnostics_path = save_smoothing_diagnostics(output_dir, trajectory["diagnostics"])
    print(
        f"saved Approach 2 {sequence_id}/{run_name} {output_variant} "
        f"-> {output_dir}"
    )
    print(f"smoothing diagnostics: {diagnostics_path}")

    if evaluate:
        evaluate_trajectory(
            sequence_id,
            output_dir / "trajectory.npz",
            output_dir,
            sample_count=evaluation_sample_count,
            sample_seed=evaluation_sample_seed,
        )
    return output_dir

def self_check():
    """Check that filtering preserves SO(3) and reduces controlled jitter."""
    n_frames = 21
    frame_indices = np.arange(40, 40 + n_frames)
    clean_translation = np.column_stack((
        np.linspace(0.0, 0.2, n_frames),
        np.linspace(-0.1, 0.1, n_frames),
        np.full(n_frames, 2.0),
    ))
    translation_noise = np.zeros_like(clean_translation)
    translation_noise[:, 0] = 0.012 * (-1.0) ** np.arange(n_frames)
    noisy_translation = clean_translation + translation_noise

    clean_angles = np.linspace(0.0, 20.0, n_frames)
    angle_noise = 4.0 * (-1.0) ** np.arange(n_frames)
    noisy_rotations = Rotation.from_euler(
        "z", (clean_angles + angle_noise)[:, None], degrees=True
    ).as_matrix()

    result = smooth_trajectory(
        noisy_rotations,
        noisy_translation,
        np.ones(n_frames),
        frame_indices,
        window_size=5,
        description="Approach 2 self-check",
    )
    report = validate_trajectory(result["R"], result["t"], result["s"])
    assert report["max_orthogonality_error"] < 1e-12
    assert np.mean(np.linalg.norm(result["t"] - clean_translation, axis=1)) < (
        np.mean(np.linalg.norm(noisy_translation - clean_translation, axis=1))
    )

    raw_changes = np.array([
        rotation_geodesic(noisy_rotations[index - 1], noisy_rotations[index])
        for index in range(1, n_frames)
    ])
    smooth_changes = np.array([
        rotation_geodesic(result["R"][index - 1], result["R"][index])
        for index in range(1, n_frames)
    ])
    assert np.std(smooth_changes) < np.std(raw_changes)
    print("Approach 2 self-check passed.")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("run_name", nargs="?")
    parser.add_argument(
        "--window",
        choices=tuple(WINDOWS) + ("all",),
        default="all",
        help="smoothing window to run (default: all)",
    )
    parser.add_argument(
        "--source-variant",
        choices=("auto",) + SOURCE_VARIANTS,
        default="auto",
        help="Approach 1 trajectory to smooth (default: object-specific)",
    )
    parser.add_argument("--polynomial-order", type=int,
                        default=DEFAULT_POLYNOMIAL_ORDER)
    parser.add_argument("--evaluation-samples", type=int,
                        default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--evaluation-seed", type=int,
                        default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--results-root", type=Path,
                        default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if (args.sequence_id is None) != (args.run_name is None):
        parser.error("provide both sequence_id and run_name, or neither")
    return args

def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return

    stretches = load_stretches()
    if args.sequence_id is None:
        selected = [
            (sequence_id, run_name)
            for sequence_id, named_ranges in stretches.items()
            for run_name in named_ranges
        ]
    else:
        if args.sequence_id not in stretches:
            raise ValueError(f"no selected stretches for {args.sequence_id}")
        if args.run_name not in stretches[args.sequence_id]:
            available = ", ".join(stretches[args.sequence_id])
            raise ValueError(
                f"no {args.run_name!r} stretch for {args.sequence_id}; "
                f"choose from {available}"
            )
        selected = [(args.sequence_id, args.run_name)]

    window_names = tuple(WINDOWS) if args.window == "all" else (args.window,)
    for sequence_id, run_name in selected:
        source_variant = (
            default_source_variant(sequence_id)
            if args.source_variant == "auto" else args.source_variant
        )
        for window_name in window_names:
            run_experiment(
                sequence_id,
                run_name,
                source_variant,
                window_name,
                results_root=args.results_root,
                polynomial_order=args.polynomial_order,
                evaluate=not args.skip_evaluation,
                evaluation_sample_count=args.evaluation_samples,
                evaluation_sample_seed=args.evaluation_seed,
            )

if __name__ == "__main__":
    main()
