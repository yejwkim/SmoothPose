"""Approach 3S: uniformly smooth robust sequential ICP trajectories.

This is a post-processing stage: it loads an existing Approach 3
robust_forward trajectory, applies the same five-frame filters to every
sequence, and evaluates the result without refitting any observations.

Translation uses a second-order Savitzky-Golay filter. Rotation uses the
triangularly weighted Markley quaternion average implemented for Approach 2.
Scale and frame indices are preserved exactly.

Usage:
    python3 approach_3_smoothed.py 01__01 full
    python3 approach_3_smoothed.py --all
    python3 approach_3_smoothed.py --self-check
"""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from approach_2 import (
    DEFAULT_POLYNOMIAL_ORDER,
    save_smoothing_diagnostics,
    smooth_trajectory,
)
from evaluation import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    evaluate_trajectory,
)
from experiment_utils import (
    DEFAULT_RESULTS_ROOT,
    experiment_output_dir,
    prepare_frame_indices,
    save_experiment,
)
from pose_utils import prepare_scales, rotation_geodesic, validate_trajectory
from toy_task.load_frame import load_sequence

SEQUENCE_IDS = ("01__01", "01__03", "01__04", "01__07")
SOURCE_APPROACH = "approach_3"
SOURCE_VARIANT = "robust_forward"
OUTPUT_APPROACH = "approach_3_smoothed"
WINDOW_SIZE = 5
OUTPUT_VARIANT = f"window_{WINDOW_SIZE:02d}"

def load_source_trajectory(sequence_id, run_name,
                           results_root=DEFAULT_RESULTS_ROOT):
    """Load and validate one saved robust-forward Approach 3 trajectory"""
    source_dir = experiment_output_dir(
        SOURCE_APPROACH,
        SOURCE_VARIANT,
        sequence_id,
        run_name,
        results_root,
    )
    trajectory_path = source_dir / "trajectory.npz"
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"missing source trajectory: {trajectory_path}\n"
            "Run Approach 3 with --variant robust --direction forward first."
        )

    sequence = load_sequence(sequence_id)
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        required = {"R", "t", "s", "frame_indices"}
        missing = required.difference(trajectory.files)
        if missing:
            raise ValueError(
                f"{trajectory_path} is missing keys: "
                f"{', '.join(sorted(missing))}"
            )
        rotations = np.asarray(trajectory["R"], dtype=np.float64)
        translations = np.asarray(trajectory["t"], dtype=np.float64)
        scales = prepare_scales(trajectory["s"], len(rotations))
        frame_indices = prepare_frame_indices(
            trajectory["frame_indices"],
            len(rotations),
            sequence_n_frames=sequence["n_frames"],
        )

    validate_trajectory(rotations, translations, scales)
    if run_name == "full" and not np.array_equal(
        frame_indices, np.arange(sequence["n_frames"])
    ):
        raise ValueError(
            f"{trajectory_path} is named 'full' but does not cover all "
            f"{sequence['n_frames']} source frames"
        )

    return {
        "R": rotations,
        "t": translations,
        "s": scales,
        "frame_indices": frame_indices,
        "path": trajectory_path,
    }

def run_experiment(sequence_id, run_name,
                   results_root=DEFAULT_RESULTS_ROOT,
                   evaluate=True,
                   evaluation_sample_count=DEFAULT_SAMPLE_COUNT,
                   evaluation_sample_seed=DEFAULT_SAMPLE_SEED):
    """Smooth, save, and optionally evaluate one Approach 3 trajectory."""
    source = load_source_trajectory(sequence_id, run_name, results_root)
    trajectory = smooth_trajectory(
        source["R"],
        source["t"],
        source["s"],
        source["frame_indices"],
        window_size=WINDOW_SIZE,
        polynomial_order=DEFAULT_POLYNOMIAL_ORDER,
        description=f"smooth {sequence_id} {run_name} Approach 3",
    )

    output_dir = save_experiment(
        OUTPUT_APPROACH,
        OUTPUT_VARIANT,
        sequence_id,
        run_name,
        trajectory["R"],
        trajectory["t"],
        trajectory["s"],
        trajectory["frame_indices"],
        parameters={
            "source_approach": SOURCE_APPROACH,
            "source_variant": SOURCE_VARIANT,
            "source_trajectory": str(source["path"].resolve()),
            "translation_filter": "Savitzky-Golay",
            "translation_polynomial_order": DEFAULT_POLYNOMIAL_ORDER,
            "rotation_filter": (
                "triangularly weighted Markley quaternion mean"
            ),
            "window_size_frames": WINDOW_SIZE,
            "uniform_configuration_across_sequences": True,
            "refits_observations": False,
            "scale_policy": "preserve source values",
        },
        results_root=results_root,
    )
    diagnostics_path = save_smoothing_diagnostics(
        output_dir, trajectory["diagnostics"]
    )
    print(
        f"saved Approach 3S {sequence_id}/{run_name} {OUTPUT_VARIANT} "
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
    """Check uniform smoothing, unchanged metadata, and valid rotations"""
    n_frames = 21
    frame_indices = np.arange(n_frames, dtype=np.int64)
    scales = np.ones(n_frames)

    clean_translation = np.column_stack((
        np.linspace(0.0, 0.2, n_frames),
        np.linspace(-0.1, 0.1, n_frames),
        np.full(n_frames, 2.0),
    ))
    translation_noise = np.zeros_like(clean_translation)
    translation_noise[:, 0] = 0.01 * (-1.0) ** np.arange(n_frames)
    noisy_translation = clean_translation + translation_noise

    clean_angles = np.linspace(0.0, 20.0, n_frames)
    noisy_angles = clean_angles + 3.0 * (-1.0) ** np.arange(n_frames)
    noisy_rotations = Rotation.from_euler(
        "z", noisy_angles[:, None], degrees=True
    ).as_matrix()

    result = smooth_trajectory(
        noisy_rotations,
        noisy_translation,
        scales,
        frame_indices,
        window_size=WINDOW_SIZE,
        polynomial_order=DEFAULT_POLYNOMIAL_ORDER,
        description="Approach 3S self-check",
    )
    report = validate_trajectory(result["R"], result["t"], result["s"])
    assert report["max_orthogonality_error"] < 1e-12
    np.testing.assert_array_equal(result["frame_indices"], frame_indices)
    np.testing.assert_array_equal(result["s"], scales)
    assert np.mean(np.linalg.norm(result["t"] - clean_translation, axis=1)) < (
        np.mean(np.linalg.norm(noisy_translation - clean_translation, axis=1))
    )

    raw_steps = np.array([
        rotation_geodesic(noisy_rotations[index - 1], noisy_rotations[index])
        for index in range(1, n_frames)
    ])
    smooth_steps = np.array([
        rotation_geodesic(result["R"][index - 1], result["R"][index])
        for index in range(1, n_frames)
    ])
    assert np.std(smooth_steps) < np.std(raw_steps)
    print("Approach 3S self-check passed.")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?", choices=SEQUENCE_IDS)
    parser.add_argument("run_name", nargs="?")
    parser.add_argument(
        "--all",
        action="store_true",
        help="smooth all four full robust-forward Approach 3 trajectories",
    )
    parser.add_argument(
        "--evaluation-samples", type=int, default=DEFAULT_SAMPLE_COUNT
    )
    parser.add_argument(
        "--evaluation-seed", type=int, default=DEFAULT_SAMPLE_SEED
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        if args.all or args.sequence_id is not None or args.run_name is not None:
            parser.error("--self-check cannot be combined with an experiment")
        return args
    if args.all:
        if args.sequence_id is not None or args.run_name is not None:
            parser.error("--all cannot be combined with sequence_id or run_name")
    elif args.sequence_id is None or args.run_name is None:
        parser.error("provide sequence_id and run_name, or use --all")
    return args

def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return

    selected = (
        [(sequence_id, "full") for sequence_id in SEQUENCE_IDS]
        if args.all else [(args.sequence_id, args.run_name)]
    )
    for sequence_id, run_name in selected:
        run_experiment(
            sequence_id,
            run_name,
            results_root=args.results_root,
            evaluate=not args.skip_evaluation,
            evaluation_sample_count=args.evaluation_samples,
            evaluation_sample_seed=args.evaluation_seed,
        )


if __name__ == "__main__":
    main()
