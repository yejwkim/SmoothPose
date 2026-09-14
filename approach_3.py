"""Approach 3: robust sequential ICP with temporal initialization

Each frame starts from the pose estimated for the preceding tracking frame.
Forward and backward runs are saved separately so propagation failures can be
identified by disagreement between tracking directions.

Variants:

sequential
    Ordinary point-to-point ICP initialized from the previous pose. This
    isolates the effect of temporal initialization from robust rejection.

robust
    Trimmed observation-to-template point-to-point ICP. A distance gate and
    trim fraction reject poor correspondences, while the one-sided direction
    does not require unseen template regions to appear in a partial cloud.

The ball keeps identity rotation in both variants because its orientation is
not observable from spherical geometry. The first pose in each tracking
direction is initialized from the corresponding Approach 1 trajectory.

Usage:
    python3 approach_3.py
    python3 approach_3.py 01__03 occlusion --variant robust --direction forward
    python3 approach_3.py 01__03 full --variant robust --direction forward
    python3 approach_3.py --self-check
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from tqdm.auto import tqdm
from approach_0 import load_stretches
from approach_1 import (
    BALL_SEQUENCE_ID,
    DEFAULT_CONVERGENCE_TOLERANCE,
    DEFAULT_ICP_SAMPLE_COUNT,
    DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
    DEFAULT_MAX_ITERATIONS,
    TRANSLATION_ONLY_VARIANT,
    fit_stretch as fit_independent_stretch,
    make_point_cloud,
    rigid_icp,
    translation_only_icp,
)
from evaluation import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    evaluate_trajectory,
    nearest_neighbor_distances,
    sample_mesh_surface,
)
from experiment_utils import (
    DEFAULT_RESULTS_ROOT,
    experiment_output_dir,
    prepare_frame_indices,
    save_experiment,
)
from pose_utils import (
    prepare_scales,
    rotation_geodesic,
    transform_points,
    validate_trajectory,
)
from toy_task.load_frame import load_frame, load_sequence, load_template

SEQUENTIAL_VARIANT = "sequential"
ROBUST_VARIANT = "robust"
VARIANTS = (SEQUENTIAL_VARIANT, ROBUST_VARIANT)
DIRECTIONS = ("forward", "backward")
SOURCE_VARIANTS = (
    "single_initialization",
    "multi_initialization",
    TRANSLATION_ONLY_VARIANT,
)
DEFAULT_TRIM_FRACTION = 0.8
DEFAULT_MIN_CORRESPONDENCES = 20

def default_source_variant(sequence_id):
    """Choose the Approach 1 boundary-pose source for one object"""
    if sequence_id == BALL_SEQUENCE_ID:
        return TRANSLATION_ONLY_VARIANT
    return "multi_initialization"

def load_source_trajectory(sequence_id, run_name, source_variant,
                           results_root=DEFAULT_RESULTS_ROOT):
    """Load the matching Approach 1 trajectory used for boundary seeding"""
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
    return {
        "R": rotations,
        "t": translations,
        "s": scales,
        "frame_indices": frame_indices,
        "path": trajectory_path,
    }


def generate_boundary_source(sequence_id, start, end, direction,
                             source_variant, scale=1.0,
                             icp_sample_count=DEFAULT_ICP_SAMPLE_COUNT,
                             sample_seed=DEFAULT_SAMPLE_SEED,
                             max_correspondence_distance=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
                             max_iterations=DEFAULT_MAX_ITERATIONS,
                             convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE):
    """Generate only the Approach 1 boundary pose needed for tracking.

    Approach 3 uses its source trajectory only at the first tracking frame.
    For a complete sequence, fitting Approach 1 independently at every frame
    would therefore be unnecessary. The one fitted boundary pose is repeated
    solely to satisfy the common trajectory interface; subsequent entries are
    never used as pose initializations.
    """
    boundary_frame = start if direction == "forward" else end - 1
    boundary = fit_independent_stretch(
        sequence_id,
        boundary_frame,
        boundary_frame + 1,
        source_variant,
        scale=scale,
        icp_sample_count=icp_sample_count,
        sample_seed=sample_seed,
        max_correspondence_distance=max_correspondence_distance,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
    )
    n_frames = end - start
    return {
        "R": np.repeat(boundary["R"], n_frames, axis=0),
        "t": np.repeat(boundary["t"], n_frames, axis=0),
        "s": np.full(n_frames, float(boundary["s"][0]), dtype=np.float64),
        "frame_indices": np.arange(start, end, dtype=np.int64),
        "path": None,
        "boundary_frame": int(boundary_frame),
    }

def estimate_rigid_transform(source, target):
    """Least-squares rigid transform from paired source to target points"""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            "source and target must have matching shape (N, 3); got "
            f"{source.shape} and {target.shape}"
        )
    if len(source) < 3:
        raise ValueError("at least three paired points are required")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation

def select_trimmed_correspondences(distances, max_distance, trim_fraction,
                                   min_correspondences):
    """Return closest gated correspondence indices and selection counts"""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError(f"distances must be one-dimensional, got {distances.shape}")
    if max_distance <= 0 or not np.isfinite(max_distance):
        raise ValueError("max distance must be finite and positive")
    if not 0 < trim_fraction <= 1:
        raise ValueError("trim fraction must be in (0, 1]")
    if min_correspondences <= 0:
        raise ValueError("minimum correspondences must be positive")

    available = np.flatnonzero(np.isfinite(distances) & (distances <= max_distance))
    if len(available) < min_correspondences:
        return np.empty(0, dtype=np.int64), len(available)

    keep_count = max(min_correspondences, int(np.ceil(trim_fraction * len(available))))
    keep_count = min(keep_count, len(available))
    if keep_count == len(available):
        retained = available
    else:
        local = np.argpartition(distances[available], keep_count - 1)[:keep_count]
        retained = available[local]
    return retained, len(available)

def trimmed_observation_to_template_icp(
        template_samples, observations, initial_rotation, initial_translation,
        scale, max_correspondence_distance, trim_fraction, max_iterations,
        convergence_tolerance, min_correspondences=DEFAULT_MIN_CORRESPONDENCES,
        translation_only=False):
    """Robust ICP using gated, trimmed observation-to-template matches"""
    template_samples = np.asarray(template_samples, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observations = observations[np.isfinite(observations).all(axis=1)]
    if template_samples.ndim != 2 or template_samples.shape[1] != 3:
        raise ValueError(
            f"template samples must have shape (N, 3), got {template_samples.shape}"
        )
    if len(template_samples) < 3 or len(observations) < 3:
        raise ValueError("template and observation clouds need at least three points")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max iterations must be positive")
    if convergence_tolerance <= 0:
        raise ValueError("convergence tolerance must be positive")

    rotation = np.asarray(initial_rotation, dtype=np.float64).copy()
    translation = np.asarray(initial_translation, dtype=np.float64).copy()
    scaled_template = float(scale) * template_samples
    termination = "maximum_iterations"
    retained = np.empty(0, dtype=np.int64)
    available_count = 0
    iteration_count = 0

    for iteration_count in range(1, max_iterations + 1):
        posed_template = scaled_template @ rotation.T + translation
        distances, template_indices = KDTree(posed_template).query(
            observations, workers=-1
        )
        retained, available_count = select_trimmed_correspondences(
            distances,
            max_correspondence_distance,
            trim_fraction,
            min_correspondences,
        )
        if not len(retained):
            termination = "insufficient_correspondences"
            break

        matched_template = scaled_template[template_indices[retained]]
        matched_observations = observations[retained]
        previous_rotation = rotation.copy()
        previous_translation = translation.copy()

        if translation_only:
            posed_matches = matched_template @ rotation.T + translation
            translation += np.mean(matched_observations - posed_matches, axis=0)
        else:
            rotation, translation = estimate_rigid_transform(
                matched_template, matched_observations
            )

        rotation_step_radians = np.radians(
            rotation_geodesic(previous_rotation, rotation)
        )
        translation_step = np.linalg.norm(translation - previous_translation)
        if max(rotation_step_radians, translation_step) <= convergence_tolerance:
            termination = "converged"
            break

    posed_template = scaled_template @ rotation.T + translation
    distances, _ = KDTree(posed_template).query(observations, workers=-1)
    retained, available_count = select_trimmed_correspondences(
        distances,
        max_correspondence_distance,
        trim_fraction,
        min_correspondences,
    )
    retained_rmse = (
        float(np.sqrt(np.mean(distances[retained] ** 2)))
        if len(retained) else float("inf")
    )
    return rotation, translation, {
        "iterations": int(iteration_count),
        "termination": termination,
        "available_correspondences": int(available_count),
        "retained_correspondences": int(len(retained)),
        "inlier_fraction": float(available_count / len(observations)),
        "retained_rmse_m": retained_rmse,
    }

def one_sided_fit_distance(template_samples, observations, rotation,
                           translation, scale):
    """Mean observation-to-posed-template distance in meters"""
    observations = np.asarray(observations, dtype=np.float64)
    observations = observations[np.isfinite(observations).all(axis=1)]
    posed = transform_points(
        template_samples, rotation, translation, scale
    )
    return float(nearest_neighbor_distances(observations, posed).mean())

def fit_stretch(sequence_id, start, end, variant, direction, source,
                icp_sample_count=DEFAULT_ICP_SAMPLE_COUNT,
                sample_seed=DEFAULT_SAMPLE_SEED,
                max_correspondence_distance=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
                trim_fraction=DEFAULT_TRIM_FRACTION,
                max_iterations=DEFAULT_MAX_ITERATIONS,
                convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE,
                min_correspondences=DEFAULT_MIN_CORRESPONDENCES):
    """Track one interval sequentially and return poses in chronological order"""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; choose from {DIRECTIONS}")

    sequence = load_sequence(sequence_id)
    if start < 0 or end <= start or end > sequence["n_frames"]:
        raise ValueError(
            f"invalid interval [{start}, {end}) for {sequence_id} with "
            f"{sequence['n_frames']} frames"
        )
    expected_indices = np.arange(start, end, dtype=np.int64)
    if not np.array_equal(source["frame_indices"], expected_indices):
        raise ValueError(
            "Approach 1 source interval does not match requested interval: "
            f"expected [{start}, {end})"
        )

    vertices, faces = load_template(sequence["template"])
    template_samples = sample_mesh_surface(
        vertices, faces, icp_sample_count, sample_seed
    )
    source_clouds = {
        float(scale): make_point_cloud(float(scale) * template_samples)
        for scale in np.unique(source["s"])
    }
    translation_only = sequence_id == BALL_SEQUENCE_ID
    order = (
        np.arange(len(expected_indices))
        if direction == "forward"
        else np.arange(len(expected_indices) - 1, -1, -1)
    )

    rotations = np.empty((len(expected_indices), 3, 3), dtype=np.float64)
    translations = np.empty((len(expected_indices), 3), dtype=np.float64)
    boundary_index = int(order[0])
    previous_rotation = source["R"][boundary_index].copy()
    previous_translation = source["t"][boundary_index].copy()
    diagnostics = []

    for tracking_order, pose_index in enumerate(tqdm(
            order,
            desc=f"fit {sequence_id} {variant} {direction}",
            unit="frame")):
        frame_index = int(expected_indices[pose_index])
        frame = load_frame(sequence, frame_index, with_image=False)
        observations = frame["points"]
        scale = float(source["s"][pose_index])
        initial_rotation = previous_rotation.copy()
        initial_translation = previous_translation.copy()

        if variant == ROBUST_VARIANT:
            rotation, translation, robust = trimmed_observation_to_template_icp(
                template_samples,
                observations,
                initial_rotation,
                initial_translation,
                scale,
                max_correspondence_distance,
                trim_fraction,
                max_iterations,
                convergence_tolerance,
                min_correspondences=min_correspondences,
                translation_only=translation_only,
            )
            fitness = robust["inlier_fraction"]
            inlier_rmse = robust["retained_rmse_m"]
        elif translation_only:
            rotation, translation, fitness, inlier_rmse = translation_only_icp(
                template_samples,
                observations,
                initial_rotation,
                initial_translation,
                scale,
                max_correspondence_distance,
                max_iterations,
                convergence_tolerance,
            )
            robust = {
                "iterations": "",
                "termination": "open3d_or_translation_icp",
                "available_correspondences": "",
                "retained_correspondences": "",
            }
        else:
            rotation, translation, fitness, inlier_rmse = rigid_icp(
                source_clouds[scale],
                observations,
                initial_rotation,
                initial_translation,
                max_correspondence_distance,
                max_iterations,
                convergence_tolerance,
            )
            robust = {
                "iterations": "",
                "termination": "open3d_or_translation_icp",
                "available_correspondences": "",
                "retained_correspondences": "",
            }

        rotations[pose_index] = rotation
        translations[pose_index] = translation
        diagnostics.append({
            "frame": frame_index,
            "tracking_order": int(tracking_order),
            "direction": direction,
            "variant": variant,
            "boundary_seed": tracking_order == 0,
            "rotation_step_from_initial_deg": rotation_geodesic(
                initial_rotation, rotation
            ),
            "translation_step_from_initial_cm": 100.0 * float(np.linalg.norm(
                translation - initial_translation
            )),
            "observation_to_template_cm": 100.0 * one_sided_fit_distance(
                template_samples,
                observations,
                rotation,
                translation,
                scale,
            ),
            "icp_fitness": float(fitness),
            "icp_inlier_rmse_cm": 100.0 * float(inlier_rmse),
            "iterations": robust["iterations"],
            "termination": robust["termination"],
            "available_correspondences": robust["available_correspondences"],
            "retained_correspondences": robust["retained_correspondences"],
        })
        previous_rotation = rotation
        previous_translation = translation

    validate_trajectory(rotations, translations, source["s"])
    diagnostics.sort(key=lambda row: row["frame"])
    return {
        "R": rotations,
        "t": translations,
        "s": source["s"].copy(),
        "frame_indices": expected_indices,
        "diagnostics": diagnostics,
    }


def save_tracking_diagnostics(output_dir, diagnostics):
    """Save per-frame sequential fitting and rejection information"""
    path = Path(output_dir) / "tracking_diagnostics.csv"
    columns = [
        "frame",
        "tracking_order",
        "direction",
        "variant",
        "boundary_seed",
        "rotation_step_from_initial_deg",
        "translation_step_from_initial_cm",
        "observation_to_template_cm",
        "icp_fitness",
        "icp_inlier_rmse_cm",
        "iterations",
        "termination",
        "available_correspondences",
        "retained_correspondences",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(diagnostics)
    return path

def run_experiment(sequence_id, run_name, interval, variant, direction,
                   source_variant=None, results_root=DEFAULT_RESULTS_ROOT,
                   icp_sample_count=DEFAULT_ICP_SAMPLE_COUNT,
                   sample_seed=DEFAULT_SAMPLE_SEED,
                   max_correspondence_distance=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
                   trim_fraction=DEFAULT_TRIM_FRACTION,
                   max_iterations=DEFAULT_MAX_ITERATIONS,
                   convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE,
                   min_correspondences=DEFAULT_MIN_CORRESPONDENCES,
                   generate_source_boundary=False,
                   evaluate=True, evaluation_sample_count=DEFAULT_SAMPLE_COUNT,
                   evaluation_sample_seed=DEFAULT_SAMPLE_SEED):
    """Fit, save, and optionally evaluate one Approach 3 experiment."""
    if source_variant is None:
        source_variant = default_source_variant(sequence_id)
    if sequence_id == BALL_SEQUENCE_ID and source_variant != TRANSLATION_ONLY_VARIANT:
        raise ValueError(
            "the ball must use the translation_only Approach 1 source so its "
            "unobservable orientation remains fixed"
        )

    start, end = interval
    if generate_source_boundary:
        source = generate_boundary_source(
            sequence_id,
            start,
            end,
            direction,
            source_variant,
            icp_sample_count=icp_sample_count,
            sample_seed=sample_seed,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
        )
    else:
        source = load_source_trajectory(
            sequence_id, run_name, source_variant, results_root
        )
    trajectory = fit_stretch(
        sequence_id,
        start,
        end,
        variant,
        direction,
        source,
        icp_sample_count=icp_sample_count,
        sample_seed=sample_seed,
        max_correspondence_distance=max_correspondence_distance,
        trim_fraction=trim_fraction,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        min_correspondences=min_correspondences,
    )
    output_variant = f"{variant}_{direction}"
    robust = variant == ROBUST_VARIANT
    translation_only = sequence_id == BALL_SEQUENCE_ID
    parameters = {
        "source_approach": "approach_1",
        "source_variant": source_variant,
        "source_trajectory": (
            str(source["path"].resolve()) if source["path"] is not None else None
        ),
        "source_boundary_frame": int(source.get("boundary_frame", start)),
        "source_boundary_generated_in_memory": bool(generate_source_boundary),
        "source_usage": "boundary pose only",
        "sequential_initialization": True,
        "tracking_direction": direction,
        "icp_type": (
            "translation-only point-to-point"
            if translation_only else "rigid point-to-point"
        ),
        "correspondence_direction": (
            "observation-to-template" if robust else "template-to-observation"
        ),
        "robust_rejection": "distance gate followed by trimming" if robust else "none",
        "trim_fraction": float(trim_fraction) if robust else None,
        "minimum_correspondences": int(min_correspondences) if robust else None,
        "template_surface_samples": int(icp_sample_count),
        "sample_seed": int(sample_seed),
        "max_correspondence_distance_m": float(max_correspondence_distance),
        "max_iterations": int(max_iterations),
        "convergence_tolerance": float(convergence_tolerance),
        "orientation_policy": (
            "fixed identity" if translation_only else "estimated by sequential rigid ICP"
        ),
    }
    output_dir = save_experiment(
        "approach_3",
        output_variant,
        sequence_id,
        run_name,
        trajectory["R"],
        trajectory["t"],
        trajectory["s"],
        trajectory["frame_indices"],
        parameters=parameters,
        results_root=results_root,
    )
    diagnostics_path = save_tracking_diagnostics(
        output_dir, trajectory["diagnostics"]
    )
    print(
        f"saved Approach 3 {sequence_id}/{run_name} {output_variant} "
        f"-> {output_dir}"
    )
    print(f"tracking diagnostics: {diagnostics_path}")

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
    """Check rigid recovery, trimming, fixed rotation, and valid outputs"""
    rng = np.random.default_rng(31)
    template = rng.normal(size=(1_000, 3)) * np.array([0.25, 0.12, 0.06])
    angle = np.radians(12.0)
    true_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    true_translation = np.array([0.18, -0.07, 2.1])

    paired_target = transform_points(
        template, true_rotation, true_translation
    )
    estimated_rotation, estimated_translation = estimate_rigid_transform(
        template, paired_target
    )
    np.testing.assert_allclose(estimated_rotation, true_rotation, atol=1e-12)
    np.testing.assert_allclose(estimated_translation, true_translation, atol=1e-12)

    noisy_observations = paired_target + rng.normal(scale=2e-4, size=paired_target.shape)
    outliers = rng.uniform(
        low=np.array([1.0, 1.0, 3.0]),
        high=np.array([1.5, 1.5, 3.5]),
        size=(250, 3),
    )
    observations = np.vstack((noisy_observations, outliers))
    initial_translation = true_translation + np.array([0.006, -0.004, 0.003])
    rotation, translation, diagnostics = trimmed_observation_to_template_icp(
        template,
        observations,
        true_rotation,
        initial_translation,
        scale=1.0,
        max_correspondence_distance=0.05,
        trim_fraction=0.9,
        max_iterations=50,
        convergence_tolerance=1e-8,
    )
    assert rotation_geodesic(rotation, true_rotation) < 0.1
    assert np.linalg.norm(translation - true_translation) < 1e-3
    assert diagnostics["available_correspondences"] == len(template)
    assert diagnostics["retained_correspondences"] == 900

    fixed_rotation, fixed_translation, _ = trimmed_observation_to_template_icp(
        template,
        observations,
        np.eye(3),
        true_translation,
        scale=1.0,
        max_correspondence_distance=0.5,
        trim_fraction=0.8,
        max_iterations=5,
        convergence_tolerance=1e-8,
        translation_only=True,
    )
    np.testing.assert_allclose(fixed_rotation, np.eye(3), atol=0.0)
    validate_trajectory(
        np.stack((rotation, fixed_rotation)),
        np.stack((translation, fixed_translation)),
        np.ones(2),
    )
    print("Approach 3 self-check passed.")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("run_name", nargs="?")
    parser.add_argument(
        "--variant",
        choices=("sequential", "robust", "both"),
        default="both",
        help="Approach 3 fitting variant to run (default: both)",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
        help="tracking direction to run (default: both)",
    )
    parser.add_argument(
        "--source-variant",
        choices=("auto",) + SOURCE_VARIANTS,
        default="auto",
        help="Approach 1 boundary-pose source (default: object-specific)",
    )
    parser.add_argument("--icp-samples", type=int, default=DEFAULT_ICP_SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
    )
    parser.add_argument("--trim-fraction", type=float, default=DEFAULT_TRIM_FRACTION)
    parser.add_argument(
        "--minimum-correspondences",
        type=int,
        default=DEFAULT_MIN_CORRESPONDENCES,
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE,
    )
    parser.add_argument("--evaluation-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if (args.sequence_id is None) != (args.run_name is None):
        parser.error("provide both sequence_id and run_name, or neither")
    if not 0 < args.trim_fraction <= 1:
        parser.error("--trim-fraction must be in (0, 1]")
    if args.minimum_correspondences <= 0:
        parser.error("--minimum-correspondences must be positive")
    return args

def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return

    stretches = load_stretches()
    if args.sequence_id is None:
        selected = [
            (sequence_id, run_name, interval)
            for sequence_id, named_ranges in stretches.items()
            for run_name, interval in named_ranges.items()
        ]
    else:
        if args.run_name == "full":
            sequence = load_sequence(args.sequence_id)
            selected = [(
                args.sequence_id,
                "full",
                (0, sequence["n_frames"]),
            )]
        elif args.sequence_id not in stretches:
            raise ValueError(f"no selected stretches for {args.sequence_id}")
        elif args.run_name not in stretches[args.sequence_id]:
            available = ", ".join(stretches[args.sequence_id])
            raise ValueError(
                f"no {args.run_name!r} stretch for {args.sequence_id}; "
                f"choose from {available}, full"
            )
        else:
            selected = [(
                args.sequence_id,
                args.run_name,
                stretches[args.sequence_id][args.run_name],
            )]

    variants = VARIANTS if args.variant == "both" else (args.variant,)
    directions = DIRECTIONS if args.direction == "both" else (args.direction,)
    for sequence_id, run_name, interval in selected:
        source_variant = (
            default_source_variant(sequence_id)
            if args.source_variant == "auto" else args.source_variant
        )
        for variant in variants:
            for direction in directions:
                run_experiment(
                    sequence_id,
                    run_name,
                    interval,
                    variant,
                    direction,
                    source_variant=source_variant,
                    results_root=args.results_root,
                    icp_sample_count=args.icp_samples,
                    sample_seed=args.sample_seed,
                    max_correspondence_distance=args.max_correspondence_distance,
                    trim_fraction=args.trim_fraction,
                    max_iterations=args.max_iterations,
                    convergence_tolerance=args.convergence_tolerance,
                    min_correspondences=args.minimum_correspondences,
                    generate_source_boundary=run_name == "full",
                    evaluate=not args.skip_evaluation,
                    evaluation_sample_count=args.evaluation_samples,
                    evaluation_sample_seed=args.evaluation_seed,
                )

if __name__ == "__main__":
    main()
