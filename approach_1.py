"""Approach 1: independent per-frame ICP.

Variants:

1A single_initialization
    Start from one PCA-plus-centroid pose, then refine rotation and translation
    with rigid point-to-point ICP.

1B multi_initialization
    Run rigid ICP from all 24 proper signed/permuted PCA alignments and retain
    the pose with the lowest symmetric Chamfer distance.

``translation_only`` reference
    Keep R equal to identity and refine only translation. This remains
    available for the ball comparison because its orientation is unobservable.

Every frame is fitted independently. No previous-frame pose or temporal cost
is used in this approach.

Usage:
    python3 approach_1.py 01__01 clear --variant both
    python3 approach_1.py

The all-stretches command also writes the ball's ``translation_only``
reference so it appears beside the two rigid-ICP variants for comparison.
"""

from argparse import ArgumentParser
import csv
from itertools import permutations, product
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm.auto import tqdm
from approach_0 import load_stretches, mesh_surface_centroid, point_centroid, principal_axes
from evaluation import (DEFAULT_SAMPLE_COUNT, DEFAULT_SAMPLE_SEED,
                        evaluate_trajectory, point_cloud_fit_metrics,
                        sample_mesh_surface)
from experiment_utils import DEFAULT_RESULTS_ROOT, save_experiment
from pose_utils import transform_points
from toy_task.load_frame import load_frame, load_sequence, load_template

PRIMARY_VARIANTS = ("single_initialization", "multi_initialization")
TRANSLATION_ONLY_VARIANT = "translation_only"
VARIANTS = PRIMARY_VARIANTS + (TRANSLATION_ONLY_VARIANT,)
BALL_SEQUENCE_ID = "01__03"
DEFAULT_ICP_SAMPLE_COUNT = 5_000
DEFAULT_MAX_CORRESPONDENCE_DISTANCE = 0.10
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_CONVERGENCE_TOLERANCE = 1e-6

def proper_signed_permutations():
    """Return the 24 orientation-preserving signed permutation matrices"""
    candidates = []
    for permutation in permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = permutation_matrix @ np.diag(signs)
            if np.linalg.det(matrix) > 0:
                candidates.append(matrix)

    identity = np.eye(3)
    candidates.sort(
        key=lambda matrix: (
            0 if np.array_equal(matrix, identity) else 1,
            tuple(matrix.ravel()),
        )
    )
    return tuple(candidates)

PCA_ORIENTATION_CANDIDATES = proper_signed_permutations()

def make_point_cloud(points):
    """Convert finite NumPy points to an Open3D point cloud"""
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("point cloud has no finite points")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud

def rigid_icp(source_cloud, observations, initial_rotation,
              initial_translation, max_correspondence_distance,
              max_iterations, convergence_tolerance):
    """Run point-to-point rigid ICP and return R, t, fitness, and inlier RMSE"""
    initial_transform = np.eye(4)
    initial_transform[:3, :3] = initial_rotation
    initial_transform[:3, 3] = initial_translation

    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        make_point_cloud(observations),
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=convergence_tolerance,
            relative_rmse=convergence_tolerance,
            max_iteration=max_iterations,
        ),
    )
    transform = np.asarray(result.transformation)
    return (
        transform[:3, :3],
        transform[:3, 3],
        float(result.fitness),
        float(result.inlier_rmse),
    )

def translation_only_icp(template_samples, observations, rotation,
                         initial_translation, scale,
                         max_correspondence_distance, max_iterations,
                         convergence_tolerance):
    """Refine translation while keeping orientation fixed

    For spherical ball so noise cannot create meaningless
    frame-to-frame rotations
    """
    observations = np.asarray(observations, dtype=np.float64)
    observations = observations[np.isfinite(observations).all(axis=1)]
    tree = KDTree(observations)
    translation = np.asarray(initial_translation, dtype=np.float64).copy()
    inliers = np.zeros(len(template_samples), dtype=bool)
    distances = np.full(len(template_samples), np.inf)

    for _ in range(max_iterations):
        posed = transform_points(template_samples, rotation, translation, scale)
        distances, indices = tree.query(posed, workers=-1)
        inliers = distances <= max_correspondence_distance
        if inliers.sum() < 3:
            break
        update = np.mean(observations[indices[inliers]] - posed[inliers], axis=0)
        translation += update
        if np.linalg.norm(update) <= convergence_tolerance:
            break

    posed = transform_points(template_samples, rotation, translation, scale)
    distances, _ = tree.query(posed, workers=-1)
    inliers = distances <= max_correspondence_distance
    fitness = float(inliers.mean())
    inlier_rmse = (
        float(np.sqrt(np.mean(distances[inliers] ** 2)))
        if inliers.any() else float("inf")
    )
    return rotation.copy(), translation, fitness, inlier_rmse

def candidate_rotations(variant, template_axes, observations):
    """Return initial rotations for one frame and Approach 1 variant"""
    if variant == TRANSLATION_ONLY_VARIANT:
        return [(np.eye(3), 0)]

    observation_axes = principal_axes(observations)
    orientation_candidates = (
        PCA_ORIENTATION_CANDIDATES[:1]
        if variant == "single_initialization"
        else PCA_ORIENTATION_CANDIDATES
    )
    return [
        (observation_axes @ candidate @ template_axes.T, candidate_index)
        for candidate_index, candidate in enumerate(orientation_candidates)
    ]

def fit_stretch(sequence_id, start, end, variant, scale=1.0,
                icp_sample_count=DEFAULT_ICP_SAMPLE_COUNT,
                sample_seed=DEFAULT_SAMPLE_SEED,
                max_correspondence_distance=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
                max_iterations=DEFAULT_MAX_ITERATIONS,
                convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE):
    """Fit one Approach 1 variant over interval [start, end)"""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if variant == TRANSLATION_ONLY_VARIANT and sequence_id != BALL_SEQUENCE_ID:
        raise ValueError("translation_only is retained only for the ball sequence 01__03")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"scale must be finite and positive, got {scale}")
    if max_correspondence_distance <= 0:
        raise ValueError("max correspondence distance must be positive")
    if max_iterations <= 0:
        raise ValueError("max iterations must be positive")
    if convergence_tolerance <= 0:
        raise ValueError("convergence tolerance must be positive")

    sequence = load_sequence(sequence_id)
    if start < 0 or end <= start or end > sequence["n_frames"]:
        raise ValueError(
            f"invalid interval [{start}, {end}) for {sequence_id} with "
            f"{sequence['n_frames']} frames"
        )

    vertices, faces = load_template(sequence["template"])
    template_samples = sample_mesh_surface(vertices, faces, icp_sample_count, sample_seed)
    template_centroid = mesh_surface_centroid(vertices, faces)
    template_axes = principal_axes(template_samples)
    scaled_source_cloud = make_point_cloud(float(scale) * template_samples)

    frame_indices = np.arange(start, end, dtype=np.int64)
    rotations = []
    translations = []
    diagnostics = []

    for frame_index in tqdm(
        frame_indices,
        desc=f"fit {sequence_id} {variant}",
        unit="frame",
    ):
        frame = load_frame(sequence, int(frame_index), with_image=False)
        observations = frame["points"]
        candidates = candidate_rotations(variant, template_axes, observations)
        best = None

        for initial_rotation, candidate_index in candidates:
            initial_translation = point_centroid(observations) - float(scale) * (
                initial_rotation @ template_centroid
            )

            if variant == TRANSLATION_ONLY_VARIANT:
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
            else:
                rotation, translation, fitness, inlier_rmse = rigid_icp(
                    scaled_source_cloud,
                    observations,
                    initial_rotation,
                    initial_translation,
                    max_correspondence_distance,
                    max_iterations,
                    convergence_tolerance,
                )

            posed_samples = transform_points(template_samples, rotation, translation, scale)
            selection_chamfer = point_cloud_fit_metrics(posed_samples, observations)["chamfer_m"]
            candidate_result = {
                "rotation": rotation,
                "translation": translation,
                "candidate_index": candidate_index,
                "candidate_count": len(candidates),
                "selection_chamfer_m": selection_chamfer,
                "icp_fitness": fitness,
                "icp_inlier_rmse_m": inlier_rmse,
            }
            if best is None or selection_chamfer < best["selection_chamfer_m"]:
                best = candidate_result

        rotations.append(best["rotation"])
        translations.append(best["translation"])
        diagnostics.append({
            "frame": int(frame_index),
            "candidate_count": int(best["candidate_count"]),
            "selected_candidate": int(best["candidate_index"]),
            "selection_chamfer_cm": 100.0 * best["selection_chamfer_m"],
            "icp_fitness": best["icp_fitness"],
            "icp_inlier_rmse_cm": 100.0 * best["icp_inlier_rmse_m"],
        })

    return {
        "R": np.asarray(rotations, dtype=np.float64),
        "t": np.asarray(translations, dtype=np.float64),
        "s": np.full(len(frame_indices), float(scale), dtype=np.float64),
        "frame_indices": frame_indices,
        "diagnostics": diagnostics,
    }

def save_fit_diagnostics(output_dir, diagnostics):
    """Save per-frame ICP candidate-selection information"""
    path = Path(output_dir) / "fit_diagnostics.csv"
    columns = [
        "frame",
        "candidate_count",
        "selected_candidate",
        "selection_chamfer_cm",
        "icp_fitness",
        "icp_inlier_rmse_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(diagnostics)
    return path

def run_experiment(sequence_id, run_name, interval, variant,
                   results_root=DEFAULT_RESULTS_ROOT, scale=1.0,
                   icp_sample_count=DEFAULT_ICP_SAMPLE_COUNT,
                   sample_seed=DEFAULT_SAMPLE_SEED,
                   max_correspondence_distance=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
                   max_iterations=DEFAULT_MAX_ITERATIONS,
                   convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE,
                   evaluate=True, evaluation_sample_count=DEFAULT_SAMPLE_COUNT,
                   evaluation_sample_seed=DEFAULT_SAMPLE_SEED):
    """Fit, save, and optionally evaluate one selected stretch"""
    start, end = interval
    trajectory = fit_stretch(
        sequence_id,
        start,
        end,
        variant,
        scale=scale,
        icp_sample_count=icp_sample_count,
        sample_seed=sample_seed,
        max_correspondence_distance=max_correspondence_distance,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
    )
    translation_only = variant == TRANSLATION_ONLY_VARIANT
    parameters = {
        "independent_per_frame": True,
        "scale": float(scale),
        "icp_type": (
            "translation-only point-to-point"
            if translation_only else "rigid point-to-point"
        ),
        "initialization": (
            "fixed orientation plus centroid"
            if translation_only else "PCA orientation plus centroid"
        ),
        "orientation_candidate_count": (
            1 if translation_only or variant == "single_initialization" else 24
        ),
        "candidate_selection": "lowest symmetric Chamfer distance",
        "template_surface_samples": int(icp_sample_count),
        "sample_seed": int(sample_seed),
        "max_correspondence_distance_m": float(max_correspondence_distance),
        "max_iterations": int(max_iterations),
        "convergence_tolerance": float(convergence_tolerance),
        "orientation_policy": (
            "fixed identity" if translation_only else "estimated by rigid ICP"
        ),
        "uses_previous_frame": False,
    }
    output_dir = save_experiment(
        "approach_1",
        variant,
        sequence_id,
        run_name,
        trajectory["R"],
        trajectory["t"],
        trajectory["s"],
        trajectory["frame_indices"],
        parameters=parameters,
        results_root=results_root,
    )
    diagnostics_path = save_fit_diagnostics(
        output_dir, trajectory["diagnostics"]
    )
    print(
        f"saved {variant} {sequence_id}/{run_name} "
        f"frames [{start}, {end}) -> {output_dir}"
    )
    print(f"ICP diagnostics: {diagnostics_path}")

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
    """Check candidate generation and ICP on a controlled rigid transform."""
    assert len(PCA_ORIENTATION_CANDIDATES) == 24
    for candidate in PCA_ORIENTATION_CANDIDATES:
        np.testing.assert_allclose(candidate.T @ candidate, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(candidate), 1.0, atol=1e-12)

    rng = np.random.default_rng(12)
    template = rng.normal(size=(1_000, 3)) * np.array([0.3, 0.15, 0.05])
    true_rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    true_translation = np.array([0.2, -0.1, 2.0])
    observations = transform_points(
        template, true_rotation, true_translation
    )
    estimated_rotation, estimated_translation, fitness, rmse = rigid_icp(
        make_point_cloud(template),
        observations,
        true_rotation,
        true_translation + np.array([0.01, -0.01, 0.005]),
        max_correspondence_distance=0.1,
        max_iterations=50,
        convergence_tolerance=1e-8,
    )
    np.testing.assert_allclose(estimated_rotation, true_rotation, atol=1e-6)
    np.testing.assert_allclose(estimated_translation, true_translation, atol=1e-6)
    np.testing.assert_allclose(fitness, 1.0, atol=1e-12)
    np.testing.assert_allclose(rmse, 0.0, atol=1e-6)
    print("Approach 1 self-check passed.")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("run_name", nargs="?")
    parser.add_argument(
        "--variant",
        choices=("single", "multi", "translation", "both"),
        default="both",
        help="Approach 1 variant to run (default: both)",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--icp-samples", type=int, default=DEFAULT_ICP_SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--max-correspondence-distance", type=float,
                        default=DEFAULT_MAX_CORRESPONDENCE_DISTANCE)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--convergence-tolerance", type=float,
                        default=DEFAULT_CONVERGENCE_TOLERANCE)
    parser.add_argument("--evaluation-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if (args.sequence_id is None) != (args.run_name is None):
        parser.error("provide both sequence_id and run_name, or neither")
    if args.variant == "translation" and args.sequence_id is None:
        parser.error("--variant translation requires sequence_id and run_name")
    if args.variant == "translation" and args.sequence_id != BALL_SEQUENCE_ID:
        parser.error("--variant translation is only defined for ball sequence 01__03")
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
        if args.sequence_id not in stretches:
            raise ValueError(f"no selected stretches for {args.sequence_id}")
        if args.run_name not in stretches[args.sequence_id]:
            available = ", ".join(stretches[args.sequence_id])
            raise ValueError(
                f"no {args.run_name!r} stretch for {args.sequence_id}; "
                f"choose from {available}"
            )
        selected = [(
            args.sequence_id,
            args.run_name,
            stretches[args.sequence_id][args.run_name],
        )]

    variants = {
        "single": ("single_initialization",),
        "multi": ("multi_initialization",),
        "translation": (TRANSLATION_ONLY_VARIANT,),
        "both": PRIMARY_VARIANTS,
    }[args.variant]
    for sequence_id, run_name, interval in selected:
        run_variants = variants
        if (args.sequence_id is None and args.variant == "both"
                and sequence_id == BALL_SEQUENCE_ID):
            run_variants = variants + (TRANSLATION_ONLY_VARIANT,)
        for variant in run_variants:
            run_experiment(
                sequence_id,
                run_name,
                interval,
                variant,
                results_root=args.results_root,
                scale=args.scale,
                icp_sample_count=args.icp_samples,
                sample_seed=args.sample_seed,
                max_correspondence_distance=args.max_correspondence_distance,
                max_iterations=args.max_iterations,
                convergence_tolerance=args.convergence_tolerance,
                evaluate=not args.skip_evaluation,
                evaluation_sample_count=args.evaluation_samples,
                evaluation_sample_seed=args.evaluation_seed,
            )

if __name__ == "__main__":
    main()
