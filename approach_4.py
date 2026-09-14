"""Approach 4: pose candidates plus dynamic-programming path selection.

For every frame, generate the same ICP-refined orientation candidates used by
Approach 1B. Then produce two controlled outputs from that one candidate set:

independent_candidates
    Select the lowest-Chamfer candidate independently in every frame. (= 1B)

temporal_path
    Use dynamic programming to minimize data cost plus rotation and
    translation transition costs over the complete stretch.

The ball has one fixed-orientation, translation-only candidate because its
rotation is not observable. Its two outputs are therefore intentionally
identical and serve as a control.

Usage:
    python3 approach_4.py
    python3 approach_4.py 01__01 motion --variant both
    python3 approach_4.py --self-check
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from approach_0 import (
    load_stretches,
    mesh_surface_centroid,
    point_centroid,
    principal_axes,
)
from approach_1 import (
    BALL_SEQUENCE_ID,
    DEFAULT_CONVERGENCE_TOLERANCE,
    DEFAULT_ICP_SAMPLE_COUNT,
    DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
    DEFAULT_MAX_ITERATIONS,
    candidate_rotations,
    make_point_cloud,
    rigid_icp,
    translation_only_icp,
)
from evaluation import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    evaluate_trajectory,
    point_cloud_fit_metrics,
    sample_mesh_surface,
)
from experiment_utils import DEFAULT_RESULTS_ROOT, save_experiment
from pose_utils import rotation_geodesic, transform_points, validate_trajectory
from toy_task.load_frame import load_frame, load_sequence, load_template


INDEPENDENT_VARIANT = "independent_candidates"
TEMPORAL_VARIANT = "temporal_path"
VARIANTS = (INDEPENDENT_VARIANT, TEMPORAL_VARIANT)

# Costs are dimensionless. Chamfer is divided by the template diagonal, while
# transitions are divided by interpretable motion scales before squaring.
DEFAULT_ROTATION_TRANSITION_WEIGHT = 0.01
DEFAULT_TRANSLATION_TRANSITION_WEIGHT = 0.01
DEFAULT_ROTATION_SCALE_DEGREES = 30.0
DEFAULT_TRANSLATION_SCALE_METERS = 0.10

def validate_cost_parameters(rotation_weight, translation_weight,
                             rotation_scale_degrees,
                             translation_scale_meters):
    """Validate nonnegative weights and positive normalization scales."""
    if not np.isfinite(rotation_weight) or rotation_weight < 0:
        raise ValueError("rotation transition weight must be finite and nonnegative")
    if not np.isfinite(translation_weight) or translation_weight < 0:
        raise ValueError("translation transition weight must be finite and nonnegative")
    if not np.isfinite(rotation_scale_degrees) or rotation_scale_degrees <= 0:
        raise ValueError("rotation scale must be finite and positive")
    if not np.isfinite(translation_scale_meters) or translation_scale_meters <= 0:
        raise ValueError("translation scale must be finite and positive")

def generate_candidates(sequence_id, start, end, scale=1.0,
                        icp_sample_count=DEFAULT_ICP_SAMPLE_COUNT,
                        sample_seed=DEFAULT_SAMPLE_SEED,
                        max_correspondence_distance=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
                        max_iterations=DEFAULT_MAX_ITERATIONS,
                        convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE):
    """Generate one deterministic ICP candidate set for interval [start, end)"""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"scale must be finite and positive, got {scale}")
    if icp_sample_count <= 0:
        raise ValueError("ICP sample count must be positive")
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
    template_samples = sample_mesh_surface(
        vertices, faces, icp_sample_count, sample_seed
    )
    template_centroid = mesh_surface_centroid(vertices, faces)
    template_axes = principal_axes(template_samples)
    template_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if template_diagonal <= 0 or not np.isfinite(template_diagonal):
        raise ValueError("template must have a finite, positive diagonal")

    frame_indices = np.arange(start, end, dtype=np.int64)
    translation_only = sequence_id == BALL_SEQUENCE_ID
    candidate_count = 1 if translation_only else 24
    rotations = np.empty(
        (len(frame_indices), candidate_count, 3, 3), dtype=np.float64
    )
    translations = np.empty(
        (len(frame_indices), candidate_count, 3), dtype=np.float64
    )
    chamfer_m = np.empty((len(frame_indices), candidate_count), dtype=np.float64)
    fitness = np.empty_like(chamfer_m)
    inlier_rmse_m = np.empty_like(chamfer_m)
    scaled_source_cloud = make_point_cloud(float(scale) * template_samples)

    for pose_index, frame_index in enumerate(tqdm(
            frame_indices,
            desc=f"generate {sequence_id} candidates",
            unit="frame")):
        frame = load_frame(sequence, int(frame_index), with_image=False)
        observations = frame["points"]
        if translation_only:
            initial_candidates = [(np.eye(3), 0)]
        else:
            initial_candidates = candidate_rotations(
                "multi_initialization", template_axes, observations
            )

        if len(initial_candidates) != candidate_count:
            raise RuntimeError(
                f"expected {candidate_count} candidates, got "
                f"{len(initial_candidates)}"
            )

        for initial_rotation, candidate_index in initial_candidates:
            initial_translation = point_centroid(observations) - float(scale) * (
                initial_rotation @ template_centroid
            )
            if translation_only:
                rotation, translation, candidate_fitness, candidate_rmse = (
                    translation_only_icp(
                        template_samples,
                        observations,
                        initial_rotation,
                        initial_translation,
                        scale,
                        max_correspondence_distance,
                        max_iterations,
                        convergence_tolerance,
                    )
                )
            else:
                rotation, translation, candidate_fitness, candidate_rmse = rigid_icp(
                    scaled_source_cloud,
                    observations,
                    initial_rotation,
                    initial_translation,
                    max_correspondence_distance,
                    max_iterations,
                    convergence_tolerance,
                )

            posed_samples = transform_points(
                template_samples, rotation, translation, scale
            )
            candidate_chamfer = point_cloud_fit_metrics(
                posed_samples, observations
            )["chamfer_m"]
            rotations[pose_index, candidate_index] = rotation
            translations[pose_index, candidate_index] = translation
            chamfer_m[pose_index, candidate_index] = candidate_chamfer
            fitness[pose_index, candidate_index] = candidate_fitness
            inlier_rmse_m[pose_index, candidate_index] = candidate_rmse

    return {
        "R": rotations,
        "t": translations,
        "chamfer_m": chamfer_m,
        "data_cost": chamfer_m / template_diagonal,
        "fitness": fitness,
        "inlier_rmse_m": inlier_rmse_m,
        "s": np.full(len(frame_indices), float(scale), dtype=np.float64),
        "frame_indices": frame_indices,
        "template_diagonal_m": template_diagonal,
        "orientation_policy": (
            "fixed identity" if translation_only else "24 proper PCA orientations"
        ),
    }

def transition_cost(previous_rotation, previous_translation,
                    current_rotation, current_translation,
                    rotation_weight=DEFAULT_ROTATION_TRANSITION_WEIGHT,
                    translation_weight=DEFAULT_TRANSLATION_TRANSITION_WEIGHT,
                    rotation_scale_degrees=DEFAULT_ROTATION_SCALE_DEGREES,
                    translation_scale_meters=DEFAULT_TRANSLATION_SCALE_METERS):
    """Return dimensionless motion cost and its physical components"""
    validate_cost_parameters(
        rotation_weight,
        translation_weight,
        rotation_scale_degrees,
        translation_scale_meters,
    )
    rotation_change = rotation_geodesic(previous_rotation, current_rotation)
    translation_change = float(np.linalg.norm(
        np.asarray(current_translation) - np.asarray(previous_translation)
    ))
    cost = (
        float(rotation_weight)
        * (rotation_change / float(rotation_scale_degrees)) ** 2
        + float(translation_weight)
        * (translation_change / float(translation_scale_meters)) ** 2
    )
    return float(cost), rotation_change, translation_change

def independent_path(data_costs):
    """Select the lowest-data-cost candidate independently per frame"""
    data_costs = np.asarray(data_costs, dtype=np.float64)
    if data_costs.ndim != 2 or not data_costs.shape[0] or not data_costs.shape[1]:
        raise ValueError(f"data costs must have nonempty shape (T, C), got {data_costs.shape}")
    if not np.isfinite(data_costs).all():
        raise ValueError("data costs must be finite")
    return np.argmin(data_costs, axis=1).astype(np.int64)

def temporal_path(candidate_rotations, candidate_translations, data_costs,
                  rotation_weight=DEFAULT_ROTATION_TRANSITION_WEIGHT,
                  translation_weight=DEFAULT_TRANSLATION_TRANSITION_WEIGHT,
                  rotation_scale_degrees=DEFAULT_ROTATION_SCALE_DEGREES,
                  translation_scale_meters=DEFAULT_TRANSLATION_SCALE_METERS,
                  description="select temporal path"):
    """Find the globally lowest-cost first-order path by dynamic programming"""
    candidate_rotations = np.asarray(candidate_rotations, dtype=np.float64)
    candidate_translations = np.asarray(candidate_translations, dtype=np.float64)
    data_costs = np.asarray(data_costs, dtype=np.float64)
    if candidate_rotations.ndim != 4 or candidate_rotations.shape[2:] != (3, 3):
        raise ValueError(
            "candidate rotations must have shape (T, C, 3, 3), got "
            f"{candidate_rotations.shape}"
        )
    n_frames, candidate_count = candidate_rotations.shape[:2]
    if candidate_translations.shape != (n_frames, candidate_count, 3):
        raise ValueError(
            "candidate translations must have shape "
            f"({n_frames}, {candidate_count}, 3), got {candidate_translations.shape}"
        )
    if data_costs.shape != (n_frames, candidate_count):
        raise ValueError(
            f"data costs must have shape ({n_frames}, {candidate_count}), "
            f"got {data_costs.shape}"
        )
    if not np.isfinite(data_costs).all():
        raise ValueError("data costs must be finite")
    validate_cost_parameters(
        rotation_weight,
        translation_weight,
        rotation_scale_degrees,
        translation_scale_meters,
    )

    accumulated = np.full((n_frames, candidate_count), np.inf, dtype=np.float64)
    backpointers = np.full((n_frames, candidate_count), -1, dtype=np.int64)
    accumulated[0] = data_costs[0]

    for frame_index in tqdm(
            range(1, n_frames), desc=description, unit="frame"):
        for current_index in range(candidate_count):
            costs = np.empty(candidate_count, dtype=np.float64)
            for previous_index in range(candidate_count):
                motion_cost, _, _ = transition_cost(
                    candidate_rotations[frame_index - 1, previous_index],
                    candidate_translations[frame_index - 1, previous_index],
                    candidate_rotations[frame_index, current_index],
                    candidate_translations[frame_index, current_index],
                    rotation_weight,
                    translation_weight,
                    rotation_scale_degrees,
                    translation_scale_meters,
                )
                costs[previous_index] = (
                    accumulated[frame_index - 1, previous_index] + motion_cost
                )
            best_previous = int(np.argmin(costs))
            backpointers[frame_index, current_index] = best_previous
            accumulated[frame_index, current_index] = (
                data_costs[frame_index, current_index] + costs[best_previous]
            )

    path = np.empty(n_frames, dtype=np.int64)
    path[-1] = int(np.argmin(accumulated[-1]))
    for frame_index in range(n_frames - 1, 0, -1):
        path[frame_index - 1] = backpointers[frame_index, path[frame_index]]
    return path, accumulated, backpointers

def selected_trajectory(candidates, path):
    """Extract one chronological trajectory from a candidate-index pat."""
    path = np.asarray(path, dtype=np.int64)
    n_frames, candidate_count = candidates["data_cost"].shape
    if path.shape != (n_frames,):
        raise ValueError(f"path must have shape ({n_frames},), got {path.shape}")
    if np.any(path < 0) or np.any(path >= candidate_count):
        raise ValueError("path contains an invalid candidate index")
    frame_rows = np.arange(n_frames)
    rotations = candidates["R"][frame_rows, path]
    translations = candidates["t"][frame_rows, path]
    validate_trajectory(rotations, translations, candidates["s"])
    return {
        "R": rotations,
        "t": translations,
        "s": candidates["s"].copy(),
        "frame_indices": candidates["frame_indices"].copy(),
    }

def path_diagnostics(candidates, path, rotation_weight, translation_weight,
                     rotation_scale_degrees, translation_scale_meters):
    """Describe selected candidate costs and transitions for every frame"""
    path = np.asarray(path, dtype=np.int64)
    rows = []
    cumulative_cost = 0.0
    previous_rotation = None
    previous_translation = None
    for pose_index, candidate_index in enumerate(path):
        candidate_index = int(candidate_index)
        data_cost = float(candidates["data_cost"][pose_index, candidate_index])
        transition = 0.0
        rotation_change = 0.0
        translation_change = 0.0
        rotation = candidates["R"][pose_index, candidate_index]
        translation = candidates["t"][pose_index, candidate_index]
        if previous_rotation is not None:
            transition, rotation_change, translation_change = transition_cost(
                previous_rotation,
                previous_translation,
                rotation,
                translation,
                rotation_weight,
                translation_weight,
                rotation_scale_degrees,
                translation_scale_meters,
            )
        cumulative_cost += data_cost + transition
        rows.append({
            "frame": int(candidates["frame_indices"][pose_index]),
            "selected_candidate": candidate_index,
            "selection_chamfer_cm": 100.0 * float(
                candidates["chamfer_m"][pose_index, candidate_index]
            ),
            "normalized_data_cost": data_cost,
            "rotation_change_deg": rotation_change,
            "translation_change_cm": 100.0 * translation_change,
            "transition_cost": transition,
            "cumulative_objective": cumulative_cost,
        })
        previous_rotation = rotation
        previous_translation = translation
    return rows

def candidate_diagnostics(candidates, independent_indices, temporal_indices):
    """Create one row for every candidate, including both selection flags"""
    rows = []
    n_frames, candidate_count = candidates["data_cost"].shape
    for pose_index in range(n_frames):
        for candidate_index in range(candidate_count):
            rows.append({
                "frame": int(candidates["frame_indices"][pose_index]),
                "candidate": candidate_index,
                "selection_chamfer_cm": 100.0 * float(
                    candidates["chamfer_m"][pose_index, candidate_index]
                ),
                "normalized_data_cost": float(
                    candidates["data_cost"][pose_index, candidate_index]
                ),
                "icp_fitness": float(
                    candidates["fitness"][pose_index, candidate_index]
                ),
                "icp_inlier_rmse_cm": 100.0 * float(
                    candidates["inlier_rmse_m"][pose_index, candidate_index]
                ),
                "selected_independently": bool(
                    independent_indices[pose_index] == candidate_index
                ),
                "selected_by_temporal_path": bool(
                    temporal_indices[pose_index] == candidate_index
                ),
            })
    return rows

def save_csv(path, rows, columns):
    """Write dictionaries to a CSV with a fixed, reproducible column order."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path

def save_candidate_set(output_dir, candidates):
    """Save all candidate poses and data scores used by both selectors."""
    path = Path(output_dir) / "candidate_set.npz"
    np.savez_compressed(
        path,
        R=candidates["R"],
        t=candidates["t"],
        s=candidates["s"],
        frame_indices=candidates["frame_indices"],
        chamfer_m=candidates["chamfer_m"],
        data_cost=candidates["data_cost"],
        fitness=candidates["fitness"],
        inlier_rmse_m=candidates["inlier_rmse_m"],
    )
    return path

def save_variant(sequence_id, run_name, variant, candidates, path,
                 shared_candidate_rows, rotation_weight, translation_weight,
                 rotation_scale_degrees, translation_scale_meters,
                 results_root, generation_parameters, evaluate,
                 evaluation_sample_count, evaluation_sample_seed):
    """Save and optionally evaluate one selection from a shared candidate set."""
    trajectory = selected_trajectory(candidates, path)
    output_dir = save_experiment(
        "approach_4",
        variant,
        sequence_id,
        run_name,
        trajectory["R"],
        trajectory["t"],
        trajectory["s"],
        trajectory["frame_indices"],
        parameters={
            **generation_parameters,
            "selection": (
                "independent lowest normalized Chamfer"
                if variant == INDEPENDENT_VARIANT
                else "global dynamic-programming path"
            ),
            "data_cost": "symmetric Chamfer divided by template diagonal",
            "rotation_transition_weight": float(rotation_weight),
            "translation_transition_weight": float(translation_weight),
            "rotation_normalization_degrees": float(rotation_scale_degrees),
            "translation_normalization_meters": float(translation_scale_meters),
        },
        results_root=results_root,
    )
    candidate_path = save_candidate_set(output_dir, candidates)
    candidate_csv = save_csv(
        output_dir / "candidate_diagnostics.csv",
        shared_candidate_rows,
        [
            "frame",
            "candidate",
            "selection_chamfer_cm",
            "normalized_data_cost",
            "icp_fitness",
            "icp_inlier_rmse_cm",
            "selected_independently",
            "selected_by_temporal_path",
        ],
    )
    selected_rows = path_diagnostics(
        candidates,
        path,
        rotation_weight,
        translation_weight,
        rotation_scale_degrees,
        translation_scale_meters,
    )
    path_csv = save_csv(
        output_dir / "path_diagnostics.csv",
        selected_rows,
        [
            "frame",
            "selected_candidate",
            "selection_chamfer_cm",
            "normalized_data_cost",
            "rotation_change_deg",
            "translation_change_cm",
            "transition_cost",
            "cumulative_objective",
        ],
    )
    print(f"saved Approach 4 {sequence_id}/{run_name} {variant} -> {output_dir}")
    print(f"candidate set: {candidate_path}")
    print(f"candidate diagnostics: {candidate_csv}")
    print(f"path diagnostics: {path_csv}")

    if evaluate:
        evaluate_trajectory(
            sequence_id,
            output_dir / "trajectory.npz",
            output_dir,
            sample_count=evaluation_sample_count,
            sample_seed=evaluation_sample_seed,
        )
    return output_dir

def run_experiment(sequence_id, run_name, interval, variants=VARIANTS,
                   results_root=DEFAULT_RESULTS_ROOT, scale=1.0,
                   icp_sample_count=DEFAULT_ICP_SAMPLE_COUNT,
                   sample_seed=DEFAULT_SAMPLE_SEED,
                   max_correspondence_distance=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
                   max_iterations=DEFAULT_MAX_ITERATIONS,
                   convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE,
                   rotation_weight=DEFAULT_ROTATION_TRANSITION_WEIGHT,
                   translation_weight=DEFAULT_TRANSLATION_TRANSITION_WEIGHT,
                   rotation_scale_degrees=DEFAULT_ROTATION_SCALE_DEGREES,
                   translation_scale_meters=DEFAULT_TRANSLATION_SCALE_METERS,
                   evaluate=True, evaluation_sample_count=DEFAULT_SAMPLE_COUNT,
                   evaluation_sample_seed=DEFAULT_SAMPLE_SEED):
    """Generate candidates once, select requested paths, save, and evaluate."""
    unknown = set(variants).difference(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(sorted(unknown))}")
    validate_cost_parameters(
        rotation_weight,
        translation_weight,
        rotation_scale_degrees,
        translation_scale_meters,
    )
    start, end = interval
    candidates = generate_candidates(
        sequence_id,
        start,
        end,
        scale=scale,
        icp_sample_count=icp_sample_count,
        sample_seed=sample_seed,
        max_correspondence_distance=max_correspondence_distance,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
    )
    independent_indices = independent_path(candidates["data_cost"])
    temporal_indices, _, _ = temporal_path(
        candidates["R"],
        candidates["t"],
        candidates["data_cost"],
        rotation_weight=rotation_weight,
        translation_weight=translation_weight,
        rotation_scale_degrees=rotation_scale_degrees,
        translation_scale_meters=translation_scale_meters,
        description=f"select {sequence_id} temporal path",
    )
    shared_candidate_rows = candidate_diagnostics(
        candidates, independent_indices, temporal_indices
    )
    generation_parameters = {
        "candidate_source": "Approach 1B PCA orientations plus rigid ICP",
        "candidate_count_per_frame": int(candidates["data_cost"].shape[1]),
        "candidate_set_shared_between_variants": True,
        "orientation_policy": candidates["orientation_policy"],
        "scale": float(scale),
        "template_surface_samples": int(icp_sample_count),
        "sample_seed": int(sample_seed),
        "max_correspondence_distance_m": float(max_correspondence_distance),
        "max_iterations": int(max_iterations),
        "convergence_tolerance": float(convergence_tolerance),
        "template_diagonal_m": float(candidates["template_diagonal_m"]),
    }

    selected_paths = {
        INDEPENDENT_VARIANT: independent_indices,
        TEMPORAL_VARIANT: temporal_indices,
    }
    output_dirs = {}
    for variant in variants:
        output_dirs[variant] = save_variant(
            sequence_id,
            run_name,
            variant,
            candidates,
            selected_paths[variant],
            shared_candidate_rows,
            rotation_weight,
            translation_weight,
            rotation_scale_degrees,
            translation_scale_meters,
            results_root,
            generation_parameters,
            evaluate,
            evaluation_sample_count,
            evaluation_sample_seed,
        )
    return output_dirs

def self_check():
    """Check exact rigid fitting and temporal selection on a toy candidate set."""
    source = np.array([
        [-0.3, -0.1, 0.0],
        [0.2, -0.2, 0.1],
        [0.1, 0.4, -0.1],
        [0.5, 0.2, 0.3],
    ])
    angle = np.radians(25.0)
    expected_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected_translation = np.array([0.2, -0.1, 1.5])
    target = transform_points(source, expected_rotation, expected_translation)

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right_transpose = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    fitted_rotation = right_transpose.T @ left.T
    if np.linalg.det(fitted_rotation) < 0:
        right_transpose[-1] *= -1.0
        fitted_rotation = right_transpose.T @ left.T
    fitted_translation = target_center - fitted_rotation @ source_center
    np.testing.assert_allclose(fitted_rotation, expected_rotation, atol=1e-12)
    np.testing.assert_allclose(fitted_translation, expected_translation, atol=1e-12)

    n_frames = 5
    identity = np.eye(3)
    flip = np.diag([-1.0, -1.0, 1.0])
    rotations = np.empty((n_frames, 2, 3, 3), dtype=np.float64)
    rotations[:, 0] = identity
    rotations[:, 1] = flip
    translations = np.zeros((n_frames, 2, 3), dtype=np.float64)
    data_costs = np.array([
        [0.010, 0.011],
        [0.012, 0.010],
        [0.010, 0.012],
        [0.012, 0.010],
        [0.010, 0.012],
    ])
    independent = independent_path(data_costs)
    temporal, accumulated, backpointers = temporal_path(
        rotations,
        translations,
        data_costs,
        rotation_weight=0.01,
        translation_weight=0.0,
        description="Approach 4 self-check",
    )
    np.testing.assert_array_equal(independent, np.array([0, 1, 0, 1, 0]))
    np.testing.assert_array_equal(temporal, np.zeros(n_frames, dtype=np.int64))
    assert np.isfinite(accumulated).all()
    assert np.all(backpointers[1:] >= 0)

    candidates = {
        "R": rotations,
        "t": translations,
        "s": np.ones(n_frames),
        "frame_indices": np.arange(10, 10 + n_frames),
        "data_cost": data_costs,
    }
    trajectory = selected_trajectory(candidates, temporal)
    validate_trajectory(trajectory["R"], trajectory["t"], trajectory["s"])
    print("Approach 4 self-check passed.")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("run_name", nargs="?")
    parser.add_argument(
        "--variant",
        choices=("independent", "temporal", "both"),
        default="both",
        help="candidate-selection variant to save (default: both)",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--icp-samples", type=int, default=DEFAULT_ICP_SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE,
    )
    parser.add_argument(
        "--rotation-weight",
        type=float,
        default=DEFAULT_ROTATION_TRANSITION_WEIGHT,
    )
    parser.add_argument(
        "--translation-weight",
        type=float,
        default=DEFAULT_TRANSLATION_TRANSITION_WEIGHT,
    )
    parser.add_argument(
        "--rotation-scale-degrees",
        type=float,
        default=DEFAULT_ROTATION_SCALE_DEGREES,
    )
    parser.add_argument(
        "--translation-scale-meters",
        type=float,
        default=DEFAULT_TRANSLATION_SCALE_METERS,
    )
    parser.add_argument("--evaluation-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if (args.sequence_id is None) != (args.run_name is None):
        parser.error("provide both sequence_id and run_name, or neither")
    try:
        validate_cost_parameters(
            args.rotation_weight,
            args.translation_weight,
            args.rotation_scale_degrees,
            args.translation_scale_meters,
        )
    except ValueError as error:
        parser.error(str(error))
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
        "independent": (INDEPENDENT_VARIANT,),
        "temporal": (TEMPORAL_VARIANT,),
        "both": VARIANTS,
    }[args.variant]
    for sequence_id, run_name, interval in selected:
        run_experiment(
            sequence_id,
            run_name,
            interval,
            variants=variants,
            results_root=args.results_root,
            scale=args.scale,
            icp_sample_count=args.icp_samples,
            sample_seed=args.sample_seed,
            max_correspondence_distance=args.max_correspondence_distance,
            max_iterations=args.max_iterations,
            convergence_tolerance=args.convergence_tolerance,
            rotation_weight=args.rotation_weight,
            translation_weight=args.translation_weight,
            rotation_scale_degrees=args.rotation_scale_degrees,
            translation_scale_meters=args.translation_scale_meters,
            evaluate=not args.skip_evaluation,
            evaluation_sample_count=args.evaluation_samples,
            evaluation_sample_seed=args.evaluation_seed,
        )

if __name__ == "__main__":
    main()
