"""Approach 5: joint trajectory refinement over a complete stretch.

The optimizer alternates nearest-neighbor correspondence updates with sparse
robust least squares. Rotations are updated in local rotation-vector
coordinates, so every decoded pose remains in SO(3).

Variants:

geometry_only
    Joint implementation of per-frame observation-to-template geometry terms.

geometry_temporal
    Add frame-to-frame velocity and acceleration penalties.

geometry_temporal_mask
    Also add approximate symmetric silhouette-contour correspondences. Mesh
    faces are still rendered for evaluation; the optimization uses projected
    3D surface samples whose contour associations are refreshed between outer
    iterations.

By default, the ball starts from Approach 3 sequential_forward so its
corrupted-frame pose is not reintroduced. Other objects start from Approach 4
temporal_path. A single sequence-wide scale can optionally be optimized.

Usage:
    python3 approach_5.py
    python3 approach_5.py 01__03 occlusion --variant temporal
    python3 approach_5.py 01__01 motion --scale-policy both
    python3 approach_5.py --self-check
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from approach_0 import load_stretches
from approach_1 import BALL_SEQUENCE_ID
from evaluation import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    evaluate_trajectory,
    render_silhouette,
    sample_mesh_surface,
)
from experiment_utils import (
    DEFAULT_RESULTS_ROOT,
    experiment_output_dir,
    prepare_frame_indices,
    save_experiment,
)
from pose_utils import prepare_scales, transform_points, validate_trajectory
from toy_task.load_frame import load_frame, load_sequence, load_template, project

GEOMETRY_ONLY = "geometry_only"
GEOMETRY_TEMPORAL = "geometry_temporal"
GEOMETRY_TEMPORAL_MASK = "geometry_temporal_mask"
VARIANTS = (GEOMETRY_ONLY, GEOMETRY_TEMPORAL, GEOMETRY_TEMPORAL_MASK)

DEFAULT_TEMPLATE_SAMPLE_COUNT = 2_000
DEFAULT_OBSERVATION_SAMPLE_COUNT = 400
DEFAULT_MASK_CONTOUR_SAMPLE_COUNT = 120
DEFAULT_MAX_CORRESPONDENCE_DISTANCE = 0.10
DEFAULT_OUTER_ITERATIONS = 2
DEFAULT_MAX_FUNCTION_EVALUATIONS = 30

DEFAULT_GEOMETRY_WEIGHT = 1.0
DEFAULT_MASK_WEIGHT = 0.20
DEFAULT_VELOCITY_WEIGHT = 0.10
DEFAULT_ACCELERATION_WEIGHT = 0.20

DEFAULT_GEOMETRY_SCALE_METERS = 0.05
DEFAULT_MASK_SCALE_PIXELS = 50.0
DEFAULT_ROTATION_VELOCITY_SCALE_DEGREES = 30.0
DEFAULT_TRANSLATION_VELOCITY_SCALE_METERS = 0.10
DEFAULT_ROTATION_ACCELERATION_SCALE_DEGREES = 15.0
DEFAULT_TRANSLATION_ACCELERATION_SCALE_METERS = 0.05

DEFAULT_SCALE_LOWER = 0.70
DEFAULT_SCALE_UPPER = 1.30
DEFAULT_ROBUST_LOSS = "soft_l1"

def default_source(sequence_id):
    """Return the default initialization approach and variant per object"""
    if sequence_id == BALL_SEQUENCE_ID:
        return "approach_3", "sequential_forward"
    return "approach_4", "temporal_path"

def default_source_variant(sequence_id, source_approach):
    """Choose a valid default variant after an explicit approach override."""
    if source_approach == "approach_3":
        return "sequential_forward"
    if source_approach == "approach_4":
        return "temporal_path"
    raise ValueError(f"unsupported source approach: {source_approach!r}")

def load_source_trajectory(sequence_id, run_name, source_approach,
                           source_variant,
                           results_root=DEFAULT_RESULTS_ROOT):
    """Load and validate a standard-layout initialization trajectory"""
    source_dir = experiment_output_dir(
        source_approach,
        source_variant,
        sequence_id,
        run_name,
        results_root,
    )
    trajectory_path = source_dir / "trajectory.npz"
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"missing source trajectory: {trajectory_path}\n"
            "Run the selected source approach first."
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
    if not np.allclose(scales, scales[0]):
        raise ValueError(
            "Approach 5 currently expects one sequence-wide source scale"
        )
    return {
        "R": rotations,
        "t": translations,
        "s": scales,
        "frame_indices": frame_indices,
        "path": trajectory_path,
    }

def validate_positive(value, name, allow_zero=False):
    """Validate one finite positive or nonnegative configuration value"""
    limit_ok = value >= 0 if allow_zero else value > 0
    if not np.isfinite(value) or not limit_ok:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value}")

def validate_parameters(parameters):
    """Validate numerical optimizer parameters in one dictionary"""
    for name in (
        "geometry_weight",
        "mask_weight",
        "velocity_weight",
        "acceleration_weight",
    ):
        validate_positive(parameters[name], name, allow_zero=True)
    for name in (
        "geometry_scale_meters",
        "mask_scale_pixels",
        "rotation_velocity_scale_degrees",
        "translation_velocity_scale_meters",
        "rotation_acceleration_scale_degrees",
        "translation_acceleration_scale_meters",
        "max_correspondence_distance",
    ):
        validate_positive(parameters[name], name)
    if parameters["outer_iterations"] <= 0:
        raise ValueError("outer iterations must be positive")
    if parameters["max_nfev"] <= 0:
        raise ValueError("maximum function evaluations must be positive")
    if not 0 < parameters["scale_lower"] < parameters["scale_upper"]:
        raise ValueError("scale bounds must satisfy 0 < lower < upper")

def subsample_rows(points, count, seed):
    """Deterministically retain at most count finite rows"""
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if count <= 0:
        raise ValueError("sample count must be positive")
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(points), size=count, replace=False))
    return points[indices]

def sample_contour(mask, count):
    """Return up to count approximately evenly spaced contour pixels (x, y)"""
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    pixels = np.concatenate([contour[:, 0, :] for contour in contours], axis=0)
    if len(pixels) > count:
        indices = np.linspace(0, len(pixels) - 1, count, dtype=np.int64)
        pixels = pixels[indices]
    return np.asarray(pixels, dtype=np.float64)

def load_problem(sequence_id, frame_indices, template_sample_count,
                 observation_sample_count, mask_contour_sample_count,
                 sample_seed):
    """Load mesh, deterministic samples, observations, masks, and intrinsics"""
    sequence = load_sequence(sequence_id)
    vertices, faces = load_template(sequence["template"])
    template_samples = sample_mesh_surface(
        vertices,
        faces,
        template_sample_count,
        sample_seed,
    )
    frames = []
    for frame_index in tqdm(
            frame_indices,
            desc=f"load {sequence_id} optimization data",
            unit="frame"):
        frame = load_frame(sequence, int(frame_index), with_image=False)
        observations = subsample_rows(
            frame["points"],
            observation_sample_count,
            sample_seed + int(frame_index),
        )
        frames.append({
            "frame": int(frame_index),
            "observations": observations,
            "mask": frame["mask"],
            "observed_contour": sample_contour(
                frame["mask"], mask_contour_sample_count
            ),
            "intrinsics": (frame["fx"], frame["fy"], frame["cx"], frame["cy"]),
            "image_shape": frame["mask"].shape,
        })
    return {
        "vertices": vertices,
        "faces": faces,
        "template_samples": template_samples,
        "frames": frames,
    }

def parameter_count(n_frames, fit_scale):
    """Return six pose corrections per frame plus optional log-scale"""
    return 6 * n_frames + (1 if fit_scale else 0)

def decode_parameters(values, base_rotations, base_translations, base_scale,
                      fit_scale):
    """Decode local pose corrections and one optional sequence-wide scale"""
    values = np.asarray(values, dtype=np.float64)
    n_frames = len(base_rotations)
    expected = parameter_count(n_frames, fit_scale)
    if values.shape != (expected,):
        raise ValueError(f"parameter vector must have shape ({expected},), got {values.shape}")
    corrections = values[:6 * n_frames].reshape(n_frames, 6)
    delta_rotations = Rotation.from_rotvec(corrections[:, :3]).as_matrix()
    rotations = delta_rotations @ np.asarray(base_rotations, dtype=np.float64)
    translations = np.asarray(base_translations, dtype=np.float64) + corrections[:, 3:]
    scale = float(base_scale)
    if fit_scale:
        scale *= float(np.exp(values[-1]))
    scales = np.full(n_frames, scale, dtype=np.float64)
    return rotations, translations, scales

def build_geometry_correspondences(problem, rotations, translations, scales,
                                   max_correspondence_distance):
    """Freeze gated observation-to-template nearest neighbors for one outer step"""
    correspondences = []
    template_samples = problem["template_samples"]
    for index, frame in enumerate(problem["frames"]):
        posed = transform_points(
            template_samples, rotations[index], translations[index], scales[index]
        )
        distances, sample_indices = KDTree(posed).query(
            frame["observations"], workers=-1
        )
        valid = np.isfinite(distances) & (distances <= max_correspondence_distance)
        correspondences.append({
            "template_indices": np.asarray(sample_indices[valid], dtype=np.int64),
            "targets": np.asarray(frame["observations"][valid], dtype=np.float64),
            "available": int(valid.sum()),
            "total": int(len(distances)),
        })
    return correspondences

def build_mask_correspondences(problem, rotations, translations, scales,
                               mask_contour_sample_count):
    """Freeze approximate symmetric contour-to-3D-sample associations"""
    correspondences = []
    vertices = problem["vertices"]
    faces = problem["faces"]
    template_samples = problem["template_samples"]
    for index, frame in enumerate(problem["frames"]):
        posed_vertices = transform_points(
            vertices, rotations[index], translations[index], scales[index]
        )
        predicted_mask = render_silhouette(
            posed_vertices,
            faces,
            frame["intrinsics"],
            frame["image_shape"],
        )
        predicted_contour = sample_contour(
            predicted_mask, mask_contour_sample_count
        )
        observed_contour = frame["observed_contour"]
        posed_samples = transform_points(
            template_samples, rotations[index], translations[index], scales[index]
        )
        valid_depth = posed_samples[:, 2] > 1e-6
        valid_indices = np.flatnonzero(valid_depth)
        if (not len(predicted_contour) or not len(observed_contour)
                or not len(valid_indices)):
            correspondences.append({
                "template_indices": np.empty(0, dtype=np.int64),
                "target_pixels": np.empty((0, 2), dtype=np.float64),
            })
            continue

        projected = project(
            posed_samples[valid_indices], *frame["intrinsics"]
        )
        finite = np.isfinite(projected).all(axis=1)
        valid_indices = valid_indices[finite]
        projected = projected[finite]
        if not len(projected):
            correspondences.append({
                "template_indices": np.empty(0, dtype=np.int64),
                "target_pixels": np.empty((0, 2), dtype=np.float64),
            })
            continue

        # Map the rendered contour to nearby 3D surface samples, then match
        # those movable samples to the observed contour in both directions.
        _, projected_lookup = KDTree(projected).query(
            predicted_contour, workers=-1
        )
        boundary_indices = np.unique(valid_indices[projected_lookup])
        boundary_projected = project(
            posed_samples[boundary_indices], *frame["intrinsics"]
        )
        if len(boundary_indices) > mask_contour_sample_count:
            keep = np.linspace(
                0,
                len(boundary_indices) - 1,
                mask_contour_sample_count,
                dtype=np.int64,
            )
            boundary_indices = boundary_indices[keep]
            boundary_projected = boundary_projected[keep]

        _, observed_lookup = KDTree(observed_contour).query(
            boundary_projected, workers=-1
        )
        forward_indices = boundary_indices
        forward_targets = observed_contour[observed_lookup]

        _, reverse_lookup = KDTree(boundary_projected).query(
            observed_contour, workers=-1
        )
        reverse_indices = boundary_indices[reverse_lookup]
        reverse_targets = observed_contour
        correspondences.append({
            "template_indices": np.concatenate(
                (forward_indices, reverse_indices)
            ).astype(np.int64),
            "target_pixels": np.vstack(
                (forward_targets, reverse_targets)
            ).astype(np.float64),
        })
    return correspondences

def residual_blocks(values, base_rotations, base_translations, base_scale,
                    fit_scale, problem, geometry_correspondences,
                    mask_correspondences, parameters, use_temporal, use_mask):
    """Return named normalized residual blocks and their frame dependencies"""
    rotations, translations, scales = decode_parameters(
        values,
        base_rotations,
        base_translations,
        base_scale,
        fit_scale,
    )
    n_frames = len(rotations)
    blocks = []
    template_samples = problem["template_samples"]

    for index, correspondence in enumerate(geometry_correspondences):
        template_indices = correspondence["template_indices"]
        if not len(template_indices):
            continue
        predicted = transform_points(
            template_samples[template_indices],
            rotations[index],
            translations[index],
            scales[index],
        )
        difference = predicted - correspondence["targets"]
        coefficient = np.sqrt(
            parameters["geometry_weight"]
            / (n_frames * difference.size)
        ) / parameters["geometry_scale_meters"]
        blocks.append((
            "geometry",
            coefficient * difference.ravel(),
            (index,),
            fit_scale,
        ))

    if use_mask and parameters["mask_weight"] > 0:
        for index, correspondence in enumerate(mask_correspondences):
            template_indices = correspondence["template_indices"]
            if not len(template_indices):
                continue
            posed = transform_points(
                template_samples[template_indices],
                rotations[index],
                translations[index],
                scales[index],
            )
            projected = project(posed, *problem["frames"][index]["intrinsics"])
            difference = projected - correspondence["target_pixels"]
            coefficient = np.sqrt(
                parameters["mask_weight"]
                / (n_frames * difference.size)
            ) / parameters["mask_scale_pixels"]
            blocks.append((
                "mask",
                coefficient * difference.ravel(),
                (index,),
                fit_scale,
            ))

    if use_temporal and n_frames > 1 and parameters["velocity_weight"] > 0:
        coefficient = np.sqrt(
            parameters["velocity_weight"] / ((n_frames - 1) * 6)
        )
        rotation_scale = np.radians(
            parameters["rotation_velocity_scale_degrees"]
        )
        translation_scale = parameters["translation_velocity_scale_meters"]
        relative_rotations = []
        translation_steps = []
        for index in range(1, n_frames):
            rotation_step = Rotation.from_matrix(
                rotations[index - 1].T @ rotations[index]
            ).as_rotvec()
            translation_step = translations[index] - translations[index - 1]
            relative_rotations.append(rotation_step)
            translation_steps.append(translation_step)
            values_block = coefficient * np.concatenate((
                rotation_step / rotation_scale,
                translation_step / translation_scale,
            ))
            blocks.append((
                "velocity",
                values_block,
                (index - 1, index),
                False,
            ))

        if n_frames > 2 and parameters["acceleration_weight"] > 0:
            acceleration_coefficient = np.sqrt(
                parameters["acceleration_weight"] / ((n_frames - 2) * 6)
            )
            rotation_acceleration_scale = np.radians(
                parameters["rotation_acceleration_scale_degrees"]
            )
            translation_acceleration_scale = (
                parameters["translation_acceleration_scale_meters"]
            )
            for index in range(1, n_frames - 1):
                rotation_acceleration = (
                    relative_rotations[index] - relative_rotations[index - 1]
                )
                translation_acceleration = (
                    translation_steps[index] - translation_steps[index - 1]
                )
                values_block = acceleration_coefficient * np.concatenate((
                    rotation_acceleration / rotation_acceleration_scale,
                    translation_acceleration / translation_acceleration_scale,
                ))
                blocks.append((
                    "acceleration",
                    values_block,
                    (index - 1, index, index + 1),
                    False,
                ))
    return blocks

def make_objective(base_rotations, base_translations, base_scale, fit_scale,
                   problem, geometry_correspondences, mask_correspondences,
                   parameters, use_temporal, use_mask):
    """Create residual function and matching sparse finite-difference pattern"""
    zero = np.zeros(parameter_count(len(base_rotations), fit_scale))

    def blocks(values):
        return residual_blocks(
            values,
            base_rotations,
            base_translations,
            base_scale,
            fit_scale,
            problem,
            geometry_correspondences,
            mask_correspondences,
            parameters,
            use_temporal,
            use_mask,
        )

    initial_blocks = blocks(zero)
    if not initial_blocks:
        raise ValueError("objective has no residuals")
    row_count = sum(len(block[1]) for block in initial_blocks)
    column_count = len(zero)
    sparsity = lil_matrix((row_count, column_count), dtype=np.int8)
    row_start = 0
    for _, residual, frame_dependencies, scale_dependency in initial_blocks:
        row_end = row_start + len(residual)
        for frame_index in frame_dependencies:
            sparsity[row_start:row_end, 6 * frame_index:6 * frame_index + 6] = 1
        if scale_dependency:
            sparsity[row_start:row_end, -1] = 1
        row_start = row_end

    def objective(values):
        current = blocks(values)
        return np.concatenate([block[1] for block in current])

    return objective, sparsity.tocsr()

def optimize_trajectory(problem, source, variant, fit_scale, parameters,
                        description="optimize trajectory"):
    """Alternate correspondence updates with sparse robust least squares"""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    validate_parameters(parameters)
    use_temporal = variant != GEOMETRY_ONLY
    use_mask = variant == GEOMETRY_TEMPORAL_MASK

    base_rotations = np.asarray(source["R"], dtype=np.float64)
    base_translations = np.asarray(source["t"], dtype=np.float64)
    base_scale = float(source["s"][0])
    values = np.zeros(parameter_count(len(base_rotations), fit_scale))
    diagnostics = []

    lower = np.full_like(values, -np.inf)
    upper = np.full_like(values, np.inf)
    if fit_scale:
        lower[-1] = np.log(parameters["scale_lower"] / base_scale)
        upper[-1] = np.log(parameters["scale_upper"] / base_scale)

    for outer_index in tqdm(
            range(parameters["outer_iterations"]),
            desc=description,
            unit="outer"):
        rotations, translations, scales = decode_parameters(
            values,
            base_rotations,
            base_translations,
            base_scale,
            fit_scale,
        )
        geometry_correspondences = build_geometry_correspondences(
            problem,
            rotations,
            translations,
            scales,
            parameters["max_correspondence_distance"],
        )
        mask_correspondences = (
            build_mask_correspondences(
                problem,
                rotations,
                translations,
                scales,
                parameters["mask_contour_sample_count"],
            )
            if use_mask else [
                {
                    "template_indices": np.empty(0, dtype=np.int64),
                    "target_pixels": np.empty((0, 2), dtype=np.float64),
                }
                for _ in rotations
            ]
        )
        objective, sparsity = make_objective(
            base_rotations,
            base_translations,
            base_scale,
            fit_scale,
            problem,
            geometry_correspondences,
            mask_correspondences,
            parameters,
            use_temporal,
            use_mask,
        )
        result = least_squares(
            objective,
            values,
            jac="2-point",
            jac_sparsity=sparsity,
            bounds=(lower, upper),
            loss=parameters["robust_loss"],
            f_scale=1.0,
            max_nfev=parameters["max_nfev"],
            verbose=0,
        )
        values = result.x
        _, _, current_scales = decode_parameters(
            values,
            base_rotations,
            base_translations,
            base_scale,
            fit_scale,
        )
        diagnostics.append({
            "outer_iteration": outer_index + 1,
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "function_evaluations": int(result.nfev),
            "status": int(result.status),
            "success": bool(result.success),
            "geometry_correspondences": int(sum(
                len(item["template_indices"])
                for item in geometry_correspondences
            )),
            "frames_without_geometry": int(sum(
                not len(item["template_indices"])
                for item in geometry_correspondences
            )),
            "mask_correspondences": int(sum(
                len(item["template_indices"])
                for item in mask_correspondences
            )),
            "scale": float(current_scales[0]),
            "message": str(result.message),
        })

    rotations, translations, scales = decode_parameters(
        values,
        base_rotations,
        base_translations,
        base_scale,
        fit_scale,
    )
    validate_trajectory(rotations, translations, scales)
    return {
        "R": rotations,
        "t": translations,
        "s": scales,
        "frame_indices": source["frame_indices"].copy(),
        "diagnostics": diagnostics,
    }

def save_optimization_diagnostics(output_dir, diagnostics):
    """Save one optimizer-status row per correspondence-update iteration"""
    path = Path(output_dir) / "optimization_diagnostics.csv"
    columns = [
        "outer_iteration",
        "cost",
        "optimality",
        "function_evaluations",
        "status",
        "success",
        "geometry_correspondences",
        "frames_without_geometry",
        "mask_correspondences",
        "scale",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(diagnostics)
    return path

def make_parameters(args=None, **overrides):
    """Return one complete optimizer-parameter dictionary"""
    parameters = {
        "geometry_weight": DEFAULT_GEOMETRY_WEIGHT,
        "mask_weight": DEFAULT_MASK_WEIGHT,
        "velocity_weight": DEFAULT_VELOCITY_WEIGHT,
        "acceleration_weight": DEFAULT_ACCELERATION_WEIGHT,
        "geometry_scale_meters": DEFAULT_GEOMETRY_SCALE_METERS,
        "mask_scale_pixels": DEFAULT_MASK_SCALE_PIXELS,
        "rotation_velocity_scale_degrees": DEFAULT_ROTATION_VELOCITY_SCALE_DEGREES,
        "translation_velocity_scale_meters": DEFAULT_TRANSLATION_VELOCITY_SCALE_METERS,
        "rotation_acceleration_scale_degrees": DEFAULT_ROTATION_ACCELERATION_SCALE_DEGREES,
        "translation_acceleration_scale_meters": DEFAULT_TRANSLATION_ACCELERATION_SCALE_METERS,
        "max_correspondence_distance": DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
        "mask_contour_sample_count": DEFAULT_MASK_CONTOUR_SAMPLE_COUNT,
        "outer_iterations": DEFAULT_OUTER_ITERATIONS,
        "max_nfev": DEFAULT_MAX_FUNCTION_EVALUATIONS,
        "scale_lower": DEFAULT_SCALE_LOWER,
        "scale_upper": DEFAULT_SCALE_UPPER,
        "robust_loss": DEFAULT_ROBUST_LOSS,
    }
    if args is not None:
        for name in parameters:
            if hasattr(args, name):
                parameters[name] = getattr(args, name)
    parameters.update(overrides)
    validate_parameters(parameters)
    return parameters

def run_experiment(sequence_id, run_name, interval, variant, fit_scale,
                   source_approach=None, source_variant=None,
                   results_root=DEFAULT_RESULTS_ROOT,
                   template_sample_count=DEFAULT_TEMPLATE_SAMPLE_COUNT,
                   observation_sample_count=DEFAULT_OBSERVATION_SAMPLE_COUNT,
                   sample_seed=DEFAULT_SAMPLE_SEED, parameters=None,
                   evaluate=True, evaluation_sample_count=DEFAULT_SAMPLE_COUNT,
                   evaluation_sample_seed=DEFAULT_SAMPLE_SEED):
    """Load, jointly refine, save, and optionally evaluate one experiment"""
    if source_approach is None:
        source_approach, _ = default_source(sequence_id)
    if source_variant is None:
        source_variant = default_source_variant(sequence_id, source_approach)
    parameters = make_parameters() if parameters is None else dict(parameters)
    validate_parameters(parameters)

    source = load_source_trajectory(
        sequence_id,
        run_name,
        source_approach,
        source_variant,
        results_root,
    )
    start, end = interval
    expected = np.arange(start, end, dtype=np.int64)
    if not np.array_equal(source["frame_indices"], expected):
        raise ValueError(
            "source trajectory does not match requested interval "
            f"[{start}, {end})"
        )
    problem = load_problem(
        sequence_id,
        source["frame_indices"],
        template_sample_count,
        observation_sample_count,
        parameters["mask_contour_sample_count"],
        sample_seed,
    )
    trajectory = optimize_trajectory(
        problem,
        source,
        variant,
        fit_scale,
        parameters,
        description=f"optimize {sequence_id} {run_name} {variant}",
    )
    scale_name = "fitted_scale" if fit_scale else "fixed_scale"
    output_variant = f"{variant}_{scale_name}"
    output_dir = save_experiment(
        "approach_5",
        output_variant,
        sequence_id,
        run_name,
        trajectory["R"],
        trajectory["t"],
        trajectory["s"],
        trajectory["frame_indices"],
        parameters={
            "source_approach": source_approach,
            "source_variant": source_variant,
            "source_trajectory": str(source["path"].resolve()),
            "optimizer": "alternating correspondences plus sparse scipy least_squares",
            "rotation_parameterization": "left-multiplied local rotation vectors",
            "geometry_correspondence_direction": "observation-to-template",
            "geometry_robust_loss": parameters["robust_loss"],
            "mask_term": (
                "approximate symmetric contour correspondences"
                if variant == GEOMETRY_TEMPORAL_MASK else "disabled"
            ),
            "temporal_terms": variant != GEOMETRY_ONLY,
            "fit_sequence_scale": bool(fit_scale),
            "template_surface_samples": int(template_sample_count),
            "observation_samples_per_frame": int(observation_sample_count),
            "sample_seed": int(sample_seed),
            **parameters,
        },
        results_root=results_root,
    )
    diagnostics_path = save_optimization_diagnostics(
        output_dir, trajectory["diagnostics"]
    )
    print(f"saved Approach 5 {sequence_id}/{run_name} {output_variant} -> {output_dir}")
    print(f"optimization diagnostics: {diagnostics_path}")
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
    """Check temporal refinement, silhouette associations, scale, and SO(3)"""
    rng = np.random.default_rng(53)
    template = rng.normal(size=(80, 3)) * np.array([0.12, 0.06, 0.03])
    n_frames = 7
    true_rotations = np.repeat(np.eye(3)[None, :, :], n_frames, axis=0)
    true_translations = np.column_stack((
        np.linspace(0.0, 0.12, n_frames),
        np.zeros(n_frames),
        np.full(n_frames, 2.0),
    ))
    source_translations = true_translations.copy()
    source_translations[3] += np.array([0.18, -0.12, 0.08])
    frames = []
    for index in range(n_frames):
        observations = transform_points(
            template,
            true_rotations[index],
            true_translations[index],
        )
        frames.append({
            "frame": index,
            "observations": observations,
            "mask": np.zeros((32, 32), dtype=bool),
            "observed_contour": np.empty((0, 2), dtype=np.float64),
            "intrinsics": (20.0, 20.0, 16.0, 16.0),
            "image_shape": (32, 32),
        })
    problem = {
        "vertices": template,
        "faces": np.empty((0, 3), dtype=np.int64),
        "template_samples": template,
        "frames": frames,
    }
    source = {
        "R": true_rotations,
        "t": source_translations,
        "s": np.ones(n_frames),
        "frame_indices": np.arange(n_frames),
    }
    parameters = make_parameters(
        geometry_weight=1.0,
        mask_weight=0.0,
        velocity_weight=0.1,
        acceleration_weight=0.5,
        max_correspondence_distance=0.5,
        outer_iterations=2,
        max_nfev=40,
    )
    result = optimize_trajectory(
        problem,
        source,
        GEOMETRY_TEMPORAL,
        fit_scale=False,
        parameters=parameters,
        description="Approach 5 self-check",
    )
    before_error = np.linalg.norm(
        source_translations[3] - true_translations[3]
    )
    after_error = np.linalg.norm(result["t"][3] - true_translations[3])
    # Frozen nearest-neighbor correspondences can explain part of a displaced
    # dense shape by selecting different surface samples. The temporal terms
    # should nevertheless remove most of the isolated trajectory error.
    assert after_error < before_error * 0.35, (
        f"outlier error was {before_error:.6f} m before refinement and "
        f"{after_error:.6f} m afterward"
    )
    report = validate_trajectory(result["R"], result["t"], result["s"])
    assert report["max_orthogonality_error"] < 1e-12

    values = np.zeros(parameter_count(n_frames, fit_scale=True))
    values[-1] = np.log(1.05)
    _, _, fitted_scales = decode_parameters(
        values,
        true_rotations,
        true_translations,
        base_scale=1.0,
        fit_scale=True,
    )
    np.testing.assert_allclose(fitted_scales, 1.05, atol=1e-12)

    # Exercise the mask path with a face-rendered planar square. Using the
    # rendered prediction as the observation should yield finite, nonempty
    # contour-to-surface associations.
    square_vertices = np.array([
        [-0.2, -0.2, 0.0],
        [0.2, -0.2, 0.0],
        [0.2, 0.2, 0.0],
        [-0.2, 0.2, 0.0],
    ])
    square_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    square_translation = np.array([[0.0, 0.0, 2.0]])
    square_rotation = np.eye(3)[None, :, :]
    square_scale = np.ones(1)
    square_intrinsics = (180.0, 180.0, 64.0, 64.0)
    square_shape = (128, 128)
    square_mask = render_silhouette(
        transform_points(square_vertices, np.eye(3), square_translation[0]),
        square_faces,
        square_intrinsics,
        square_shape,
    )
    square_problem = {
        "vertices": square_vertices,
        "faces": square_faces,
        "template_samples": sample_mesh_surface(
            square_vertices, square_faces, 500, 59
        ),
        "frames": [{
            "mask": square_mask,
            "observed_contour": sample_contour(square_mask, 80),
            "intrinsics": square_intrinsics,
            "image_shape": square_shape,
        }],
    }
    mask_correspondences = build_mask_correspondences(
        square_problem,
        square_rotation,
        square_translation,
        square_scale,
        mask_contour_sample_count=80,
    )
    assert len(mask_correspondences[0]["template_indices"]) > 0
    assert np.isfinite(mask_correspondences[0]["target_pixels"]).all()
    print("Approach 5 self-check passed.")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("run_name", nargs="?")
    parser.add_argument(
        "--variant",
        choices=("geometry", "temporal", "mask", "all"),
        default="all",
        help="objective variant to run (default: all)",
    )
    parser.add_argument(
        "--scale-policy",
        choices=("fixed", "fitted", "both"),
        default="fixed",
        help="sequence-wide scale policy (default: fixed)",
    )
    parser.add_argument(
        "--source-approach",
        choices=("auto", "approach_3", "approach_4"),
        default="auto",
    )
    parser.add_argument(
        "--source-variant",
        help="source variant; omission uses the object-specific default",
    )
    parser.add_argument(
        "--template-samples", type=int, default=DEFAULT_TEMPLATE_SAMPLE_COUNT
    )
    parser.add_argument(
        "--observation-samples", type=int,
        default=DEFAULT_OBSERVATION_SAMPLE_COUNT
    )
    parser.add_argument(
        "--mask-contour-samples", dest="mask_contour_sample_count",
        type=int, default=DEFAULT_MASK_CONTOUR_SAMPLE_COUNT
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--max-correspondence-distance", type=float,
        default=DEFAULT_MAX_CORRESPONDENCE_DISTANCE
    )
    parser.add_argument("--outer-iterations", type=int, default=DEFAULT_OUTER_ITERATIONS)
    parser.add_argument(
        "--max-nfev", type=int, default=DEFAULT_MAX_FUNCTION_EVALUATIONS
    )
    parser.add_argument("--geometry-weight", type=float, default=DEFAULT_GEOMETRY_WEIGHT)
    parser.add_argument("--mask-weight", type=float, default=DEFAULT_MASK_WEIGHT)
    parser.add_argument("--velocity-weight", type=float, default=DEFAULT_VELOCITY_WEIGHT)
    parser.add_argument(
        "--acceleration-weight", type=float,
        default=DEFAULT_ACCELERATION_WEIGHT
    )
    parser.add_argument(
        "--geometry-scale-meters", type=float,
        default=DEFAULT_GEOMETRY_SCALE_METERS
    )
    parser.add_argument(
        "--mask-scale-pixels", type=float, default=DEFAULT_MASK_SCALE_PIXELS
    )
    parser.add_argument(
        "--rotation-velocity-scale-degrees", type=float,
        default=DEFAULT_ROTATION_VELOCITY_SCALE_DEGREES
    )
    parser.add_argument(
        "--translation-velocity-scale-meters", type=float,
        default=DEFAULT_TRANSLATION_VELOCITY_SCALE_METERS
    )
    parser.add_argument(
        "--rotation-acceleration-scale-degrees", type=float,
        default=DEFAULT_ROTATION_ACCELERATION_SCALE_DEGREES
    )
    parser.add_argument(
        "--translation-acceleration-scale-meters", type=float,
        default=DEFAULT_TRANSLATION_ACCELERATION_SCALE_METERS
    )
    parser.add_argument("--scale-lower", type=float, default=DEFAULT_SCALE_LOWER)
    parser.add_argument("--scale-upper", type=float, default=DEFAULT_SCALE_UPPER)
    parser.add_argument("--evaluation-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if (args.sequence_id is None) != (args.run_name is None):
        parser.error("provide both sequence_id and run_name, or neither")
    if args.template_samples <= 0 or args.observation_samples <= 0:
        parser.error("sample counts must be positive")
    if args.mask_contour_sample_count <= 0:
        parser.error("mask contour sample count must be positive")
    try:
        make_parameters(args)
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
        "geometry": (GEOMETRY_ONLY,),
        "temporal": (GEOMETRY_TEMPORAL,),
        "mask": (GEOMETRY_TEMPORAL_MASK,),
        "all": VARIANTS,
    }[args.variant]
    scale_policies = {
        "fixed": (False,),
        "fitted": (True,),
        "both": (False, True),
    }[args.scale_policy]
    parameters = make_parameters(args)

    for sequence_id, run_name, interval in selected:
        default_approach, _ = default_source(sequence_id)
        source_approach = (
            default_approach
            if args.source_approach == "auto" else args.source_approach
        )
        source_variant = (
            default_source_variant(sequence_id, source_approach)
            if args.source_variant is None else args.source_variant
        )
        for variant in variants:
            for fit_scale in scale_policies:
                run_experiment(
                    sequence_id,
                    run_name,
                    interval,
                    variant,
                    fit_scale,
                    source_approach=source_approach,
                    source_variant=source_variant,
                    results_root=args.results_root,
                    template_sample_count=args.template_samples,
                    observation_sample_count=args.observation_samples,
                    sample_seed=args.sample_seed,
                    parameters=parameters,
                    evaluate=not args.skip_evaluation,
                    evaluation_sample_count=args.evaluation_samples,
                    evaluation_sample_seed=args.evaluation_seed,
                )

if __name__ == "__main__":
    main()
