"""Approach 0 diagnostic baselines: centroid placement and PCA alignment

Variants:

0A fixed_orientation
    Keep R equal to identity and align the template surface centroid with each
    observed point-cloud centroid

0B pca_orientation
    Independently align the template and observation principal axes in every
    frame, then align their centroids. PCA signs and axis ordering are
    inherently ambiguous, so this baseline is expected to expose jumps

Usage:
Run one stretch only:
    python3 approach_0.py 01__01 clear --variant both

Run all manually selected stretches:
    python3 approach_0.py
"""

from argparse import ArgumentParser
import ast
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from evaluation import (DEFAULT_SAMPLE_COUNT, DEFAULT_SAMPLE_SEED,
                        evaluate_trajectory, sample_mesh_surface)
from experiment_utils import DEFAULT_RESULTS_ROOT, save_experiment
from pose_utils import transform_points
from toy_task.load_frame import load_frame, load_sequence, load_template

ROOT = Path(__file__).resolve().parent
SELECTION_PATH = ROOT / "stretches" / "SELECTION.md"
VARIANTS = ("fixed_orientation", "pca_orientation")

def load_stretches(path=SELECTION_PATH):
    """Read the literal STRETCHES dictionary from SELECTION.md """
    path = Path(path)
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stretches = None
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == "STRETCHES":
            stretches = ast.literal_eval(statement.value)
            break

    if not isinstance(stretches, dict):
        raise ValueError(f"{path} must define a literal STRETCHES dictionary")

    normalized = {}
    for sequence_id, named_ranges in stretches.items():
        if not isinstance(named_ranges, dict):
            raise ValueError(f"stretches for {sequence_id} must be a dictionary")
        normalized[str(sequence_id)] = {}
        for run_name, interval in named_ranges.items():
            if (not isinstance(interval, (tuple, list)) or len(interval) != 2
                    or not all(isinstance(value, int) for value in interval)):
                raise ValueError(
                    f"{sequence_id}/{run_name} must be an integer (start, end) pair"
                )
            start, end = interval
            if start < 0 or end <= start:
                raise ValueError(
                    f"invalid half-open interval for {sequence_id}/{run_name}: "
                    f"[{start}, {end})"
                )
            normalized[str(sequence_id)][str(run_name)] = (start, end)
    return normalized

def mesh_surface_centroid(vertices, faces):
    """Compute the exact area-weighted centroid of a triangular mesh surface"""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("template mesh has no finite, positive-area triangles")
    triangle_centroids = triangles.mean(axis=1)
    return np.sum(areas[:, None] * triangle_centroids, axis=0) / total_area

def point_centroid(points):
    """Return the centroid of finite observed points"""
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("point cloud has no finite points")
    return points.mean(axis=0)

def principal_axes(points):
    """Return deterministic, right-handed PCA axes as matrix columns.

    Each of the first two eigenvectors is signed so its largest-magnitude
    component is positive.  The third is their cross product.  This removes
    implementation-dependent signs but cannot remove the real PCA ambiguity
    of symmetric or partially observed objects.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("at least three finite points are required for PCA")

    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]

    first = axes[:, 0]
    second = axes[:, 1]
    if first[np.argmax(np.abs(first))] < 0:
        first = -first
    if second[np.argmax(np.abs(second))] < 0:
        second = -second
    third = np.cross(first, second)
    third /= np.linalg.norm(third)
    return np.column_stack((first, second, third))

def fit_stretch(sequence_id, start, end, variant, scale=1.0,
                pca_sample_count=DEFAULT_SAMPLE_COUNT,
                pca_sample_seed=DEFAULT_SAMPLE_SEED):
    """Fit one Approach 0 variant over interval [start, end)"""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")

    sequence = load_sequence(sequence_id)
    if end > sequence["n_frames"]:
        raise ValueError(
            f"interval [{start}, {end}) exceeds {sequence_id}'s "
            f"{sequence['n_frames']} frames"
        )

    vertices, faces = load_template(sequence["template"])
    template_centroid = mesh_surface_centroid(vertices, faces)
    template_axes = None
    if variant == "pca_orientation":
        template_samples = sample_mesh_surface(vertices, faces, pca_sample_count, pca_sample_seed)
        template_axes = principal_axes(template_samples)

    frame_indices = np.arange(start, end, dtype=np.int64)
    rotations = []
    translations = []

    for frame_index in tqdm(
        frame_indices,
        desc=f"fit {sequence_id} {variant}",
        unit="frame",
    ):
        frame = load_frame(sequence, int(frame_index), with_image=False)
        observation_centroid = point_centroid(frame["points"])

        if variant == "fixed_orientation":
            rotation = np.eye(3)
        else:
            observation_axes = principal_axes(frame["points"])
            rotation = observation_axes @ template_axes.T

        # t is expressed for the original, uncentered template convention.
        translation = observation_centroid - float(scale) * (rotation @ template_centroid)
        rotations.append(rotation)
        translations.append(translation)

    return {
        "R": np.asarray(rotations, dtype=np.float64),
        "t": np.asarray(translations, dtype=np.float64),
        "s": np.full(len(frame_indices), float(scale), dtype=np.float64),
        "frame_indices": frame_indices,
    }

def run_experiment(sequence_id, run_name, interval, variant,
                   results_root=DEFAULT_RESULTS_ROOT, scale=1.0,
                   pca_sample_count=DEFAULT_SAMPLE_COUNT,
                   pca_sample_seed=DEFAULT_SAMPLE_SEED,
                   evaluate=True, evaluation_sample_count=DEFAULT_SAMPLE_COUNT,
                   evaluation_sample_seed=DEFAULT_SAMPLE_SEED):
    """Fit, save, and optionally evaluate one selected stretch."""
    start, end = interval
    trajectory = fit_stretch(
        sequence_id,
        start,
        end,
        variant,
        scale=scale,
        pca_sample_count=pca_sample_count,
        pca_sample_seed=pca_sample_seed,
    )
    parameters = {
        "scale": float(scale),
        "translation": "align observed and template surface centroids",
        "orientation": (
            "identity"
            if variant == "fixed_orientation"
            else "independent per-frame PCA with deterministic axis signs"
        ),
    }
    if variant == "pca_orientation":
        parameters.update({
            "template_pca_surface_samples": int(pca_sample_count),
            "template_pca_sample_seed": int(pca_sample_seed),
            "temporal_axis_disambiguation": False,
        })

    output_dir = save_experiment(
        "approach_0",
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
    print(
        f"saved {variant} {sequence_id}/{run_name} "
        f"frames [{start}, {end}) -> {output_dir}"
    )

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
    """Check centroid placement, PCA covariance alignment, and valid rotations."""
    rng = np.random.default_rng(4)
    template = rng.normal(size=(2_000, 3)) * np.array([3.0, 1.5, 0.4])
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([0.2, -0.4, 2.0])
    observation = transform_points(template, rotation, translation)

    estimated_rotation = principal_axes(observation) @ principal_axes(template).T
    np.testing.assert_allclose(estimated_rotation.T @ estimated_rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(estimated_rotation), 1.0, atol=1e-12)

    transformed = transform_points(
        template,
        estimated_rotation,
        observation.mean(axis=0) - estimated_rotation @ template.mean(axis=0),
    )
    estimated_covariance = np.cov(transformed, rowvar=False, bias=True)
    observed_covariance = np.cov(observation, rowvar=False, bias=True)
    np.testing.assert_allclose(estimated_covariance, observed_covariance, atol=1e-12)
    np.testing.assert_allclose(transformed.mean(axis=0), observation.mean(axis=0), atol=1e-12)

    stretches = load_stretches()
    assert stretches["01__01"]["clear"] == (74, 95)
    print("Approach 0 self-check passed.")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("run_name", nargs="?")
    parser.add_argument(
        "--variant",
        choices=("fixed", "pca", "both"),
        default="both",
        help="Approach 0 variant to run (default: both)",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--pca-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--pca-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--mesh-samples", type=int, default=DEFAULT_SAMPLE_COUNT,
                        help="surface samples used by evaluation")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED,
                        help="surface-sampling seed used by evaluation")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
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
        "fixed": ("fixed_orientation",),
        "pca": ("pca_orientation",),
        "both": VARIANTS,
    }[args.variant]

    for sequence_id, run_name, interval in selected:
        for variant in variants:
            run_experiment(
                sequence_id,
                run_name,
                interval,
                variant,
                results_root=args.results_root,
                scale=args.scale,
                pca_sample_count=args.pca_samples,
                pca_sample_seed=args.pca_seed,
                evaluate=not args.skip_evaluation,
                evaluation_sample_count=args.mesh_samples,
                evaluation_sample_seed=args.sample_seed,
            )

if __name__ == "__main__":
    main()
