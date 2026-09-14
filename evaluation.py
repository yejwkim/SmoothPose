"""Evaluate a trajectory against one sequence.

Trajectory files are .npz files containing:
    R: (K, 3, 3) template-to-camera rotation matrices
    t: (K, 3) camera-space translations in meters
    s: scalar, (1,), or (K,) positive scale values
    frame_indices: optional (K,) contiguous source-frame indices

If frame_indices is absent, the trajectory must be complete and K must equal
the sequence frame count. Partial trajectories must include frame_indices.

Evaluation per frame:
- silhouette IoU
- Chamfer distance
- observation-to-mesh distance (in cm)
- rotation change (in degrees)
- translation change (in cm)
- large rotation jumps & symmetry-equivalent flips

Usage:
python3 evaluation.py <sequence_id> <trajectory.npz> [output_directory]

When output_directory is omitted, outputs are written beside trajectory.npz
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import KDTree
from tqdm.auto import tqdm
from experiment_utils import prepare_frame_indices
from pose_utils import (center_template, prepare_scales, rotation_geodesic, transform_points,
                        validate_trajectory)
from toy_task.load_frame import load_frame, load_sequence, load_template, project

ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE_COUNT = 20_000
DEFAULT_SAMPLE_SEED = 42
DEFAULT_ROTATION_JUMP_DEGREES = 60.0
DEFAULT_SYMMETRY_TOLERANCE = 0.03

def sample_mesh_surface(vertices, faces, count, seed):
    """Deterministically sample triangle surfaces in proportion to area
    Preparation for point-cloud distance calcuations
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if count <= 0:
        raise ValueError(f"sample count must be positive, got {count}")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")

    triangles = vertices[faces]
    cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    total_area = areas.sum()
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("mesh has no finite, positive-area triangles")

    rng = np.random.default_rng(seed)
    selected_faces = rng.choice(len(triangles), size=count, replace=True, p=areas / total_area)
    selected = triangles[selected_faces]

    # Square-root barycentric sampling is uniform over each triangle's area
    root_u = np.sqrt(rng.random(count))
    v = rng.random(count)
    weight_a = 1.0 - root_u
    weight_b = root_u * (1.0 - v)
    weight_c = root_u * v

    return (weight_a[:, None] * selected[:, 0] + weight_b[:, None] * selected[:, 1]
            + weight_c[:, None] * selected[:, 2])

def nearest_neighbor_distances(source, target):
    """Distance from each source point to its nearest target point."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        raise ValueError("point sets must both be nonempty")
    return KDTree(target).query(source, workers=-1)[0]

def point_cloud_fit_metrics(mesh_samples, observations):
    """Return unsquared mean nearest-neighbor distances in meters

    Chamfer = Average of two directional mean Euclidean distances
    """
    observations = np.asarray(observations, dtype=np.float64)
    observations = observations[np.isfinite(observations).all(axis=1)]
    mesh_samples = np.asarray(mesh_samples, dtype=np.float64)
    mesh_samples = mesh_samples[np.isfinite(mesh_samples).all(axis=1)]

    observation_to_mesh = nearest_neighbor_distances(observations, mesh_samples)
    mesh_to_observation = nearest_neighbor_distances(mesh_samples, observations)

    observation_to_mesh_mean = float(observation_to_mesh.mean())
    mesh_to_observation_mean = float(mesh_to_observation.mean())
    chamfer = 0.5 * (observation_to_mesh_mean + mesh_to_observation_mean)

    return {
        "chamfer_m": chamfer,
        "observation_to_mesh_m": observation_to_mesh_mean,
        "mesh_to_observation_m": mesh_to_observation_mean,
    }

def render_silhouette(vertices_camera, faces, intrinsics, image_shape):
    """Construct predicted values into silhouette"""
    vertices_camera = np.asarray(vertices_camera, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    fx, fy, cx, cy = intrinsics
    height, width = image_shape

    valid_vertices = (np.isfinite(vertices_camera).all(axis=1) & (vertices_camera[:, 2] > 1e-6))
    pixels = np.full((len(vertices_camera), 2), np.nan, dtype=np.float64)
    pixels[valid_vertices] = project(vertices_camera[valid_vertices], fx, fy, cx, cy)

    silhouette = np.zeros((height, width), dtype=np.uint8)
    coordinate_limit = 100_000

    for face in faces:
        if not valid_vertices[face].all():
            continue

        triangle = np.rint(pixels[face]).astype(np.int32)
        if np.max(np.abs(triangle)) > coordinate_limit:
            continue
        cv2.fillConvexPoly(silhouette, triangle, 1)

    return silhouette.astype(bool)

def silhouette_iou(predicted_mask, observed_mask):
    """Intersection over union of two Boolean silhouettes; Compute Silhouette IOU"""
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if predicted_mask.shape != observed_mask.shape:
        raise ValueError(
            "silhouette shapes differ: "
            f"{predicted_mask.shape} and {observed_mask.shape}"
        )

    union = np.logical_or(predicted_mask, observed_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(predicted_mask, observed_mask).sum()
    return float(intersection / union)

def draw_overlay(image, predicted_mask, observed_mask):
    """Overlay fit regions: red=observed only, blue=predicted only, green=both"""
    overlay = image.copy()
    observed_only = observed_mask & ~predicted_mask
    predicted_only = predicted_mask & ~observed_mask
    intersection = predicted_mask & observed_mask

    colors = np.zeros_like(overlay)
    colors[observed_only] = (0, 0, 255)
    colors[predicted_only] = (255, 0, 0)
    colors[intersection] = (0, 255, 0)

    occupied = predicted_mask | observed_mask
    overlay[occupied] = (0.55 * overlay[occupied] + 0.45 * colors[occupied]).astype(np.uint8)

    predicted_contours, _ = cv2.findContours(predicted_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, predicted_contours, -1, (255, 255, 255), 2)
    return overlay

def symmetry_shape_error_ratio(centered_template_samples, previous_rotation, current_rotation,
                               previous_scale, current_scale, template_diagonal):
    """Measure whether a large rotation leaves template geometry unchanged

    Translation removed; Centered template samples compared after two rotations
    Lower result could indicate symmetry-equivalent poses
    """
    previous = (float(previous_scale) * (centered_template_samples @ previous_rotation.T))
    current = (float(current_scale) * (centered_template_samples @ current_rotation.T))
    distances = point_cloud_fit_metrics(previous, current)
    return float(distances["chamfer_m"] / template_diagonal)

def is_possible_symmetry_flip(rotation_change_degrees, shape_error_ratio,
                              rotation_jump_degrees=DEFAULT_ROTATION_JUMP_DEGREES,
                              symmetry_tolerance=DEFAULT_SYMMETRY_TOLERANCE):
    """Decides if given values demonstrate symmetry flip"""
    return bool(rotation_change_degrees >= rotation_jump_degrees and
                shape_error_ratio <= symmetry_tolerance)

def format_metric(value, unit=""):
    """Helper for formatting"""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}{unit}"

def finite_summary(values):
    """JSON format statistics"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

def evaluate_trajectory(sequence_id, trajectory_path, output_dir=None,
                        sample_count=DEFAULT_SAMPLE_COUNT, sample_seed=DEFAULT_SAMPLE_SEED,
                        rotation_jump_degrees=DEFAULT_ROTATION_JUMP_DEGREES,
                        symmetry_tolerance=DEFAULT_SYMMETRY_TOLERANCE):
    """Evaluate one complete or contiguous partial trajectory."""
    trajectory_path = Path(trajectory_path)
    output_dir = trajectory_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence = load_sequence(sequence_id)
    vertices, faces = load_template(sequence["template"])
    trajectory = np.load(trajectory_path, allow_pickle=False)

    required_keys = {"R", "t", "s"}
    missing_keys = required_keys.difference(trajectory.files)
    if missing_keys:
        raise ValueError(f"trajectory is missing keys: {', '.join(sorted(missing_keys))}")

    rotations = np.asarray(trajectory["R"], dtype=np.float64)
    translations = np.asarray(trajectory["t"], dtype=np.float64)
    scales = prepare_scales(trajectory["s"], len(rotations))
    if "frame_indices" in trajectory.files:
        frame_indices = prepare_frame_indices(
            trajectory["frame_indices"], len(rotations), sequence["n_frames"]
        )
    else:
        if len(rotations) != sequence["n_frames"]:
            raise ValueError(
                "a partial trajectory must contain frame_indices; "
                f"trajectory has {len(rotations)} poses but {sequence_id} "
                f"has {sequence['n_frames']} frames"
            )
        frame_indices = prepare_frame_indices(
            None, len(rotations), sequence["n_frames"]
        )
    validation = validate_trajectory(rotations, translations, scales)

    # Same template samples reused for every frame so that metrics are comparable
    template_samples = sample_mesh_surface(vertices, faces, sample_count, sample_seed)
    _, template_center = center_template(vertices)
    centered_template_samples = template_samples - template_center
    template_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if template_diagonal <= 0:
        raise ValueError("template has zero diagonal length")

    video_path = Path(sequence["dir"]) / "video.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = float(sequence["fps"])

    # Partial stretches are contiguous, so seek once and then decode normally.
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))

    overlay_path = output_dir / "overlay.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create {overlay_path}")

    columns = ["frame", "silhouette_iou", "chamfer_cm", "observation_to_mesh_cm",
               "mesh_to_observation_cm", "rotation_change_deg", "translation_change_cm",
               "symmetry_shape_error_ratio", "large_rotation_jump", "possible_symmetry_flip"]
    rows = []

    try:
        frame_progress = tqdm(
            enumerate(frame_indices),
            total=len(frame_indices),
            desc=f"evaluate {sequence_id}",
            unit="frame",
        )
        for pose_index, frame_index in frame_progress:
            frame_index = int(frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"could not read video frame {frame_index}")

            frame = load_frame(sequence, frame_index, with_image=False)
            rotation = rotations[pose_index]
            translation = translations[pose_index]
            scale = scales[pose_index]

            posed_vertices = transform_points(vertices, rotation, translation, scale)
            posed_samples = transform_points(template_samples, rotation, translation, scale)
            predicted_mask = render_silhouette(posed_vertices, faces,
                                            (frame["fx"], frame["fy"], frame["cx"], frame["cy"]),
                                            frame["mask"].shape)

            iou = silhouette_iou(predicted_mask, frame["mask"])
            fit = point_cloud_fit_metrics(posed_samples, frame["points"])

            rotation_change = np.nan
            translation_change_cm = np.nan
            shape_error_ratio = np.nan
            large_rotation_jump = False
            possible_symmetry_flip = False

            if pose_index > 0:
                rotation_change = rotation_geodesic(rotations[pose_index - 1], rotation)
                translation_change_cm = 100.0 * float(np.linalg.norm(translation -
                                                                    translations[pose_index - 1]))
                large_rotation_jump = (rotation_change >= rotation_jump_degrees)

                # Geometry Comparison made for frames with large rotation jump
                if large_rotation_jump:
                    shape_error_ratio = symmetry_shape_error_ratio(centered_template_samples,
                            rotations[pose_index - 1], rotation, scales[pose_index - 1],
                            scale, template_diagonal)
                    possible_symmetry_flip = is_possible_symmetry_flip(rotation_change,
                            shape_error_ratio, rotation_jump_degrees, symmetry_tolerance)

            chamfer_cm = 100.0 * fit["chamfer_m"]
            observation_to_mesh_cm = (100.0 * fit["observation_to_mesh_m"])
            mesh_to_observation_cm = (100.0 * fit["mesh_to_observation_m"])

            rows.append([frame_index, iou, chamfer_cm, observation_to_mesh_cm,
                         mesh_to_observation_cm, rotation_change, translation_change_cm,
                         shape_error_ratio, int(large_rotation_jump), int(possible_symmetry_flip)])

            overlay = draw_overlay(image, predicted_mask, frame["mask"])
            text = (
                f"frame {frame_index:05d}  IoU {iou:.3f}  "
                f"Chamfer {chamfer_cm:.2f}cm  "
                f"dR {format_metric(rotation_change, 'deg')}  "
                f"dt {format_metric(translation_change_cm, 'cm')}"
            )
            cv2.rectangle(overlay, (18, 15), (1380, 69), (0, 0, 0), -1)
            cv2.putText(overlay, text, (30, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                        2, cv2.LINE_AA)
            if possible_symmetry_flip:
                cv2.putText(overlay, "POSSIBLE SYMMETRY FLIP", (30, 105), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 3, cv2.LINE_AA)
            writer.write(overlay)
    finally:
        writer.release()
        capture.release()

    table = np.asarray(rows, dtype=np.float64)
    metrics_path = output_dir / "metrics.csv"
    np.savetxt(metrics_path, table, delimiter=",", header=",".join(columns), comments="",
               fmt=["%d"] + ["%.9g"] * 7 + ["%d", "%d"])

    column_index = {name: index for index, name in enumerate(columns)}
    large_jump_frames = table[
        table[:, column_index["large_rotation_jump"]].astype(bool), 0
    ].astype(int).tolist()
    symmetry_flip_frames = table[
        table[:, column_index["possible_symmetry_flip"]].astype(bool), 0
    ].astype(int).tolist()

    summary = {
        "sequence_id": sequence_id,
        "trajectory": str(trajectory_path.resolve()),
        "sequence_n_frames": int(sequence["n_frames"]),
        "evaluated_n_frames": int(len(frame_indices)),
        "complete_sequence": bool(len(frame_indices) == sequence["n_frames"]),
        "frame_range": {
            "start": int(frame_indices[0]),
            "end_exclusive": int(frame_indices[-1] + 1),
        },
        "mesh_sampling": {
            "count": int(sample_count),
            "seed": int(sample_seed),
            "method": "area-weighted uniform triangle sampling",
        },
        "chamfer_definition": (
            "0.5 * (mean observation-to-mesh nearest-neighbor Euclidean "
            "distance + mean mesh-to-observation nearest-neighbor Euclidean "
            "distance), reported in centimeters without squaring"
        ),
        "rotation_jump_threshold_degrees": float(rotation_jump_degrees),
        "symmetry_shape_tolerance_ratio": float(symmetry_tolerance),
        "trajectory_validation": validation,
        "silhouette_iou": finite_summary(
            table[:, column_index["silhouette_iou"]]
        ),
        "chamfer_cm": finite_summary(table[:, column_index["chamfer_cm"]]),
        "observation_to_mesh_cm": finite_summary(
            table[:, column_index["observation_to_mesh_cm"]]
        ),
        "rotation_change_deg": finite_summary(
            table[:, column_index["rotation_change_deg"]]
        ),
        "translation_change_cm": finite_summary(
            table[:, column_index["translation_change_cm"]]
        ),
        "large_rotation_jump_frames": large_jump_frames,
        "possible_symmetry_flip_frames": symmetry_flip_frames,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(f"metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    print(f"overlay: {overlay_path}")
    print(f"large rotation jumps: {large_jump_frames}")
    print(f"possible symmetry flips: {symmetry_flip_frames}")
    return summary

def self_check():
    """Exercise metric, rasterization, sampling, and smoothness primitives; Sanity test"""
    np.testing.assert_array_equal(
        prepare_frame_indices([74, 75, 76], 3, sequence_n_frames=321),
        [74, 75, 76],
    )
    try:
        prepare_frame_indices([74, 76], 2, sequence_n_frames=321)
    except ValueError:
        pass
    else:
        raise AssertionError("noncontiguous partial trajectory was accepted")

    vertices = np.array([
        [-0.5, -0.5, 2.0],
        [0.5, -0.5, 2.0],
        [0.5, 0.5, 2.0],
        [-0.5, 0.5, 2.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    first = sample_mesh_surface(vertices, faces, 1_000, seed=7)
    second = sample_mesh_surface(vertices, faces, 1_000, seed=7)
    np.testing.assert_array_equal(first, second)

    fit = point_cloud_fit_metrics(first, first.copy())
    np.testing.assert_allclose(fit["chamfer_m"], 0.0, atol=1e-12)

    mask = render_silhouette(
        vertices,
        faces,
        (100.0, 100.0, 100.0, 100.0),
        (200, 200),
    )
    np.testing.assert_allclose(silhouette_iou(mask, mask), 1.0)

    identity = np.eye(3)
    quarter_turn = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(
        rotation_geodesic(identity, quarter_turn),
        90.0,
        atol=1e-12,
    )

    # Check symmetric objects
    centered_samples = first - vertices.mean(axis=0)
    shape_error_ratio = symmetry_shape_error_ratio(centered_samples, identity, quarter_turn,
            1.0, 1.0, float(np.linalg.norm(np.ptp(vertices, axis=0))))
    assert is_possible_symmetry_flip(90.0, shape_error_ratio)
    print("Evaluation self-check passed.")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("trajectory", nargs="?", type=Path)
    parser.add_argument(
        "output_dir", nargs="?", type=Path,
        help="output directory (default: directory containing trajectory.npz)",
    )
    parser.add_argument("--mesh-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--rotation-jump-degrees", type=float, default=DEFAULT_ROTATION_JUMP_DEGREES)
    parser.add_argument("--symmetry-tolerance", type=float, default=DEFAULT_SYMMETRY_TOLERANCE)
    parser.add_argument("--self-check", action="store_true",
                        help="test evaluation primitives without a trajectory")
    args = parser.parse_args()

    if not args.self_check:
        missing = [
            name
            for name in ("sequence_id", "trajectory")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "sequence_id and trajectory are required unless "
                "--self-check is used"
            )
    return args

def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return

    evaluate_trajectory(
        args.sequence_id,
        args.trajectory,
        args.output_dir,
        sample_count=args.mesh_samples,
        sample_seed=args.sample_seed,
        rotation_jump_degrees=args.rotation_jump_degrees,
        symmetry_tolerance=args.symmetry_tolerance,
    )

if __name__ == "__main__":
    main()
