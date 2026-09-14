"""Automatically select frames that are clear, in motion, or occluded for each sequence"""
from pathlib import Path
import cv2
import numpy as np
from test_frames import DEFAULT_RADIUS, make_clip
from toy_task.load_frame import load_frame, load_sequence, project

ROOT = Path(__file__).resolve().parent
SEQUENCES = ["01__01", "01__03", "01__04", "01__07"]

def rolling_mean(values, radius):
    """Smooth values across a window of given frames"""
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")

def compute_diagnostics(sequence_id):
    """Produce simple observation quality and motion metrics for all frame"""
    sequence = load_sequence(sequence_id)
    n = sequence["n_frames"]

    point_count = np.zeros(n)
    mask_area = np.zeros(n)
    projection_agreement = np.zeros(n)
    centroid = np.zeros((n, 3))

    for frame_index in range(n):
        frame = load_frame(sequence, frame_index, with_image=False)
        points = frame["points"]
        mask = frame["mask"]

        point_count[frame_index] = len(points) # Number of observation
        mask_area[frame_index] = mask.mean() # Image pixels in object mask
        centroid[frame_index] = points.mean(axis=0) # Centroid of observed point cloud

        uv = project(points, frame["fx"], frame["fy"], frame["cx"], frame["cy"])
        pixels = np.rint(uv).astype(np.int64)
        h, w = mask.shape

        valid = (np.isfinite(uv).all(axis=1) & (points[:, 2] > 0) & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))

        if valid.any():
            p = pixels[valid]
            projection_agreement[frame_index] = mask[p[:, 1], p[:, 0]].mean()
        else:
            projection_agreement[frame_index] = 0.0

    # Observed motion approximation via centroid of object point cloud
    centroid_motion = np.zeros(n)
    centroid_motion[1:] = np.linalg.norm(np.diff(centroid, axis=0), axis=1)

    return {
        "point_count": point_count,
        "mask_area": mask_area,
        "projection_agreement": projection_agreement,
        "centroid_motion_m": centroid_motion,
    }

def candidate_centers(diagnostics, window=21):
    """Suggest candidate for clear/motion/occlusion frames; Actual selection to be done manually"""
    radius = window // 2

    agreement = rolling_mean(diagnostics["projection_agreement"], radius)
    motion = rolling_mean(diagnostics["centroid_motion_m"], radius)

    normalized_count = diagnostics["point_count"] / diagnostics["point_count"].max()
    normalized_area = diagnostics["mask_area"] / diagnostics["mask_area"].max()

    # Clear frame should have more observations and point-to-mask agreement
    clear_score = agreement * normalized_count
    # Occluded frame having less points, smaller mask, and more points outside mask
    occlusion_score = (0.2 * (1.0 - normalized_count) + 0.3 * (1.0 - normalized_area)
            + 0.5 * (1.0 - diagnostics["projection_agreement"]))

    valid = np.arange(radius, len(agreement) - radius)

    return {
        "clear": int(valid[np.argmax(clear_score[valid])]),
        "motion": int(valid[np.argmax(motion[valid])]),
        "occlusion": int(valid[np.argmax(occlusion_score[valid])]),
    }

def make_contact_sheet(sequence_id, indices, output_path):
    """Grid of video frames with object-mask outlines"""
    sequence = load_sequence(sequence_id)
    thumbnails = []

    for frame_index in indices:
        frame = load_frame(sequence, int(frame_index))
        image = frame["image"].copy()

        contours, _ = cv2.findContours(
            frame["mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        cv2.putText(image, f"{frame_index:05d}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (0, 255, 255), 4, cv2.LINE_AA)

        image = cv2.resize(image, (480, 270))
        thumbnails.append(image)

    rows = []
    for start in range(0, len(thumbnails), 4):
        row = thumbnails[start:start + 4]
        while len(row) < 4:
            row.append(np.zeros_like(thumbnails[0]))
        rows.append(np.hstack(row))

    contact_sheet = np.vstack(rows)
    if not cv2.imwrite(str(output_path), contact_sheet):
        raise RuntimeError(f"Could not write {output_path}")

def main():
    for sequence_id in SEQUENCES:
        output_dir = ROOT / "outputs" / "inspection" / sequence_id
        output_dir.mkdir(parents=True, exist_ok=True)

        sequence = load_sequence(sequence_id)
        diagnostics = compute_diagnostics(sequence_id)
        centers = candidate_centers(diagnostics)
        n = sequence["n_frames"]

        table = np.column_stack([
            np.arange(n),
            diagnostics["point_count"],
            diagnostics["mask_area"],
            diagnostics["projection_agreement"],
            diagnostics["centroid_motion_m"],
        ])

        np.savetxt(output_dir / "diagnostics.csv", table, delimiter=",",
            header=(
                "frame,point_count,mask_area_fraction,"
                "projection_agreement,centroid_motion_m"
            ), comments="")

        overview_indices = np.linspace(0, n - 1, 16, dtype=int)
        make_contact_sheet(sequence_id, overview_indices,
            output_dir / "contact_sheet.jpg",
        )

        print(f"\n{sequence_id}")
        
        # Print 21-frame interval
        for name, center in centers.items():
            start = max(0, center - 10)
            end = min(n, center + 11)
            
            print(f"  {name:9s}: [{start}, {end}) centered at {center}")
            
            stretch_indices = np.linspace(start, end - 1, 12, dtype=int)
            make_contact_sheet(sequence_id, stretch_indices,
                    output_dir / f"{name}.jpg")
            make_clip(sequence_id, center, DEFAULT_RADIUS, output_name=f"{name}.mp4")

if __name__ == "__main__":
    main()
