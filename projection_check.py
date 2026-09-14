"""Verify data-loading, coordinate, camera-projection pipeline"""
from pathlib import Path
import cv2
import numpy as np
from toy_task.load_frame import load_frame, load_sequence, project

# Setup directories
ROOT = Path(__file__).resolve().parent
SEQUENCES = ["01__01", "01__03", "01__04", "01__07"]

def projection_stats(frame):
    """Project a frame's 3D point cloud and calculate basic validity score"""
    points = frame["points"]
    uv = project(points, frame["fx"], frame["fy"], frame["cx"], frame["cy"]) # image pixel
    
    h, w = frame["mask"].shape
    pixels = np.rint(uv).astype(np.int64)
    
    finite = np.isfinite(uv).all(axis=1)
    positive_depth = points[:, 2] > 0
    in_image = (finite & positive_depth & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
                (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    
    inside_mask = np.zeros(len(points), dtype=bool)
    valid_pixels = pixels[in_image]
    inside_mask[in_image] = frame["mask"][valid_pixels[:, 1], valid_pixels[:, 0]]
    
    stat = {
        "positive_depth_fraction": positive_depth.mean(),
        "in_image_fraction": in_image.mean(),
        "inside_mask_given_in_image": (
            inside_mask[in_image].mean() if in_image.any() else np.nan
        )
    }
    return uv, stat

def draw_projection(frame, uv, stride=10):
    """Draw mask boundary (green) and 1/10 of point cloud (pink) for a given iamge frame"""
    image = frame["image"].copy()
    mask = frame["mask"]
    
    # Draw green object-mask boundary
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
    h, w = mask.shape
    for u, v in uv[::stride]:
        if np.isfinite(u) and np.isfinite(v):
            x, y = int(round(u)), int(round(v))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(image, (x, y), 1, (255, 0, 255), -1)
    
    return image

def main():
    """Verify data-loading, coordinate, camera-projection pipeline"""
    for sequence_id in SEQUENCES:
        output_dir = ROOT / "outputs" / "projection_checks" / sequence_id
        output_dir.mkdir(parents=True, exist_ok=True)

        sequence = load_sequence(sequence_id)
        n = sequence["n_frames"]
        indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        
        for frame_index in indices:
            frame = load_frame(sequence, frame_index)
            uv, stats = projection_stats(frame)
            overlay = draw_projection(frame, uv)
            
            output_path = output_dir / f"{frame_index:05d}.jpg"
            cv2.imwrite(str(output_path), overlay)
            
            print(
                sequence_id,
                frame_index,
                f"points={len(frame['points'])}",
                f"z+={stats['positive_depth_fraction']:.3f}",
                f"in_image={stats['in_image_fraction']:.3f}",
                f"inside_mask={stats['inside_mask_given_in_image']:.3f}",
            )

if __name__ == "__main__":
    main()
