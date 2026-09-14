"""Produce a short, slowed-down video around a selected center frame.

Usage:
    python3 test_frames.py <sequence_id> <center_frame> [radius]

Examples:
    python3 test_frames.py 01__01 25
    python3 test_frames.py 01__03 277 15

The default radius is 10, which selects 21 frames when the center is not near
a sequence boundary. Source videos are 30 FPS; clips are written at 10 FPS so
that frame-to-frame motion is easier to inspect.
"""

from argparse import ArgumentParser
from pathlib import Path
import cv2
from toy_task.load_frame import load_sequence

ROOT = Path(__file__).resolve().parent
SEQUENCE_IDS = ("01__01", "01__03", "01__04", "01__07")
DEFAULT_RADIUS = 10
OUTPUT_FPS = 10.0

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sequence_id", choices=SEQUENCE_IDS, help="sequence to inspect")
    parser.add_argument("center_frame", type=int, help="center frame of the extracted clip")
    parser.add_argument("radius", type=int, nargs="?", default=DEFAULT_RADIUS,
            help=f"frames before and after the center (default: {DEFAULT_RADIUS})")
    return parser.parse_args()

def make_clip(sequence_id, center_frame, radius, output_dir=None, output_name=None):
    """Write frames [center-radius, center+radius] as a diagnostic video."""
    sequence = load_sequence(sequence_id)
    n_frames = sequence["n_frames"]

    if not 0 <= center_frame < n_frames:
        raise ValueError(
            f"center frame must be between 0 and {n_frames - 1}, "
            f"got {center_frame}"
        )
    if radius < 0:
        raise ValueError(f"radius must be nonnegative, got {radius}")

    start_frame = max(0, center_frame - radius)
    end_frame = min(n_frames, center_frame + radius + 1)

    video_path = Path(sequence["dir"]) / "video.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))

    if output_dir is None:
        output_dir = ROOT / "outputs" / "test_frames" / sequence_id
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        output_name = f"center_{center_frame:05d}_radius_{radius}.mp4"
    output_path = output_dir / output_name

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create {output_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_index in range(start_frame, end_frame):
        ok, image = capture.read()
        if not ok:
            writer.release()
            capture.release()
            raise RuntimeError(f"could not read frame {frame_index}")

        mask_path = (
            Path(sequence["dir"])
            / "object_masks"
            / f"{frame_index:05d}.png"
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            writer.release()
            capture.release()
            raise RuntimeError(f"could not read {mask_path}")

        contours, _ = cv2.findContours(
            (mask > 127).astype("uint8"),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # The selected center is yellow; surrounding-frame masks are green.
        contour_color = (
            (0, 255, 255) if frame_index == center_frame else (0, 255, 0)
        )
        cv2.drawContours(image, contours, -1, contour_color, 3)

        source_time = frame_index / source_fps
        label = (
            f"{sequence_id}  frame {frame_index:05d}  "
            f"source time {source_time:.2f}s"
        )
        if frame_index == center_frame:
            label += "  CENTER"

        # Dark background keeps the label readable on bright video regions.
        cv2.rectangle(image, (18, 16), (850, 70), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (30, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(image)

    writer.release()
    capture.release()

    frame_count = end_frame - start_frame
    print(f"wrote: {output_path}")
    print(f"source frames: [{start_frame}, {end_frame}) ({frame_count} frames)")
    print(f"source duration: {frame_count / source_fps:.2f} seconds")
    print(f"output duration: {frame_count / OUTPUT_FPS:.2f} seconds")
    return output_path


def main():
    args = parse_args()
    make_clip(args.sequence_id, args.center_frame, args.radius)


if __name__ == "__main__":
    main()
