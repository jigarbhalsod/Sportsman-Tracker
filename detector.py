"""
Phase 2 — Object Detection Pipeline
Detects people in the input video using YOLOv8.
Saves sample annotated frames and a short preview clip.
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VIDEO_PATH      = "input_video.mp4"   # input video
OUTPUT_DIR      = "output_phase2"     # folder for results
MODEL_SIZE      = "yolov8n.pt"        # nano = fastest; swap to yolov8m.pt for better accuracy
CONFIDENCE      = 0.25                 # detection confidence threshold (0.0 – 1.0)
FRAME_SKIP      = 3                   # process every Nth frame (3 = ~10fps on a 30fps video)
MAX_FRAMES      = 5400    # process only first ~3 mins (3*60*30)
SAMPLE_FRAMES   = [99, 501, 999, 2001, 5001]
PREVIEW_SECONDS = 20                  # length of annotated preview clip (seconds)
CLASS_PERSON    = 0                   # COCO class ID for 'person'
BOX_COLOR       = (0, 200, 100)       # bounding box color (BGR)
BOX_THICKNESS   = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX
# ──────────────────────────────────────────────────────────────────────────────


def setup_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "sample_frames"), exist_ok=True)


def draw_detections(frame: np.ndarray, boxes, confidences) -> tuple[np.ndarray, int]:
    """Draw bounding boxes on frame. Returns annotated frame + detection count."""
    annotated = frame.copy()
    count = 0
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"person {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw + 4, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                    FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        count += 1
    # overlay frame stats
    cv2.putText(annotated, f"Detections: {count}", (10, 22),
                FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Detections: {count}", (10, 22),
                FONT, 0.6, (0, 180, 80), 1, cv2.LINE_AA)
    return annotated, count


def run_detection():
    setup_output_dir(OUTPUT_DIR)

    print(f"\n{'='*55}")
    print(f"  Phase 2 — Detection Pipeline")
    print(f"{'='*55}")
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Model      : {MODEL_SIZE}")
    print(f"  Confidence : {CONFIDENCE}")
    print(f"  Frame skip : every {FRAME_SKIP} frames")
    print(f"{'='*55}\n")

    # load model
    print("[1/4] Loading YOLOv8 model...")
    model = YOLO(MODEL_SIZE)
    print("      Model ready.\n")

    # open video
    print("[2/4] Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      {total_frames} frames | {fps:.2f} fps | {width}x{height}\n")

    # preview clip writer
    preview_frames = int(PREVIEW_SECONDS * fps)
    preview_path   = os.path.join(OUTPUT_DIR, "preview_detected.mp4")
    fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
    writer         = cv2.VideoWriter(preview_path, fourcc, fps, (width, height))

    print("[3/4] Running detection...\n")
    frame_idx       = 0
    processed       = 0
    total_detections = 0
    max_in_frame    = 0
    sample_saved    = set()

    while True:
        if frame_idx >= MAX_FRAMES:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # only process every Nth frame
        if frame_idx % FRAME_SKIP == 0:
            results = model(frame, classes=[CLASS_PERSON],
                            conf=CONFIDENCE, verbose=False)[0]

            boxes       = results.boxes.xyxy.cpu().numpy() if results.boxes else []
            confidences = results.boxes.conf.cpu().numpy() if results.boxes else []

            annotated, count = draw_detections(frame, boxes, confidences)
            total_detections += count
            max_in_frame = max(max_in_frame, count)
            processed += 1

            # write preview clip (first N seconds only)
            if frame_idx < preview_frames:
                writer.write(annotated)

            # save sample frames
            if frame_idx in SAMPLE_FRAMES and frame_idx not in sample_saved:
                out_path = os.path.join(OUTPUT_DIR, "sample_frames",
                                        f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, annotated)
                sample_saved.add(frame_idx)
                print(f"      Saved sample: frame {frame_idx:6d} | {count} detections")

            # progress every 500 processed frames
            if processed % 500 == 0:
                pct = (frame_idx / total_frames) * 100
                avg = total_detections / processed if processed else 0
                print(f"      Progress: {pct:5.1f}% | "
                      f"processed {processed} frames | "
                      f"avg detections/frame: {avg:.1f}")
        else:
            # still write non-processed frames to keep preview smooth
            if frame_idx < preview_frames:
                writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # ── Summary ──────────────────────────────────────────────────────────────
    avg_det = total_detections / processed if processed else 0
    print(f"\n[4/4] Done!\n")
    print(f"{'='*55}")
    print(f"  Frames processed   : {processed}")
    print(f"  Total detections   : {total_detections}")
    print(f"  Avg detections/frame: {avg_det:.1f}")
    print(f"  Max in single frame: {max_in_frame}")
    print(f"{'='*55}")
    print(f"\n  Output folder : {OUTPUT_DIR}/")
    print(f"  Preview clip  : {preview_path}")
    print(f"  Sample frames : {OUTPUT_DIR}/sample_frames/\n")

    # ── Confidence tuning hint ───────────────────────────────────────────────
    if avg_det < 2:
        print("  HINT: Average detections look low.")
        print("        Try lowering CONFIDENCE to 0.25 in the CONFIG section.\n")
    elif avg_det > 15:
        print("  HINT: Many detections per frame — possible false positives.")
        print("        Try raising CONFIDENCE to 0.5 or switching to yolov8m.pt.\n")


if __name__ == "__main__":
    run_detection()
