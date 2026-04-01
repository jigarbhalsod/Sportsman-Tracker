"""
Phase 3 — Multi-Object Tracking + Persistent ID Assignment
Detects people using YOLOv8 and tracks them with ByteTrack.
Each person gets a unique persistent ID across the full video.
"""

import cv2
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VIDEO_PATH       = "input_video.mp4"
OUTPUT_DIR       = "output_phase3"
MODEL_SIZE       = "yolov8n.pt"
CONFIDENCE       = 0.35                # detection confidence threshold
FRAME_SKIP       = 2                   # process every Nth frame
MAX_FRAMES       = None                # ~5 mins of video; set to None for full video
CLASS_PERSON     = 0                   # COCO class ID for person

# visual settings
BOX_COLOR        = (0, 200, 100)       # BGR green for boxes
ID_COLORS        = [                   # one color per ID (cycles if more than 12)
    (0, 200, 100), (255, 120,  30), (30, 150, 255), (220,  50, 220),
    (0, 220, 220), (255, 220,   0), (180,  80, 255), (255,  80, 120),
    (80, 255, 180), (255, 180,  80), (80, 120, 255), (200, 200,  50),
]
BOX_THICKNESS    = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# trajectory trail settings
TRAIL_LENGTH     = 45                  # how many past positions to draw
TRAIL_THICKNESS  = 2

# sample frames to save
SAMPLE_FRAMES    = [500, 1000, 2000, 4000, 8000]

# preview clip
PREVIEW_SECONDS  = 30
# ──────────────────────────────────────────────────────────────────────────────


def get_id_color(track_id: int) -> tuple:
    return ID_COLORS[track_id % len(ID_COLORS)]


def draw_tracked_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    trail_history: dict,
    frame_idx: int,
    total_ids: set,
) -> np.ndarray:
    annotated = frame.copy()

    if detections.tracker_id is None:
        return annotated

    active_ids = set()

    for i, track_id in enumerate(detections.tracker_id):
        if track_id is None:
            continue

        color = get_id_color(track_id)
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0

        # bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # ID label background + text
        label = f"#{track_id}"
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # confidence score below box
        conf_label = f"{conf:.2f}"
        cv2.putText(annotated, conf_label, (x1 + 2, y2 + 14),
                    FONT, 0.38, color, 1, cv2.LINE_AA)

        # center point for trail
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        trail_history[track_id].append((cx, cy))
        if len(trail_history[track_id]) > TRAIL_LENGTH:
            trail_history[track_id].pop(0)

        # draw trail
        pts = trail_history[track_id]
        for j in range(1, len(pts)):
            alpha = j / len(pts)
            thickness = max(1, int(TRAIL_THICKNESS * alpha))
            cv2.line(annotated, pts[j - 1], pts[j], color, thickness)

        active_ids.add(track_id)
        total_ids.add(track_id)

    # ── HUD overlay ──────────────────────────────────────────────────────────
    hud_lines = [
        f"Frame     : {frame_idx}",
        f"Active IDs: {len(active_ids)}",
        f"Total IDs : {len(total_ids)}",
    ]
    for li, text in enumerate(hud_lines):
        y = 20 + li * 20
        cv2.putText(annotated, text, (8, y), FONT, 0.48,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, text, (8, y), FONT, 0.48,
                    (0, 180, 80), 1, cv2.LINE_AA)

    return annotated


def run_tracker():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "sample_frames"), exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Phase 3 — Tracking + ID Assignment")
    print(f"{'='*55}")
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Model      : {MODEL_SIZE}")
    print(f"  Confidence : {CONFIDENCE}")
    print(f"  Frame skip : every {FRAME_SKIP} frames")
    print(f"  Max frames : {MAX_FRAMES or 'full video'}")
    print(f"{'='*55}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("[1/4] Loading YOLOv8 model...")
    model = YOLO(MODEL_SIZE)
    print("      Model ready.\n")

    # ── Init ByteTrack via supervision ────────────────────────────────────────
    print("[2/4] Initialising ByteTrack tracker...")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=90,
        minimum_matching_threshold=0.7,
        frame_rate=30,
    )
    print("      Tracker ready.\n")

    # ── Open video ────────────────────────────────────────────────────────────
    print("[3/4] Opening video...")
    cap          = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit        = min(MAX_FRAMES, total_frames) if MAX_FRAMES else total_frames
    print(f"      {total_frames} frames | {fps:.2f} fps | {width}x{height}")
    print(f"      Processing first {limit} frames\n")

    # ── Video writer ──────────────────────────────────────────────────────────
    preview_frames = int(PREVIEW_SECONDS * fps)
    output_path    = os.path.join(OUTPUT_DIR, "tracked_output.mp4")
    fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
    writer         = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── State ─────────────────────────────────────────────────────────────────
    trail_history  = defaultdict(list)   # track_id → list of (cx, cy) points
    total_ids      = set()               # all unique IDs seen so far
    id_first_seen  = {}                  # track_id → frame_idx first appeared
    frame_idx      = 0
    processed      = 0
    sample_saved   = set()

    print("[4/4] Running detection + tracking...\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= limit:
            break

        if frame_idx % FRAME_SKIP == 0:
            # ── Run YOLO detection ────────────────────────────────────────────
            results = model(frame, classes=[CLASS_PERSON],
                            conf=CONFIDENCE, verbose=False)[0]

            # convert to supervision Detections
            detections = sv.Detections.from_ultralytics(results)

            # filter to persons only (redundant but safe)
            detections = detections[detections.class_id == CLASS_PERSON]

            # ── Run ByteTrack ─────────────────────────────────────────────────
            detections = tracker.update_with_detections(detections)

            # record first-seen frame per ID
            if detections.tracker_id is not None:
                for tid in detections.tracker_id:
                    if tid not in id_first_seen:
                        id_first_seen[tid] = frame_idx

            # ── Annotate frame ────────────────────────────────────────────────
            annotated = draw_tracked_frame(
                frame, detections, trail_history, frame_idx, total_ids
            )
            processed += 1

            # write to output video (full duration)
            writer.write(annotated)

            # save sample frames
            if frame_idx in SAMPLE_FRAMES and frame_idx not in sample_saved:
                out_path = os.path.join(OUTPUT_DIR, "sample_frames",
                                        f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, annotated)
                sample_saved.add(frame_idx)
                active = len(detections.tracker_id) if detections.tracker_id is not None else 0
                print(f"      Saved sample : frame {frame_idx:6d} | "
                      f"active={active} | total IDs={len(total_ids)}")

            # progress log
            if processed % 500 == 0:
                pct = (frame_idx / limit) * 100
                print(f"      Progress: {pct:5.1f}% | "
                      f"processed {processed} frames | "
                      f"unique IDs so far: {len(total_ids)}")
        else:
            # write un-processed frames as-is to keep video smooth
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Frames processed   : {processed}")
    print(f"  Total unique IDs   : {len(total_ids)}")
    print(f"  ID range           : #{min(total_ids)} — #{max(total_ids)}" if total_ids else "")
    print(f"{'='*55}")
    print(f"\n  Output video  : {output_path}")
    print(f"  Sample frames : {OUTPUT_DIR}/sample_frames/\n")

    # ── ID registry dump ──────────────────────────────────────────────────────
    print("  ID first-seen registry:")
    print(f"  {'ID':>4}  {'First seen at frame':>20}")
    print(f"  {'-'*30}")
    for tid in sorted(id_first_seen.keys()):
        print(f"  #{tid:>3}  frame {id_first_seen[tid]:>10}")
    print()

    if len(total_ids) > 30:
        print("  NOTE: High ID count may indicate ID switching (re-ID churn).")
        print("        Consider raising CONFIDENCE or minimum_matching_threshold.\n")


if __name__ == "__main__":
    run_tracker()
