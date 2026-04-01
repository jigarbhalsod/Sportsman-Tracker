"""
Phase 4 — Final Annotated Output + Optional Enhancements
Produces:
  1. Full annotated tracking video (tracked_final.mp4)
  2. Movement heatmap image (heatmap.jpg)
  3. Active ID count over time chart (id_count_chart.jpg)
  4. Summary statistics
"""

import cv2
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VIDEO_PATH      = "input_video.mp4"
OUTPUT_DIR      = "output_phase4"
MODEL_SIZE      = "yolov8n.pt"
CONFIDENCE      = 0.35
FRAME_SKIP      = 2
MAX_FRAMES      = None             # None = full video; set e.g. 9000 for quick test

CLASS_PERSON    = 0
TRAIL_LENGTH    = 40
FONT            = cv2.FONT_HERSHEY_SIMPLEX

ID_COLORS = [
    (0, 200, 100),  (255, 120,  30), (30, 150, 255), (220,  50, 220),
    (0, 220, 220),  (255, 220,   0), (180,  80, 255), (255,  80, 120),
    (80, 255, 180), (255, 180,  80), (80, 120, 255),  (200, 200,  50),
]
# ──────────────────────────────────────────────────────────────────────────────


def get_color(track_id: int) -> tuple:
    return ID_COLORS[track_id % len(ID_COLORS)]


def draw_frame(frame, detections, trail_history, frame_idx, total_ids):
    out = frame.copy()
    if detections.tracker_id is None:
        return out, 0

    active = 0
    for i, tid in enumerate(detections.tracker_id):
        if tid is None:
            continue
        color = get_color(tid)
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"#{tid}"
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x1 + 3, y1 - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{conf:.2f}", (x1 + 2, y2 + 14), FONT, 0.38, color, 1, cv2.LINE_AA)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        trail_history[tid].append((cx, cy))
        if len(trail_history[tid]) > TRAIL_LENGTH:
            trail_history[tid].pop(0)

        pts = trail_history[tid]
        for j in range(1, len(pts)):
            alpha = j / len(pts)
            cv2.line(out, pts[j - 1], pts[j], color, max(1, int(2 * alpha)))

        total_ids.add(tid)
        active += 1

    for li, text in enumerate([f"Frame     : {frame_idx}",
                                f"Active IDs: {active}",
                                f"Total IDs : {len(total_ids)}"]):
        y = 20 + li * 20
        cv2.putText(out, text, (8, y), FONT, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (8, y), FONT, 0.48, (0, 180, 80), 1, cv2.LINE_AA)

    return out, active


def build_heatmap(accumulator: np.ndarray, width: int, height: int, output_path: str):
    """Normalize accumulator and save as a coloured heatmap."""
    if accumulator.max() == 0:
        return
    norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # blend with a dark background for readability
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    blended = cv2.addWeighted(bg, 0.3, heatmap, 0.7, 0)
    cv2.putText(blended, "Movement heatmap", (10, 24),
                FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, blended)
    print(f"      Heatmap saved : {output_path}")


def build_count_chart(frame_indices: list, counts: list, output_path: str):
    """Draw a simple line chart of active ID count over time."""
    W, H = 800, 300
    pad = 50
    chart = np.ones((H, W, 3), dtype=np.uint8) * 240

    if not counts or max(counts) == 0:
        return

    max_count = max(counts) + 2
    n = len(counts)

    # grid lines
    for g in range(0, max_count + 1, max(1, max_count // 5)):
        y = H - pad - int((g / max_count) * (H - 2 * pad))
        cv2.line(chart, (pad, y), (W - pad, y), (200, 200, 200), 1)
        cv2.putText(chart, str(g), (4, y + 4), FONT, 0.35, (100, 100, 100), 1)

    # plot line
    pts = []
    for i, c in enumerate(counts):
        x = pad + int((i / max(n - 1, 1)) * (W - 2 * pad))
        y = H - pad - int((c / max_count) * (H - 2 * pad))
        pts.append((x, y))

    for i in range(1, len(pts)):
        cv2.line(chart, pts[i - 1], pts[i], (0, 150, 80), 2)

    # axes
    cv2.line(chart, (pad, pad), (pad, H - pad), (80, 80, 80), 1)
    cv2.line(chart, (pad, H - pad), (W - pad, H - pad), (80, 80, 80), 1)

    # labels
    cv2.putText(chart, "Active detections per frame", (pad, 22),
                FONT, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(chart, "Frame index", (W // 2 - 40, H - 8),
                FONT, 0.4, (80, 80, 80), 1)

    cv2.imwrite(output_path, chart)
    print(f"      Count chart saved : {output_path}")


def run_phase4():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "sample_frames"), exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Phase 4 — Final Annotated Output")
    print(f"{'='*55}")
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Model      : {MODEL_SIZE}")
    print(f"  Confidence : {CONFIDENCE}")
    print(f"  Frame skip : {FRAME_SKIP}")
    print(f"  Max frames : {MAX_FRAMES or 'full video'}")
    print(f"{'='*55}\n")

    print("[1/5] Loading model...")
    model = YOLO(MODEL_SIZE)

    print("[2/5] Initialising ByteTrack...")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=90,
        minimum_matching_threshold=0.7,
        frame_rate=30,
    )

    print("[3/5] Opening video...")
    cap          = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit        = min(MAX_FRAMES, total_frames) if MAX_FRAMES else total_frames
    print(f"      {total_frames} frames | {fps:.2f} fps | {width}x{height}")
    print(f"      Processing {limit} frames\n")

    output_path = os.path.join(OUTPUT_DIR, "tracked_final.mp4")
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    writer      = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # heatmap accumulator
    heatmap_acc = np.zeros((height, width), dtype=np.float32)

    trail_history  = defaultdict(list)
    total_ids      = set()
    id_first_seen  = {}
    frame_counts   = []      # active count per sampled frame
    frame_indices  = []

    frame_idx  = 0
    processed  = 0
    SAMPLE_AT  = [500, 1000, 2000, 4000, 6000, 8000]
    saved      = set()

    print("[4/5] Running detection + tracking...\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= limit:
            break

        if frame_idx % FRAME_SKIP == 0:
            results    = model(frame, classes=[CLASS_PERSON],
                               conf=CONFIDENCE, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == CLASS_PERSON]
            detections = tracker.update_with_detections(detections)

            if detections.tracker_id is not None:
                for j, tid in enumerate(detections.tracker_id):
                    if tid not in id_first_seen:
                        id_first_seen[tid] = frame_idx
                    # accumulate heatmap
                    x1, y1, x2, y2 = map(int, detections.xyxy[j])
                    cx = np.clip((x1 + x2) // 2, 0, width - 1)
                    cy = np.clip((y1 + y2) // 2, 0, height - 1)
                    r  = max(10, (y2 - y1) // 4)
                    cv2.circle(heatmap_acc, (cx, cy), r,
                               1.0, thickness=-1)

            annotated, active = draw_frame(
                frame, detections, trail_history, frame_idx, total_ids
            )
            writer.write(annotated)

            frame_counts.append(active)
            frame_indices.append(frame_idx)

            # save sample frames
            if frame_idx in SAMPLE_AT and frame_idx not in saved:
                sp = os.path.join(OUTPUT_DIR, "sample_frames",
                                  f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(sp, annotated)
                saved.add(frame_idx)
                print(f"      Sample saved: frame {frame_idx} | "
                      f"active={active} | total IDs={len(total_ids)}")

            processed += 1
            if processed % 1000 == 0:
                pct = (frame_idx / limit) * 100
                print(f"      {pct:5.1f}% | {processed} frames | "
                      f"unique IDs: {len(total_ids)}")
        else:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # ── Post-processing ───────────────────────────────────────────────────────
    print("\n[5/5] Generating enhancements...\n")
    build_heatmap(heatmap_acc, width, height,
                  os.path.join(OUTPUT_DIR, "heatmap.jpg"))
    build_count_chart(frame_indices, frame_counts,
                      os.path.join(OUTPUT_DIR, "id_count_chart.jpg"))

    # ── Final summary ─────────────────────────────────────────────────────────
    avg_active = sum(frame_counts) / len(frame_counts) if frame_counts else 0
    max_active = max(frame_counts) if frame_counts else 0

    print(f"\n{'='*55}")
    print(f"  Frames processed    : {processed}")
    print(f"  Total unique IDs    : {len(total_ids)}")
    print(f"  Avg active/frame    : {avg_active:.1f}")
    print(f"  Max active in frame : {max_active}")
    print(f"{'='*55}")
    print(f"\n  Output video  : {output_path}")
    print(f"  Heatmap       : {OUTPUT_DIR}/heatmap.jpg")
    print(f"  Count chart   : {OUTPUT_DIR}/id_count_chart.jpg")
    print(f"  Sample frames : {OUTPUT_DIR}/sample_frames/\n")


if __name__ == "__main__":
    run_phase4()
