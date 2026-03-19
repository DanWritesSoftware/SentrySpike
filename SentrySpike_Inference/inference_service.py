import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akida
from akida import devices                           # Akida runtime + hardware detection

import cv2
import numpy as np
import json
import time

from config import Config
from preprocess_frame import preprocess_frame_for_akidanet
import database as db

CFG = Config()


def analyze_burst_frames(frames_bgr, model_ak, *, roi_bbox=None):
    '''
    Runs inference over a list of frames (optionally cropped to ROI),
    aggregates predictions, and computes stability metrics.

    Returns a result dict with ok, final_label, final_conf, stability, used_frames.
    '''
    print("[Analysiing Frames...]")

    outputs = []        # raw scalar gate potential per frame
    top1_labels = []    # per-frame label for stability calculation

    for frame in frames_bgr:
        if roi_bbox is not None:
            x, y, w, h = roi_bbox
            frame = frame[y:y + h, x:x + w]

        x_in = preprocess_frame_for_akidanet(frame, CFG.image_size)
        logits = model_ak.predict(x_in)

        v = np.squeeze(logits)
        if v.ndim != 0:
            print(f"[Burst Analysis] Unexpected logits shape: {v.shape}")
            continue

        outputs.append(float(v))
        top1_labels.append("animal" if float(v) > 0.0 else "empty")

    if not outputs:
        print("[Burst Analysis] No usable frames")
        return {"ok": False, "used_frames": 0}

    agg = np.mean(outputs)
    final_label = "animal" if agg > CFG.gate_threshold else "empty"
    final_conf = float(agg)

    agree_count = sum(1 for l in top1_labels if l == final_label)
    stability = agree_count / len(top1_labels)

    print(
        f"[BURST RESULT] {final_label} "
        f"({final_conf:.2f}) "
        f"stability {agree_count}/{len(top1_labels)}"
    )

    return {
        "ok": True,
        "final_label": final_label,
        "final_conf": final_conf,
        "stability": stability,
        "agree_count": agree_count,
        "used_frames": len(outputs),
    }


def process_event(event, model_ak):
    '''
    Load frames for one event, run inference, and write results back to the DB.
    '''
    event_id = event["event_id"]
    print(f"[Inference] Processing event {event_id}")

    frame_rows = db.get_event_frames(event_id)
    if not frame_rows:
        print(f"[Inference] No frames for {event_id}, skipping")
        return

    # Load frames from disk; grab ROI from the first frame that has one
    frames_bgr = []
    roi_bbox = None
    for row in frame_rows:
        img = cv2.imread(row["image_path"])
        if img is None:
            print(f"[Inference] Could not load {row['image_path']}, skipping frame")
            continue
        frames_bgr.append(img)
        if roi_bbox is None and row["roi_json"]:
            roi_data = json.loads(row["roi_json"])
            roi_bbox = tuple(roi_data["roi_bbox"]) if roi_data["roi_bbox"] else None

    if not frames_bgr:
        print(f"[Inference] No loadable frames for {event_id}, skipping")
        return

    result = analyze_burst_frames(frames_bgr, model_ak, roi_bbox=roi_bbox)
    if not result["ok"]:
        return

    # Write per-frame gate scores
    frame_scores = [
        (result["final_conf"], None, row["frame_id"])
        for row in frame_rows
    ]
    db.update_frame_scores(frame_scores)

    # Write event-level results and mark complete
    db.update_event_predictions(
        event_id,
        gate_label=1 if result["final_label"] == "animal" else 0,
        top_prediction=result["final_label"],
        confidence=result["final_conf"],
        predictions_json=json.dumps({
            "final_label": result["final_label"],
            "final_conf":  result["final_conf"],
            "stability":   result["stability"],
            "agree_count": result["agree_count"],
            "used_frames": result["used_frames"],
        })
    )


def main():
    db.init_db()

    print("Loading gate model...")
    try:
        model_ak = akida.Model(CFG.gate_model_path)
    except Exception as e:
        print(f"Gate model loading failed ({e})")
        raise SystemExit

    try:
        hw_list = devices()
        if hw_list:
            hw = hw_list[0]
            print(f"Mapping to hardware: {hw.version}")
            model_ak.map(hw)
        else:
            print("No Akida hardware detected; running on CPU backend.")
    except Exception as e:
        print(f"Hardware map failed ({e}), running on CPU backend.")

    print("Waiting for events...")
    while True:
        pending = db.get_pending_events()
        for event in pending:
            process_event(event, model_ak)
        time.sleep(2)


if __name__ == "__main__":
    main()
