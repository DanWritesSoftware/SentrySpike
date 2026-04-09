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

HEAVY_LABEL_MAP = {
    0: "canid",
    1: "felid",
    2: "deer",
    3: "rabbit",
    4: "raccoon",
    5: "bird",
    6: "opossum",
    7: "small_mammal",
}


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_gate(frames_bgr, model_gate):
    '''
    Run the gate model over a burst and decide animal vs empty.

    Returns a dict with ok, final_label, final_conf, stability, agree_count, used_frames.
    The gate model outputs a single scalar potential per frame; scores above gate_threshold
    are classified as animal.
    '''
    print("[Gate] Analysing frames...")

    outputs = []
    top1_labels = []

    for frame in frames_bgr:
        x_in = preprocess_frame_for_akidanet(frame, CFG.image_size)
        logits = model_gate.predict(x_in)

        v = np.squeeze(logits)
        if v.ndim != 0:
            print(f"[Gate] Unexpected logits shape: {v.shape}, skipping frame")
            continue

        score = float(v)
        outputs.append(score)
        top1_labels.append("animal" if score > 0.0 else "empty")

    if not outputs:
        print("[Gate] No usable frames")
        return {"ok": False, "used_frames": 0}

    agg = float(np.mean(outputs))
    final_label = "animal" if agg > CFG.gate_threshold else "empty"

    agree_count = sum(1 for l in top1_labels if l == final_label)
    stability = agree_count / len(top1_labels)

    if final_label == "animal" and stability < CFG.stability_threshold:
        print(
            f"[Gate] animal overridden -> empty "
            f"(stability {agree_count}/{len(top1_labels)} < {CFG.stability_threshold})"
        )
        final_label = "empty"
    else:
        print(
            f"[Gate] {final_label} "
            f"(agg={agg:.2f}) "
            f"stability {agree_count}/{len(top1_labels)}"
        )

    return {
        "ok": True,
        "final_label": final_label,
        "final_conf": agg,
        "stability": stability,
        "agree_count": agree_count,
        "used_frames": len(outputs),
    }


def run_heavy(frames_bgr, model_heavy):
    '''
    Run the heavy species-classification model over a burst.

    Each frame produces a softmax distribution over 8 classes. Probabilities are
    averaged across frames, then argmax gives the top prediction.

    Returns a dict with ok, top_prediction, confidence, class_scores, used_frames.
    '''
    print("[Heavy] Classifying species...")

    per_frame_probs = []

    for frame in frames_bgr:
        x_in = preprocess_frame_for_akidanet(frame, CFG.image_size)
        logits = model_heavy.predict(x_in)

        v = np.squeeze(logits)
        if v.ndim != 1 or len(v) != len(HEAVY_LABEL_MAP):
            print(f"[Heavy] Unexpected output shape: {v.shape}, skipping frame")
            continue

        per_frame_probs.append(_softmax(v.astype(np.float32)))

    if not per_frame_probs:
        print("[Heavy] No usable frames")
        return {"ok": False, "used_frames": 0}

    mean_probs = np.mean(per_frame_probs, axis=0)
    top_idx = int(np.argmax(mean_probs))
    top_label = HEAVY_LABEL_MAP[top_idx]
    top_conf = float(mean_probs[top_idx])

    class_scores = {HEAVY_LABEL_MAP[i]: float(mean_probs[i]) for i in range(len(HEAVY_LABEL_MAP))}

    if top_conf < CFG.heavy_confidence_threshold:
        print(
            f"[Heavy] low confidence ({top_conf:.2%} < {CFG.heavy_confidence_threshold:.0%}) "
            f"— falling back to 'animal'"
        )
        return {
            "ok": True,
            "top_prediction": "animal",
            "confidence": top_conf,
            "class_scores": class_scores,
            "used_frames": len(per_frame_probs),
            "low_confidence": True,
        }

    print(f"[Heavy] {top_label} ({top_conf:.2%}) from {len(per_frame_probs)} frames")

    return {
        "ok": True,
        "top_prediction": top_label,
        "confidence": top_conf,
        "class_scores": class_scores,
        "used_frames": len(per_frame_probs),
        "low_confidence": False,
    }


def process_event(event, model_gate, model_heavy):
    '''
    Two-stage inference pipeline for one event:
      1. Gate model  — animal vs empty
         - empty:  delete frames from disk and remove the event from the DB
         - animal: proceed to stage 2
      2. Heavy model — species classification
         - writes results back to the DB and marks the event complete
    '''
    event_id = event["event_id"]
    print(f"[Inference] Processing event {event_id}")

    frame_rows = db.get_event_frames(event_id)
    if not frame_rows:
        print(f"[Inference] No frames for {event_id}, skipping")
        return

    frames_bgr = []
    for row in frame_rows:
        img = cv2.imread(row["image_path"])
        if img is None:
            print(f"[Inference] Could not load {row['image_path']}, skipping frame")
            continue
        frames_bgr.append(img)

    if not frames_bgr:
        print(f"[Inference] No loadable frames for {event_id}, skipping")
        return

    # --- Stage 1: Gate ---
    gate = run_gate(frames_bgr, model_gate)
    if not gate["ok"]:
        return

    if gate["final_label"] == "empty":
        print(f"[Inference] Event {event_id} is empty — deleting")
        db.delete_event_with_files(event_id)
        return

    # Write per-frame gate scores now that we know it's worth keeping
    frame_scores = [
        (outputs_score, None, row["frame_id"])
        for outputs_score, row in zip(
            # re-derive per-frame scores: we only have the aggregate, so store it uniformly
            [gate["final_conf"]] * len(frame_rows),
            frame_rows,
        )
    ]
    db.update_frame_scores(frame_scores)

    # --- Stage 2: Heavy model ---
    heavy = run_heavy(frames_bgr, model_heavy)
    if not heavy["ok"]:
        # Heavy model failed — still mark complete using gate label so it isn't stuck
        print(f"[Inference] Heavy model failed for {event_id}, storing gate result only")
        db.update_event_predictions(
            event_id,
            gate_label=1,
            top_prediction="animal",
            confidence=gate["final_conf"],
            predictions_json=json.dumps({
                "gate": gate,
                "heavy": None,
            })
        )
        return

    db.update_event_predictions(
        event_id,
        gate_label=1,
        top_prediction=heavy["top_prediction"],
        confidence=heavy["confidence"],
        predictions_json=json.dumps({
            "gate": {
                "final_label": gate["final_label"],
                "final_conf":  gate["final_conf"],
                "stability":   gate["stability"],
                "agree_count": gate["agree_count"],
                "used_frames": gate["used_frames"],
            },
            "heavy": {
                "top_prediction":  heavy["top_prediction"],
                "confidence":      heavy["confidence"],
                "class_scores":    heavy["class_scores"],
                "used_frames":     heavy["used_frames"],
                "low_confidence":  heavy["low_confidence"],
            },
        })
    )


def main():
    db.init_db()

    print("Loading gate model...")
    try:
        model_gate = akida.Model(CFG.gate_model_path)
    except Exception as e:
        print(f"Gate model loading failed ({e})")
        raise SystemExit

    print("Loading heavy model...")
    try:
        model_heavy = akida.Model(CFG.heavy_model_path)
    except Exception as e:
        print(f"Heavy model loading failed ({e})")
        raise SystemExit

    try:
        hw_list = devices()
        if hw_list:
            hw = hw_list[0]
            print(f"Mapping models to hardware: {hw.version}")
            model_gate.map(hw)
            model_heavy.map(hw)
        else:
            print("No Akida hardware detected; running on CPU backend.")
    except Exception as e:
        print(f"Hardware map failed ({e}), running on CPU backend.")

    print("Waiting for events...")
    while True:
        pending = db.get_pending_events()
        for event in pending:
            process_event(event, model_gate, model_heavy)
        time.sleep(2)


if __name__ == "__main__":
    main()
