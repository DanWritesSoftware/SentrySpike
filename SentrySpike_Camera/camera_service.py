import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                           # Centralized configuration class for camera and motion detection parameters
from motion_detection import MotionDetect           # Motion detection related functions

import cv2                                          # OpenCV for camera access and image processing
import time                                         # Used for timing loop
import os                                           # Directory creation
import json                                         # ROI serialization for DB
from datetime import datetime                       # Timestamped event IDs
import database as db                               # Event and frame persistence


CFG = Config()                                      # Instantiate the configuration with default values
detector = MotionDetect.create_motion_detector(CFG) # Motion detector needs persistance

TUNING_FLAG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.tuning_active')


def capture_burst_frames(cap, *, n_frames, sleep_s):
    '''
    Captures a burst of frames from an open cv2.VideoCapture.

    Returns:
        frames (list[np.ndarray]): Raw BGR frames
        timestamps (list[str]): ISO timestamp for each captured frame
    '''
    frames = []
    timestamps = []

    for _ in range(n_frames):
        ok, frame = cap.read()
        if ok:
            frames.append(frame.copy())
            timestamps.append(datetime.now().isoformat())
        time.sleep(sleep_s)

    return frames, timestamps

def save_burst_images(frames, event_id):
    '''
    Save a burst of full frames to a subfolder named by event_id under CFG.save_directory.

    Args:
        frames (list[np.ndarray]): BGR frames to save
        event_id (str): Unique event identifier, used as the folder name

    Returns:
        paths (list[str]): Absolute path to each saved frame, in order
    '''
    folder = os.path.join(CFG.save_directory, event_id)
    os.makedirs(folder, exist_ok=True)

    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(folder, f"frame_{i:02d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(os.path.abspath(path))

    print(f"[Save] {len(frames)} frames -> {folder}/")
    return paths


def main():
    '''
    Open Camera
    '''
    db.init_db()

    print("Opening camera...")
    cap = cv2.VideoCapture(CFG.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {CFG.camera_index}")
    print("Camera open.\n")

    '''
    Begin Watch Loop
    '''
    prev_frame = None
    consecutive_motion = 0
    motion_union = None
    last_trigger_time = 0.0

    try:
        while True:
            if os.path.exists(TUNING_FLAG):
                print("Tuning page active — pausing camera.")
                cap.release()
                while os.path.exists(TUNING_FLAG):
                    time.sleep(0.5)
                print("Tuning page closed — resuming camera.")
                cap = cv2.VideoCapture(CFG.camera_index)
                prev_frame = None
                detector = MotionDetect.create_motion_detector(CFG)
                consecutive_motion = 0
                motion_union = None
                continue

            # read frame
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; retrying...")
                continue

            # Initiailze prev_frame on first good frame
            if prev_frame is None:
                prev_frame = frame.copy()
                time.sleep(0.03)
                continue

            # Cooldown to avoid re-triggering constantly
            now = time.time()
            if (now - last_trigger_time) < CFG.cooldown_seconds:
                prev_frame = frame.copy()
                time.sleep(0.03)
                continue

            # Detect Motion (different algorithm based on config)
            event_active = consecutive_motion > 0
            if CFG.motion_method == "bgsub":
                bbox, _ = detector.detect(frame, event_active=event_active)
            else:
                bbox, _ = detector.detect(frame, prev_frame)
            
            if bbox is not None:
                consecutive_motion += 1
                motion_union = MotionDetect.union_bbox(motion_union, bbox)
            else:
                consecutive_motion = 0
                motion_union = None

            # Trigger burst when motion persists
            if consecutive_motion >= CFG.motion_confirm_frames:
                trigger_hits = consecutive_motion
                consecutive_motion = 0
                trigger_bbox = motion_union   # save before reset; used as ROI fallback below
                motion_union = None

                # Generate event ID and open DB record
                now_dt = datetime.now()
                event_id = now_dt.strftime("%Y%m%d_%H%M%S_%f")
                start_time = now_dt.isoformat()
                db.create_event(event_id, start_time)
                print(f"[MOTION] Triggered (hits={trigger_hits}) event={event_id}")

                # Capture burst
                frames, timestamps = capture_burst_frames(
                    cap,
                    n_frames=CFG.burst_frames,
                    sleep_s=CFG.burst_sleep_seconds
                    )
                if len(frames) < 2:
                    print("[Burst Capture] Not enough frames captured.")
                    db.delete_event(event_id)
                    continue

                # Compute motion bbox over burst, make ROI square for downstream use
                burst_bbox = MotionDetect.bbox_over_burst(
                    frames,
                    diff_thresh=CFG.motion_difference_threshold,
                    min_area=CFG.motion_min_area,
                    blur_ksize=CFG.motion_blur_ksize,
                    morph_iters=CFG.motion_morph_iterations,
                    pad=CFG.motion_pad
                    )
                # Prefer the bbox derived from the burst itself; fall back to the
                # bbox that triggered the event for cases where the subject has
                # already left the frame by the time the burst is captured.
                # burst_bbox is already padded by bbox_over_burst, so pass pad=0.
                # trigger_bbox comes raw from the detector, so it still needs padding.
                H, W = frames[0].shape[:2]
                if burst_bbox is not None:
                    roi = MotionDetect.square_bbox_in_bounds(
                        burst_bbox[0], burst_bbox[1], burst_bbox[2], burst_bbox[3],
                        W, H, pad=0
                        )
                elif trigger_bbox is not None:
                    roi = MotionDetect.square_bbox_in_bounds(
                        trigger_bbox[0], trigger_bbox[1], trigger_bbox[2], trigger_bbox[3],
                        W, H, pad=CFG.motion_pad
                        )
                else:
                    roi = None

                # Save frames, then batch-insert records
                paths = save_burst_images(frames, event_id)
                roi_json = json.dumps({"roi_bbox": list(roi) if roi is not None else None})
                frame_rows = [
                    (f"{event_id}_frame_{i:02d}", event_id, timestamps[i], paths[i], None, roi_json)
                    for i in range(len(frames))
                ]
                db.add_frames_batch(frame_rows)

                end_time = datetime.now().isoformat()
                db.close_event(event_id, end_time, len(frames))
                last_trigger_time = time.time()
            # 3. Advance
            prev_frame = frame.copy()
            time.sleep(0.03)
    finally:
        cap.release()

if __name__ == "__main__":
    main()