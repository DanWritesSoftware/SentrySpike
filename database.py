import sqlite3
from config import Config

CFG = Config()

'''
Connection Management and Schema Initialization
'''

def get_connection():
    '''
    Return a configured SQLite connection.

    Sets WAL mode for concurrent reads during inference/serving,
    enables foreign keys, and uses Row factory for dict-like access.
    '''
    conn = sqlite3.connect(str(CFG.database_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    '''
    Create tables and indices if they don't exist.

    Safe to call from any service on startup
    '''
    conn = get_connection()
    try:
        conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                event_id        TEXT UNIQUE,
                start_time      TEXT,
                end_time        TEXT,
                gate_label      INTEGER,
                top_prediction  TEXT,
                confidence      REAL,
                predictions_json TEXT,
                frame_count     INTEGER,
                merged          INTEGER DEFAULT 0,
                reviewed        INTEGER DEFAULT 0,
                starred         INTEGER DEFAULT 0,
                created_at      TEXT,
                status          TEXT NOT NULL DEFAULT 'capturing'
            );

            CREATE TABLE IF NOT EXISTS event_frames (
                frame_id        TEXT UNIQUE,
                event_id        TEXT,
                timestamp       TEXT,
                image_path      TEXT,
                is_thumbnail    INTEGER DEFAULT 0,
                gate_score      REAL,
                predictions_json TEXT,
                roi_json        TEXT,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_start_time
                ON events(start_time);
            CREATE INDEX IF NOT EXISTS idx_events_status
                ON events(status);
            CREATE INDEX IF NOT EXISTS idx_frames_event_id
                ON event_frames(event_id);          
                           """)
        conn.commit()
    finally:
        conn.close()

'''
Camera Service
'''

def create_event(event_id, start_time):
    '''
    Insert a new event when motion is first detected.

    status starts as 'capturing', the camera service will close it once the burst is complete and cooldown expires
    '''
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO events (event_id, start_time, created_at, status)
               VALUES (?, ?, ?, 'capturing')""",
            (event_id, start_time, start_time)
        )
        conn.commit()
    finally:
        conn.close()

def add_frame(frame_id, event_id, timestamp, image_path, gate_score=None, roi_json=None):
    '''
    Insert a single captured frame belonging to an event

    called once per frame during a burst capture
    '''
    conn = get_connection()
    try:
        conn.execute(
                """INSERT INTO event_frames 
                (frame_id, event_id, timestamp, image_path, gate_score, roi_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (frame_id, event_id, timestamp, image_path, gate_score, roi_json)
        )
        conn.commit()
    finally:
        conn.close()

def add_frames_batch(frames):
    '''
    Insert multiple frames in a single transaction

    frames: list of (frame_id, event_id, timestamp, image_path, gate_score, roi_json
    '''
    conn = get_connection()
    try:
        conn.executemany(
            """INSERT INTO event_frames
               (frame_id, event_id, timestamp, image_path, gate_score, roi_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            frames
        )
        conn.commit()
    finally:
        conn.close()

def close_event(event_id, end_time, frame_count):
    '''
    Close an event after burst capture and cooldown

    transitions status from 'capturing' to 'awaiting_inference' so the inference worker picks it up.
    '''
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE events
               SET end_time = ?, frame_count = ?, status = 'awaiting_inference'
               WHERE event_id = ?""",
            (end_time, frame_count, event_id)
        )

        # Pick the middle frame by rowid order (insertion order = capture order)
        conn.execute(
            """UPDATE event_frames
               SET is_thumbnail = 1
               WHERE frame_id = (
                   SELECT frame_id FROM event_frames
                   WHERE event_id = ?
                   ORDER BY timestamp
                   LIMIT 1 OFFSET (
                       SELECT COUNT(*) / 2 FROM event_frames
                       WHERE event_id = ?
                   )
               )""",
            (event_id, event_id)
        )
        conn.commit()
    finally:
        conn.close()

'''
Inference Service
'''

def get_pending_events():
    '''
    Get events waiting for inference, oldest first.

    returns events with status='awaiting inference'
    '''
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT event_id, start_time, frame_count
               FROM events
               WHERE status = 'awaiting_inference'
               ORDER BY created_at ASC"""
        ).fetchall()
        return rows
    finally:
        conn.close()

def get_event_frames(event_id):
    '''
    Get all frames for an event, ordered by capture time.

    used by the inference service to find the image paths, and by Flask for the event detail view.
    '''
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT frame_id, timestamp, image_path, is_thumbnail,
                      gate_score, predictions_json, roi_json
               FROM event_frames
               WHERE event_id = ?
               ORDER BY timestamp ASC""",
            (event_id,)
        ).fetchall()
        return rows
    finally:
        conn.close()

def update_frame_scores(frame_scores):
    '''
    Write per-frame gate scores after batch inference.

    frame_scores: list of (gate_score, predictions_json, frame_id)

    called after running the gate model on all frames in a burst
    '''
    conn = get_connection()
    try:
        conn.executemany(
            """UPDATE event_frames
               SET gate_score = ?, predictions_json = ?
               WHERE frame_id = ?""",
            frame_scores
        )
        conn.commit()
    finally:
        conn.close()

def update_event_predictions(event_id, gate_label, top_prediction=None, confidence=None, predictions_json=None):
    '''
    Write inference results and mark event as complete.

    called once per event after the gate model (and eventually the specialist model) have run. Transitions status to 'complete'
    '''
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE events
               SET gate_label = ?,
                   top_prediction = ?,
                   confidence = ?,
                   predictions_json = ?,
                   status = 'complete'
               WHERE event_id = ?""",
            (gate_label, top_prediction, confidence, predictions_json, event_id)
        )
        conn.commit()
    finally:
        conn.close()