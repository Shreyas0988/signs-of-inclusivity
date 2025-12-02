from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_file,
    make_response,
)
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import sqlite3
import os
import random
import datetime
import uuid
import json
import base64
import secrets

import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
socketio = SocketIO(app, cors_allowed_origins="*", message_queue=None)

DB_NAME = "quiz_stats.db"
VIDEO_FOLDER = "videos"
PROMPT_FILE = "prompts.json"
LABEL_FILE = "labels.json"

if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)

try:
    with open(PROMPT_FILE, "r") as f:
        PROMPTS = json.load(f)
except FileNotFoundError:
    print(f"ERROR: {PROMPT_FILE} not found.")
    PROMPTS = []
except json.JSONDecodeError:
    print(f"ERROR: Could not parse {PROMPT_FILE}.")
    PROMPTS = []

MODEL_LOADED = False
try:
    with open(LABEL_FILE, "r") as f:
        LABELS = json.load(f)
    print(f"Loaded {len(LABELS)} labels from {LABEL_FILE}.")

    MODEL = load_model("gesture_model.h5")

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    HANDS = mp_hands.Hands(max_num_hands=2)
    POSE = mp_pose.Pose()

    SEQUENCE_LENGTH = 30

    FRAME_BUFFER = {}

    print("Sign recognition model loaded successfully.")
    MODEL_LOADED = True
except FileNotFoundError:
    print(f"ERROR: Label file '{LABEL_FILE}' not found.")
except Exception as e:
    print(f"ERROR: Could not load ML components. Error: {e}")


def normalize_landmarks(hand_landmarks, pose_landmarks):
    if pose_landmarks is None or not pose_landmarks:
        return None

    landmarks = []

    try:
        head = pose_landmarks[0]
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
    except IndexError:
        return None

    shoulder_dist = (
        np.linalg.norm(
            [left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1]]
        )
        + 1e-6
    )

    if hand_landmarks:
        for lm in hand_landmarks:
            x = (lm[0] - head[0]) / shoulder_dist
            y = (lm[1] - head[1]) / shoulder_dist
            z = lm[2] / shoulder_dist
            landmarks.extend([x, y, z])
    else:
        landmarks.extend([0] * 63)

    for lm in pose_landmarks:
        x = (lm[0] - head[0]) / shoulder_dist
        y = (lm[1] - head[1]) / shoulder_dist
        z = lm[2] / shoulder_dist
        landmarks.extend([x, y, z])

    return landmarks


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            question_index INTEGER, 
            is_correct BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


def get_video_filename(prompt_text):
    clean_name = (
        prompt_text.replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .replace("’", "")
        .replace(" ", "_")
    )
    return f"{clean_name}.mp4"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_question", methods=["GET"])
def get_question():
    user_ip = request.remote_addr

    if not PROMPTS:
        return jsonify({"error": "Quiz data not loaded."}), 500

    conn = get_db()
    cursor = conn.cursor()
    twenty_four_hours_ago = datetime.datetime.now() - datetime.timedelta(hours=24)

    cursor.execute(
        """
        SELECT question_index, is_correct 
        FROM attempts 
        WHERE ip_address = ? AND timestamp > ?
    """,
        (user_ip, twenty_four_hours_ago),
    )
    rows = cursor.fetchall()
    conn.close()

    weights = [1.0] * len(PROMPTS)
    for row in rows:
        idx = row["question_index"]
        if 0 <= idx < len(weights):
            if not row["is_correct"]:
                weights[idx] += 3.0
            else:
                weights[idx] = max(0.1, weights[idx] - 0.5)

    selected_internal_index = random.choices(range(len(PROMPTS)), weights=weights, k=1)[
        0
    ]

    question_token = str(uuid.uuid4())

    session["current_question_token"] = question_token
    session[question_token] = selected_internal_index

    total_attempts = len(rows)
    correct_attempts = sum(1 for r in rows if r["is_correct"])

    return jsonify(
        {
            "question_token": question_token,
            "stats": {"correct_24h": correct_attempts, "total_24h": total_attempts},
        }
    )


@app.route("/get_video_content")
def get_video_content():
    question_token = request.args.get("token")

    if not question_token:
        return "Missing question token", 400

    idx = session.get(question_token)

    if idx is None:
        return "Invalid or expired question token", 400

    if idx >= len(PROMPTS):
        return "Internal prompt index out of bounds.", 500

    prompt = PROMPTS[idx]
    filename = get_video_filename(prompt["text"])
    file_path = os.path.join(VIDEO_FOLDER, filename)

    if not os.path.exists(file_path):
        return "Video file not found on server", 404

    response = make_response(
        send_file(
            file_path,
            mimetype="video/mp4",
            download_name="challenge.mp4",
            as_attachment=False,
        )
    )

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    return response


@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    data = request.json
    user_answer = data.get("answer", "").lower()
    user_ip = request.remote_addr
    question_token = data.get("question_token")

    if not question_token:
        return jsonify({"error": "Missing question token on submit"}), 400

    internal_index = session.get(question_token)

    if internal_index is None:
        return jsonify({"error": "Invalid or expired question token"}), 400

    if internal_index >= len(PROMPTS):
        return jsonify({"error": "Internal prompt index out of bounds."}), 500

    target_data = PROMPTS[internal_index]
    required_keywords = target_data["keywords"]

    is_correct = all(keyword in user_answer for keyword in required_keywords)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO attempts (ip_address, question_index, is_correct)
        VALUES (?, ?, ?)
    """,
        (user_ip, internal_index, is_correct),
    )
    conn.commit()
    conn.close()

    session.pop("current_question_token", None)
    session.pop(question_token, None)

    return jsonify(
        {
            "correct": is_correct,
            "correct_answer": target_data["text"] if not is_correct else None,
        }
    )


@app.route("/avg_colour", methods=["POST"])
def avg_colour():
    if "frame" not in request.files:
        return jsonify({"colour": "#000000"})

    file = request.files["frame"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"colour": "#000000"})

    avg_colour = cv2.mean(img)[:3]
    avg_colour_hex = "#{:02x}{:02x}{:02x}".format(
        int(avg_colour[2]), int(avg_colour[1]), int(avg_colour[0])
    )
    return jsonify({"colour": avg_colour_hex})


@socketio.on("connect")
def handle_connect():
    FRAME_BUFFER[request.sid] = []
    print(f"Client {request.sid} connected")


@socketio.on("disconnect")
def handle_disconnect():
    FRAME_BUFFER.pop(request.sid, None)
    print(f"Client {request.sid} disconnected")


@socketio.on("stream_frames")
def handle_stream_frames(data):
    if not MODEL_LOADED:
        return

    base64_frames = data.get("frames", [])

    frame_buffer = FRAME_BUFFER.get(request.sid)
    if frame_buffer is None:
        FRAME_BUFFER[request.sid] = []
        frame_buffer = FRAME_BUFFER[request.sid]

    for base64_img in base64_frames:
        if not base64_img:
            continue

        try:
            img_bytes = base64.b64decode(base64_img)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = HANDS.process(frame_rgb)
            pose_results = POSE.process(frame_rgb)

            hand_lm = None
            if hand_results.multi_hand_landmarks:
                hand_lm = [
                    [lm.x, lm.y, lm.z]
                    for lm in hand_results.multi_hand_landmarks[0].landmark
                ]

            pose_lm = None
            if pose_results.pose_landmarks:
                pose_lm = [
                    [lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark
                ]

            normalized = normalize_landmarks(hand_lm, pose_lm)
            if normalized:
                frame_buffer.append(normalized)

        except Exception as e:
            print(f"Frame processing error for client {request.sid}: {e}")
            continue

    if len(frame_buffer) >= SEQUENCE_LENGTH:
        sequence_input = np.expand_dims(frame_buffer[-SEQUENCE_LENGTH:], axis=0)

        pred = MODEL.predict(sequence_input, verbose=0)
        predicted_label = LABELS[int(np.argmax(pred))]

        emit("prediction", predicted_label)

        frame_buffer[:] = frame_buffer[-SEQUENCE_LENGTH:]

    FRAME_BUFFER[request.sid] = frame_buffer


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)
