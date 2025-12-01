import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from flask import Flask, request, jsonify, send_from_directory
import base64
from datetime import datetime
import re
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===========================
# Logging setup
# ===========================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# ===========================
# Parameters
# ===========================
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
FRAMES_PER_VIDEO = 20
NUM_CONSIDERED_FRAMES = 100
FRAME_STEP = 5
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'lstm_video_classifier.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
LOG_FILE = 'predictions.log'
STATIC_FILE = 'index.html'

# ===========================
# Email Alert Configuration (multiple recipients)
# ===========================
ALERT_EMAILS = [
    "mridulagrawal06@gmail.com",
    "parthkothawade2310@gmail.com",
    "Mohiul.islam@vit.ac.in",
    # add more recipients here as needed
]

# Gmail account used to send alerts
SENDER_EMAIL = "parthkothawade2310@gmail.com"
# App password for the Gmail account (replace with your app password)
SENDER_PASSWORD = "ckmt lwqz lufo xncz"

# Classes that trigger an alert (display labels)
THREAT_CLASSES = {
    'Abuse & Violence',
    'Chasing & Snatching',
    'Threatening using object',
    'Women surrounded by men'
}

# ===========================
# Category display mapping
# ===========================
DISPLAY_MAPPING = {
    'Abuse:Assault:Violence:Fight': 'Abuse & Violence',
    'Chasing:stalking:snatching': 'Chasing & Snatching',
    'Normal': 'Normal',
    'Threatening using object': 'Threatening using object',
    'Women surrounded by men': 'Women surrounded by men'
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===========================
# Load model and label encoder
# ===========================
try:
    model = load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    max_seq_len = FRAMES_PER_VIDEO
    logger.info("✅ Model and label encoder loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or label encoder: {e}")
    raise

# ===========================
# Helper Functions
# ===========================


def sanitize_filename(name):
    return re.sub(r'[^\w\-_\.]', '_', name)


def send_alert_email(predicted_label):
    """
    Send an alert email to all addresses in ALERT_EMAILS using the configured sender.
    """
    subject = f"⚠️ Alert: {predicted_label} Activity Detected!"
    body = (
        f"A new video was detected with potential threat activity.\n\n"
        f"Predicted category: {predicted_label}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Please review the footage immediately.\n\n"
        f"--\nAutomated Alert System"
    )

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(ALERT_EMAILS)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        logger.info(f"✅ Alert email sent to: {', '.join(ALERT_EMAILS)}")
        # also log recipients in predictions.log for audit
        with open(LOG_FILE, 'a') as f:
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ALERT_SENT_TO: {', '.join(ALERT_EMAILS)} | LABEL: {predicted_label}\n")
    except Exception as e:
        logger.error(f"❌ Failed to send alert email: {e}")
        with open(LOG_FILE, 'a') as f:
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ALERT_FAILED | ERROR: {e} | LABEL: {predicted_label}\n")

# ===========================
# Frame Extraction
# ===========================


def extract_uniform_frames(video_path, num_frames=FRAMES_PER_VIDEO, considered_frames=NUM_CONSIDERED_FRAMES):
    logger.info(f"Extracting frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return [], None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        logger.warning(f"No frames in video: {video_path}")
        return [], None

    # Determine indices to consider (approx evenly spaced)
    step = max(total_frames // considered_frames, 1)
    considered_indices = [i for i in range(0, total_frames, step)]
    selected_indices = considered_indices[::FRAME_STEP][:num_frames]

    frames = []
    sample_frame = None
    for idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(frame)
            if sample_frame is None:
                sample_frame = frame.copy()
        else:
            logger.debug(
                f"Failed to read frame at index {idx} from {video_path}")
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames, sample_frame

# ===========================
# Prediction
# ===========================


def predict_video_class(video_path):
    # Use MobileNetV2 (feature extractor)
    mobilenet_model = MobileNetV2(
        weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    frames, sample_frame = extract_uniform_frames(video_path)
    if not frames:
        return None, None, None

    feature_list = []
    for frame in frames:
        img_array = image.img_to_array(frame)
        img_array = preprocess_input(img_array)
        feature_list.append(img_array)

    frames_array = np.array(feature_list)
    features = mobilenet_model.predict(frames_array, verbose=0)
    features_padded = pad_sequences(
        [features], maxlen=max_seq_len, dtype='float32', padding='post')

    prediction = model.predict(features_padded, verbose=0)[0]
    predicted_class_index = int(np.argmax(prediction))
    predicted_label = label_encoder.inverse_transform(
        [predicted_class_index])[0]
    confidence_scores = {label: float(
        prediction[i]) for i, label in enumerate(label_encoder.classes_)}

    display_predicted_label = DISPLAY_MAPPING.get(
        predicted_label, predicted_label)
    display_confidence_scores = {DISPLAY_MAPPING.get(
        label, label): score for label, score in confidence_scores.items()}

    # Convert one sample frame to base64 for preview
    sample_frame_b64 = None
    if sample_frame is not None:
        _, buffer = cv2.imencode('.jpg', sample_frame)
        sample_frame_b64 = base64.b64encode(buffer).decode('utf-8')

    # Trigger email alert if predicted class is considered a threat
    if display_predicted_label in THREAT_CLASSES:
        try:
            send_alert_email(display_predicted_label)
        except Exception as e:
            logger.error(f"Error while sending alert: {e}")

    return display_predicted_label, display_confidence_scores, sample_frame_b64

# ===========================
# Logging Predictions
# ===========================


def log_prediction(video_name, predicted_label, confidence_scores):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(
            f"{timestamp} | Video: {video_name} | Predicted: {predicted_label} | Scores: {confidence_scores}\n")
    logger.info(f"Logged prediction for {video_name}")

# ===========================
# Flask Routes
# ===========================


@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received /predict request")
    if 'video' not in request.files:
        logger.warning("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        logger.warning("Empty filename in upload")
        return jsonify({'error': 'No video selected'}), 400

    video_name = sanitize_filename(video_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, video_name)
    video_file.save(video_path)
    logger.info(f"Saved uploaded video to {video_path}")

    try:
        predicted_label, confidence_scores, sample_frame_b64 = predict_video_class(
            video_path)
    except Exception as e:
        logger.error(f"Prediction failed for {video_path}: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': 'Failed to process video'}), 500

    if predicted_label is None:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': 'Failed to extract frames or process video'}), 400

    log_prediction(video_name, predicted_label, confidence_scores)

    # Clean up uploaded video file
    try:
        os.remove(video_path)
        logger.debug(f"Removed uploaded file {video_path}")
    except Exception as e:
        logger.warning(f"Could not remove uploaded file {video_path}: {e}")

    return jsonify({
        'predicted_label': predicted_label,
        'confidence_scores': confidence_scores,
        'sample_frame': sample_frame_b64
    })


@app.route('/')
def serve_gui():
    return send_from_directory(app.static_folder, STATIC_FILE)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Flask server running at http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
