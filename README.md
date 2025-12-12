# Video Classification for Women's Safety Using Deep Learning

## Overview

This project implements a deep learning-based video classification system specifically designed for detecting anomalies in surveillance videos to enhance women's safety. The system can classify videos into five categories:

- Abuse & Violence
- Chasing & Snatching
- Normal
- Threatening using object
- Women surrounded by men

When the system detects potential threats, it automatically sends email alerts to predefined recipients.

## Features

- **Deep Learning Model**: Uses LSTM neural networks combined with MobileNetV2 for accurate video classification
- **Real-time Classification**: Processes uploaded videos and provides instant classification results
- **Email Alerts**: Automatically sends email notifications when threatening activities are detected
- **Web Interface**: User-friendly web interface built with Flask and Tailwind CSS
- **Sample Frame Preview**: Displays a representative frame from the analyzed video

## Technologies Used

- Python 3.x
- TensorFlow/Keras
- OpenCV
- Flask
- HTML/CSS/JavaScript
- Tailwind CSS

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/video-classification-womens-safety.git
   cd video-classification-womens-safety
   ```

2. Create a virtual environment:

   ```bash
   python -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r custom_video_classifier/requirements.txt
   ```

## Usage

1. Start the Flask server:

   ```bash
   python custom_video_classifier/app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`

3. Upload a video file using the interface

4. Click "Predict" to classify the video

5. View the results and confidence scores for each category

## Model Architecture

The system uses a two-stage approach:

1. **Feature Extraction**: MobileNetV2 extracts features from video frames
2. **Sequence Modeling**: LSTM network analyzes the temporal sequence of features to classify the entire video

The model was trained on a custom dataset containing videos in the five categories mentioned above.

## Email Alert System

The system is configured to send email alerts when it detects any of the following threat categories:

- Abuse & Violence
- Chasing & Snatching
- Threatening using object
- Women surrounded by men

To configure email alerts:

1. Update the `ALERT_EMAILS` list in [app.py](custom_video_classifier/app.py) with recipient addresses
2. Set your Gmail account credentials in `SENDER_EMAIL` and `SENDER_PASSWORD`

Note: For Gmail, you'll need to use an App Password rather than your regular password.

## Project Structure

```
video_classification_env/
├── custom_video_classifier/
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── lstm_video_classifier.h5  # Trained model
│   ├── label_encoder.pkl      # Label encoder for classes
│   ├── static/                # Web interface files
│   │   └── index.html         # Main HTML interface
│   └── uploads/               # Temporary upload directory
└── myvenv/                    # Python virtual environment
```

## Team Members

- **Reshika Srivastava** (22BEC0651)
- **Parth Sunil Kothawade** (22BEC0634)
- **Mridul Agrawal** (22BEC0772)

## Project Guide

**Dr. Mohiul Islam**  
Assistant Professor (Senior)  
Vellore Institute of Technology

---

© 2025 Vellore Institute of Technology — Department of Electronics and Communication Engineering
