Parkinson’s Disease Detection via Eye-Tracking

This project presents an eye-tracking–based system for the early detection of Parkinson’s Disease using video analysis and machine learning. High-frame-rate eye movement recordings are processed to extract clinically relevant gaze features, which are then analyzed using temporal deep learning models. The system is designed to run on low-cost edge hardware, enabling portable and privacy-preserving screening.

✅ Key Features

Eye landmark detection using MediaPipe Face Mesh

Robust gaze and eye-movement feature extraction using OpenCV and NumPy

Noise-aware data preprocessing and behavioral labeling

Temporal modeling using an LSTM neural network

High-speed video processing using a 100 FPS camera

Designed for edge deployment (Raspberry Pi)

Offline processing to ensure data privacy

Binary prediction: Parkinson’s / Non-Parkinson’s

🧠 System & ML Pipeline Overview

Video Acquisition
Eye movement videos are recorded using a high-frame-rate camera to capture subtle eye dynamics.

Landmark Detection
Facial and eye landmarks are detected using MediaPipe to localize eye regions accurately.

Feature Extraction
Frame-wise eye features are extracted, including:

Eye Aspect Ratio (EAR)

Blink detection

Saccade velocity

Fixation (visual intake) detection

Pupil diameter and pupil center coordinates

Binocular Point of Regard (PoR)

Data Preprocessing

Noise reduction using median filtering

Handling of missing or blank frames

Removal of spurious micro-movements

Semantic labeling of eye behavior (blink, saccade, fixation)

Temporal Modeling
Cleaned frame-level features are organized into fixed-length sequences suitable for time-series learning.

Model Inference
An LSTM model analyzes temporal eye-movement patterns and outputs a probability score indicating Parkinson’s likelihood.

🗃 Project Structure
📁 eye_parkinsons_project/
│
├── main.py                    # Entry point for video processing pipeline
├── pipeline.py                # Feature extraction and sequence construction
├── model.py                   # LSTM model definition and inference
├── preprocessing.py           # Noise removal and data cleaning logic
├── utils.py                   # EAR, blink, saccade, fixation, PoR functions
├── models/
│   └── lstm_model.pt          # Trained LSTM model
├── data/
│   └── sample_video.mp4       # Example input video
├── eye_metrics_output.csv     # Frame-wise extracted features
└── README.md

⚙️ Requirements
pip install -r requirements.txt


Core Dependencies

Python 3.8+

OpenCV

MediaPipe

NumPy, Pandas

PyTorch (for LSTM)

Optional

Streamlit (for UI)

Docker (for full pipeline integration)

🚀 How to Run
Process a Recorded Video
python main.py --video data/sample_video.mp4

Live Capture (Edge Device)
python main.py --live

Output

eye_metrics_output.csv – cleaned and structured gaze features

Console output – Parkinson’s prediction with confidence

📊 Sample Features Extracted
Feature	Description
Eye Aspect Ratio	Measures eye openness for blink detection
Saccade Velocity	Speed of gaze movement between frames
Fixation Flag	Indicates stable visual attention
Blink Indicator	Frame-level blink detection
Pupil Diameter	Approximate pupil size (left & right eyes)
Point of Regard	Estimated binocular gaze location
🐳 Docker Integration

The complete pipeline is containerized using Docker, ensuring:

consistent runtime environment,

easy deployment across systems,

reproducibility of results,

simplified dependency management.

This makes the system suitable for deployment on edge devices and future clinical setups.

🔮 Future Improvements

Validation on larger and more diverse clinical datasets

Integration of multimodal biomarkers (speech, handwriting)

Model optimization for faster edge inference

Conversion to ONNX / TensorFlow Lite for embedded deployment

Real-time feedback interface for clinicians


