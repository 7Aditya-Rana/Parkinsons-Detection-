# Parkinson's Detection via Eye Tracking

This project is a real-time eye-tracking system designed to assist in early Parkinson's Disease detection. It captures eye movement using a high-FPS camera (e.g., Raspberry Pi camera), extracts meaningful gaze features using MediaPipe and OpenCV, and uses an LSTM model to predict Parkinson’s likelihood based on temporal eye movement patterns.

## ✅ Key Features

- Eye landmark detection using **MediaPipe Face Mesh**
- Gaze feature extraction using **OpenCV + NumPy**
- Temporal modeling using an **LSTM neural network**
- Real-time processing on **Raspberry Pi + 100 FPS camera**
- Gaze data saved to **CSV**
- Binary classification: Parkinson's Detected / Not Detected

## 🧠 ML Pipeline Overview

1. **Video Input**: Captured via Pi camera or uploaded file
2. **Landmark Detection**: Using MediaPipe for facial and eye landmarks
3. **Feature Extraction**:
   - Eye Aspect Ratio (EAR)
   - Blink detection
   - Saccade velocity
   - Fixation events
   - Pupil diameter and coordinates
   - Binocular Point of Regard (PoR)
4. **Time-Series Construction**: Frame-wise features aggregated into sequences
5. **Model Inference**:
   - **LSTM model** takes time-series gaze features as input
   - Outputs Parkinson’s probability

## 🗃 Project Structure

```
📁 eye_parkinsons_project/
│
├── main.py               # Entry script: video capture, gaze processing
├── pipeline.py           # Feature extraction and formatting into sequences
├── model.py              # LSTM model architecture and inference logic
├── utils.py              # EAR, saccade, blink, fixation, PoR calculations
├── mongo.py              # Optional: MongoDB logging of features/results
├── models/
│   └── lstm_model.pt     # Trained PyTorch LSTM model
├── data/
│   └── sample_video.mp4  # Sample input video
├── eye_metrics_output.csv # Feature log per frame
└── README.md
```

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy, Pandas
- PyTorch (for LSTM)
- Streamlit (optional for UI)
- MongoDB (optional for DB storage)

## 🚀 How to Run

### Process a Video:

```bash
python main.py --video data/sample_video.mp4
```

### Live Capture from Pi Camera:

```bash
python main.py --live
```

### (Optional) Streamlit Interface:

```bash
streamlit run app.py
```

### Output:

- `eye_metrics_output.csv`: all extracted gaze metrics
- Console / log: Parkinson’s prediction

## 📊 Sample Features Extracted

| Feature             | Description                      |
|---------------------|----------------------------------|
| EAR                | Eye Aspect Ratio (blink signal)  |
| Saccade Velocity   | Speed of gaze shift              |
| Fixation Flag      | Binary marker for fixation       |
| Blink Count        | Number of blinks per window      |
| Pupil Diameter     | Size of pupils (both eyes)       |
| Point of Regard    | (x,y) screen point of focus      |

## 🔮 Future Improvements

- Incorporate CNN + eye image data (if needed)
- Use larger annotated datasets for better generalization
- Export model to ONNX / TensorFlow Lite for Pi deployment
- Add UI for live feedback to patients/clinicians


