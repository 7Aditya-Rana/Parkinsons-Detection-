# 🧠 Parkinson’s Disease Detection via Eye-Tracking

An edge-based eye-tracking system for early Parkinson’s disease detection using high-frame-rate video analysis and LSTM-based temporal modeling. The project focuses on robust data preprocessing, meaningful eye-movement feature extraction, and privacy-preserving deployment on low-cost hardware.

---

## ✨ Highlights

- High-frame-rate (100 FPS) eye movement capture
- Eye landmark detection using **MediaPipe Face Mesh**
- Gaze and eye-movement feature extraction using **OpenCV + NumPy**
- Noise-aware data preprocessing and behavioral labeling
- Temporal modeling with an **LSTM neural network**
- Designed for **edge deployment (Raspberry Pi)**
- Offline processing to ensure data privacy

---

## 🧩 System Pipeline

1. **Video Acquisition**  
   Eye movement videos are recorded using a high-speed camera to capture subtle eye dynamics.

2. **Landmark Detection**  
   Facial and eye landmarks are detected using MediaPipe to localize eye regions.

3. **Feature Extraction**  
   Frame-wise features extracted include:
   - Eye Aspect Ratio (EAR)
   - Blink detection
   - Saccade velocity
   - Fixation (visual intake) events
   - Pupil diameter and coordinates
   - Binocular Point of Regard (PoR)

4. **Data Preprocessing**
   - Noise reduction using median filtering  
   - Handling of missing and blank frames  
   - Removal of spurious micro-movements  
   - Semantic labeling of eye behavior  

5. **Temporal Modeling**
   Cleaned features are structured into fixed-length sequences for time-series learning.

6. **Model Inference**
   An **LSTM model** analyzes eye-movement patterns and outputs Parkinson’s probability.

---

## 📁 Project Structure

```text
eye_parkinsons_project/
│
├── main.py                    # Entry point for pipeline execution
├── pipeline.py                # Feature extraction and sequence construction
├── preprocessing.py           # Noise removal and data cleaning
├── model.py                   # LSTM model architecture and inference
├── utils.py                   # EAR, blink, saccade, fixation, PoR utilities
├── models/
│   └── lstm_model.pt          # Trained LSTM model
├── data/
│   └── sample_video.mp4       # Example input video
├── eye_metrics_output.csv     # Frame-wise extracted features
└── README.md
```
## ⚙️ Requirements & Usage

### Installation
```bash
pip install -r requirements.txt

```
Core Dependencies
-Python 3.8+
-OpenCV
-MediaPipe
-NumPy
-Pandas
-PyTorch

Optional Dependencies
-Docker (for containerized deployment)
-Streamlit (for UI)

🚀 Usage
Process a Video File
python main.py --video data/sample_video.mp4

Live Capture (Edge Device)
python main.py --live
## 📓 Jupyter Notebooks

The following notebooks document the complete data preprocessing and model development workflow:

- **Data Preprocessing – Parkinson’s Patients**  
  `notebooks/preprocessing_parkinsons.ipynb`

- **Data Preprocessing – Healthy Controls**  
  `notebooks/preprocessing_healthy.ipynb`

- **Final LSTM Model Training and Evaluation**  
  `notebooks/final_lstm_model.ipynb`
📤 Output

eye_metrics_output.csv – cleaned and structured gaze features

Console output – Parkinson’s disease prediction with confidence score

🐳 Docker Integration

The complete pipeline is containerized using Docker, which ensures:
-consistent execution across different systems,
-simplified dependency and environment management,
-reproducible experimental results,
-easy deployment on edge devices such as Raspberry Pi


🔮 Future Work

-Validation on larger clinical datasets
-Multimodal fusion (speech, handwriting)
-Edge-optimized inference (ONNX / TFLite)
-Real-time clinician feedback interface
