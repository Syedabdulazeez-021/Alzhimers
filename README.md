# 🧠 Alzheimer’s Early Cognitive Screening using Eye Blink & Gaze Analysis

This project implements a real-time eye blink detection and gaze tracking system using **MediaPipe** and **OpenCV**.  
The objective is to analyze blink patterns and gaze behavior that may contribute to early cognitive assessment related to neurological conditions such as Alzheimer’s disease.

---

## 📌 Project Overview

This system performs:

- Real-time face detection
- Eye landmark detection using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) calculation
- Blink detection using thresholding
- Gaze tracking experiment with stimulus points
- Blink rate calculation
- Real-time EAR plotting
- Data logging to CSV

---

## 🧠 Motivation

Research indicates that:

- Abnormal blink rate may correlate with cognitive decline
- Gaze fixation irregularities may reflect attention and memory impairment
- Delayed visual response to stimuli may indicate neurological dysfunction

This project simulates a basic cognitive screening framework by analyzing:

- Blink frequency
- Blink duration
- Gaze direction
- Stimulus response behavior

---

## ⚙️ Technologies Used

- Python
- OpenCV
- MediaPipe FaceMesh
- NumPy
- Matplotlib / Plotly (for visualization)
- Custom GUI (if implemented)

---

## 🔍 System Workflow

### Step 1: Face Detection
MediaPipe FaceMesh detects 468 facial landmarks in real-time.

### Step 2: Eye Landmark Extraction
Specific landmark indices corresponding to left and right eyes are extracted.

### Step 3: Eye Aspect Ratio (EAR) Calculation

EAR formula:

EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)

When EAR drops below a predefined threshold for consecutive frames → a blink is detected.

### Step 4: Blink Counter
A counter increments when EAR remains below threshold for a minimum number of frames.

### Step 5: Gaze Detection
The eye region is analyzed to determine gaze direction based on pupil positioning.

### Step 6: Stimulus Experiment
Stimulus points appear on the screen, and gaze movement is tracked to measure response.

---

## 📊 Output

The system provides:

- Real-time blink counter
- EAR graph visualization
- Gaze direction detection
- CSV data logging for further analysis
- Stimulus response behavior

---

## 📁 Project Structure

```
Alzhimers/
│
├── blink_counter.py
├── blink_counter_and_EAR_plot.py
├── blink_gaze_mediapipe.py
├── blink_gaze_gui.py
├── gaze_stimulus_experiment.py
├── FaceMeshModule.py
├── utils.py
├── requirements.txt
└── DATA/
```

---

## 🚀 Installation & Execution

### 1️⃣ Clone Repository

```
git clone https://github.com/Syedabdulazeez-021/Alzhimers.git
cd Alzhimers
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Blink Detection

```
python blink_counter.py
```

### 4️⃣ Run Gaze Experiment

```
python gaze_stimulus_experiment.py
```

---

## 📈 Future Enhancements

- Machine learning based cognitive scoring
- Blink anomaly classification
- Dementia dataset integration
- Performance optimization
- Clinical validation support

---

## 👨‍💻 Author

**Syed Abdul Azeez**  
B.Tech Computer Science  
AI & Computer Vision Enthusiast  

---

## ⚠ Disclaimer

This project is a research-oriented implementation and is not a certified medical diagnostic tool.
