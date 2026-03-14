# 🧠 Alzheimer’s Early Cognitive Screening using Eye Blink & Gaze Analysis

This project implements a **real-time eye blink detection and gaze tracking system** using **MediaPipe and OpenCV**.  
The objective is to analyze **blink patterns, gaze direction, and visual response behavior** that may help in **early cognitive screening related to neurological conditions such as Alzheimer’s disease**.

The system combines **eye blink analysis and gaze stimulus experiments** to measure behavioral patterns associated with attention, reaction time, and visual tracking.

---

# 📌 Project Overview

This system performs the following tasks:

- Real-time **face detection**
- **Eye landmark detection** using MediaPipe Face Mesh
- **Eye Aspect Ratio (EAR)** calculation
- **Blink detection** using EAR thresholding
- **Blink rate calculation**
- **Gaze direction detection**
- **Stimulus-based gaze experiment**
- **Reaction time measurement**
- **Saccade movement analysis**
- **Real-time EAR visualization**
- **Data logging to CSV for analysis**

---

# 🧠 Motivation

Research suggests that neurological disorders such as **Alzheimer’s disease and cognitive decline** may influence:

- Blink frequency  
- Blink duration  
- Eye movement patterns  
- Visual attention behavior  
- Reaction time to stimuli  

This project simulates a **basic cognitive screening framework** by analyzing:

- Blink frequency  
- Blink duration  
- Eye openness patterns  
- Gaze direction  
- Visual stimulus response  

These behavioral signals may provide **early indicators of cognitive impairment**.

---

# ⚙️ Technologies Used

The project uses the following technologies and libraries:

| Library | Purpose |
|------|------|
| Python | Programming language |
| OpenCV | Webcam video capture and image processing |
| MediaPipe FaceMesh | Facial landmark detection (468 landmarks) |
| NumPy | Mathematical computations |
| Matplotlib | Real-time plotting and visualization |
| CSV / Pandas | Data logging and analysis |

---

# 🔍 System Workflow

## Step 1 — Face Detection

The webcam captures real-time video frames.  
**MediaPipe Face Mesh** detects **468 facial landmarks**.

These include landmarks for:

- Eyes  
- Nose  
- Mouth  
- Face contour  

---

## Step 2 — Eye Landmark Extraction

Specific landmark indices corresponding to **left and right eyes** are extracted.

These landmarks are used for **EAR calculation**.

---

## Step 3 — Eye Aspect Ratio (EAR)

EAR measures the **vertical eye opening relative to eye width**.

### Formula

```
EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)
```

Where:

- p1–p6 represent eye landmarks  
- Vertical distances indicate eyelid distance  
- Horizontal distance represents eye width  

---

## Step 4 — Blink Detection

A blink is detected when:

```
EAR < Threshold
```

for a certain number of consecutive frames.

The system then:

- Increments blink counter  
- Records blink timestamp  
- Updates blink rate  

---

## Step 5 — Blink Rate Calculation

```
Blink Rate = Total Blinks / Time (minutes)
```

Blink irregularities may indicate:

- Reduced attention  
- Fatigue  
- Neurological abnormalities  

---

# 👁 Gaze Detection

The eye region is analyzed to determine **gaze direction** based on pupil movement.

Possible gaze directions:

- Left  
- Right  
- Center  

This helps analyze **visual attention behavior**.

---

# 🎯 Gaze Stimulus Experiment

The module:

```
gaze_stimulus_experiment.py
```

implements a **visual stimulus experiment** to measure gaze response.

### Experiment Procedure

1. A **stimulus point appears on the screen**
2. The user is asked to **look at the stimulus**
3. Eye movement is tracked
4. System measures response behavior

### Stimulus Positions

- Left  
- Right  
- Top  
- Bottom  
- Center  

---

## Metrics Recorded

The experiment measures:

- Stimulus direction  
- Gaze direction detected  
- Reaction time  
- Accuracy of response  
- Timestamp  

Example CSV output:

```
timestamp,stimulus_direction,gaze_direction,reaction_time,accuracy
```

This data helps analyze **visual attention and cognitive response speed**.

---

# 📊 Output

The system provides:

### Real-Time Metrics

- Blink counter  
- Blink rate  
- Eye openness (EAR)  
- Gaze direction  

### Visualization

- EAR graph over time  
- Blink pattern graph  
- Stimulus response tracking  

### Data Logging

All experiment data is stored in **CSV format** for later analysis.

---

# 📁 Project Structure

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
│
└── DATA/
```

---

# 🚀 Installation & Execution

## Clone Repository

```
git clone https://github.com/Syedabdulazeez-021/Alzhimers.git
cd Alzhimers
```

---

## Install Dependencies

```
pip install -r requirements.txt
```

or install manually

```
pip install opencv-python mediapipe numpy matplotlib
```

---

## Run Blink Detection

```
python blink_counter.py
```

---

## Run Gaze Stimulus Experiment

```
python gaze_stimulus_experiment.py
```

This will:

- Show visual stimulus points  
- Track gaze movement  
- Measure reaction time  
- Save results to CSV  

---

# 📈 Future Enhancements

- Machine learning based **cognitive scoring**
- Blink anomaly classification
- Dementia dataset integration
- Advanced gaze tracking algorithms
- Clinical validation
- Real-time cognitive monitoring dashboard

---

# 👨‍💻 Author

**Syed Abdul Azeez**  
B.Tech Computer Science  
AI & Computer Vision Enthusiast  

---

# ⚠ Disclaimer

This project is intended for **research and educational purposes only**.

It is **not a certified medical diagnostic tool** and should not be used as a substitute for professional medical evaluation.
