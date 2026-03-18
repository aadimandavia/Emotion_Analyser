# 🌿 ArvyaX ML Internship Assignment
## Emotion Understanding → Decision → Guidance System

---

## 📌 Overview

Most AI systems stop at prediction. This one goes further.

**ArvyaX** is an intelligent pipeline that reads noisy human reflections — like journal entries written at the end of a long day — and transforms them into structured emotional understanding, reasoned decisions, and meaningful, human-like guidance.

It's not just a classifier. It's a hybrid ML + reasoning system built to actually help people.

---

## 🎯 What the System Does

The pipeline handles four interconnected tasks:

**1. Emotional Understanding**
- Predicts the user's emotional state (e.g., anxious, calm, overwhelmed)
- Estimates emotional intensity on a scale of 1–5

**2. Decision Layer**
- Recommends what the user should do
- Suggests *when* they should do it

**3. Uncertainty Awareness**
- Outputs a confidence score per prediction
- Flags low-confidence outputs so the user knows when to take guidance with a grain of salt

**4. Supportive Feedback**
- Generates a warm, human-like message — not a robotic label

---

## 🧠 System Architecture
```
User Input (journal entry + contextual metadata)
        ↓
Preprocessing
  └── TF-IDF vectorization + categorical encoding
        ↓
ML Models
  ├── Emotion Classifier
  ├── Intensity Classifier
  ├── Stress Model (regression/classification)
  └── Energy Model (regression/classification)
        ↓
Semantic Correction Layer
  └── Adjusts predictions based on text tone signals
        ↓
Hybrid Decision Engine
  ├── ML Action Model (learned from pseudo-labels)
  └── Rule-Based Reasoning (edge cases + safety overrides)
        ↓
Confidence Calibration
        ↓
Final Output
  └── Recommended action + timing + supportive message
```

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| ML Models | scikit-learn (RandomForest) |
| Text Features | TF-IDF Vectorizer |
| API | FastAPI |
| UI | HTML + CSS |

---

## 📊 Feature Engineering

The system combines three types of features to build a rich picture of the user's state:

**Text Features**
- TF-IDF with `max_features=500` — captures emotionally meaningful words from journal entries

**Categorical Features** *(one-hot encoded)*
- `ambience_type` — the environment the user was in
- `time_of_day` — morning / afternoon / evening / night
- `previous_day_mood` — yesterday's emotional state
- `face_emotion_hint` — optional visual cue from camera
- `reflection_quality` — coherence of the journal entry

**Numerical Features**
- `sleep_hours`
- `duration_min`
- `energy_level`
- `stress_level`

---

## 🤖 Models at a Glance

| Task | Model | Type |
|---|---|---|
| Emotional State | RandomForest | Classification |
| Intensity (1–5) | RandomForest | Classification |
| Stress Level | RandomForest | Regression / Classification |
| Energy Level | RandomForest | Regression / Classification |
| Recommended Action | RandomForest | Classification |

> **Why classification for intensity?**
> Intensity is a discrete 1–5 scale, so it naturally fits a multi-class classification approach rather than continuous regression. This keeps outputs interpretable and consistent.

---

## 🧩 Decision Engine — The Core Innovation

The decision layer isn't just another model. It's a **three-layer hybrid**:

**Layer 1 — ML Action Model**
Learns patterns from pseudo-labeled training data to recommend context-appropriate actions.

**Layer 2 — Rule-Based Reasoning**
Ensures logical consistency and handles edge cases that confuse the ML model.

**Layer 3 — Safety Overrides**
Hard rules that activate in high-risk situations:
- High stress detected → recommend grounding exercises
- Low energy detected → recommend rest before anything else

This layered design means the system degrades gracefully — even when the ML model is uncertain, the rules keep recommendations sensible.

---

## 🔍 Uncertainty Modeling

- **Confidence** = the maximum class probability from the model's output
- **Uncertain** = flagged when confidence < 0.5

On top of this, a **semantic calibration layer** reads the text for uncertainty signals (hedging language, ambiguity, contradiction) and adjusts the confidence score accordingly — so the output reflects both model certainty *and* input quality.

---

## 🧪 Ablation Study

| Feature Set | Result |
|---|---|
| Text features only | Moderate performance |
| Text + Metadata | Significantly better ✅ |

Adding structured metadata (stress levels, sleep hours, energy) dramatically improves the quality of predictions. Raw text alone just doesn't capture enough signal.

---

## ⚠️ Known Failure Cases & How We Handle Them

| Failure Case | Example | Mitigation |
|---|---|---|
| Short / empty text | *"ok", "fine"* | Detected → confidence flagged low |
| Ambiguous emotion | *"I feel weird"* | Semantic calibration layer applied |
| Conflicting signals | *"tired but happy"* | Hybrid decision engine reconciles |
| Noisy labels | Same text → different labels | Semantic correction + calibration |

---

## 📱 Edge / Deployment Considerations

For on-device or low-latency deployment, the following trade-offs apply:

| Factor | Trade-off |
|---|---|
| Accuracy | vs. Speed |
| Model size | vs. Performance |

**Recommended path for production:**
- Replace RandomForest with **LightGBM / XGBoost** (faster, lighter)
- Or swap in a small neural model
- Reduce TF-IDF features from 500 → ~100–200
- Apply quantization for edge hardware
- Enable batch inference where possible

---

## 🛠️ How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train models**
```bash
python src/train.py
```

**3. Start the API**
```bash
uvicorn app.main:app --reload
```

**4. Open the UI**
```
http://127.0.0.1:8000
```

---

## 📁 Project Structure
```
Emotion-Analyser/
│
├── data/               # Raw and processed datasets
├── models/             # Saved trained models
│
├── src/
│   ├── preprocess.py   # Feature engineering + encoding
│   ├── train.py        # Model training pipeline
│   └── predict.py      # Inference logic
│
├── app/
│   └── main.py         # FastAPI application
│
├── Ui/
|   └── index.html      # Simple Frontend
├── outputs/            # Prediction outputs and logs
└── README.md
```

---

## 🌟 Key Highlights

- ✅ Handles noisy, real-world journal data gracefully
- ✅ Hybrid ML + rule-based reasoning for robust decisions
- ✅ Uncertainty-aware — the system knows when it doesn't know
- ✅ Product-oriented design with end-user guidance in mind
- ✅ Full pipeline from raw input to UI — nothing left out

---

*Built as part of the ArvyaX ML Internship Program.*
