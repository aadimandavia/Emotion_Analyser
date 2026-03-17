🌿 ArvyaX ML Internship Assignment

Emotion Understanding → Decision → Guidance System

📌 Overview

This project builds an intelligent system that goes beyond prediction.
It understands human emotional signals from noisy reflections and recommends meaningful actions.

The system is designed as a hybrid AI pipeline combining machine learning + reasoning.

🎯 Objectives

The system performs:

Emotional Understanding

Predict emotional state

Predict intensity (1–5)

Decision Layer

What should the user do?

When should they do it?

Uncertainty Awareness

Confidence score

Uncertainty flag

Supportive Feedback

Human-like guidance message

🧠 System Architecture
User Input (journal + context)
        ↓
Preprocessing (TF-IDF + encoding)
        ↓
ML Models:
   - Emotion (classification)
   - Intensity (classification)
   - Stress (regression/classification)
   - Energy (regression/classification)
        ↓
Semantic Correction Layer
        ↓
Hybrid Decision Engine:
   ML Action Model + Rule-based Reasoning
        ↓
Confidence Calibration
        ↓
Final Output (Action + Timing + Message)
⚙️ Tech Stack

Python

scikit-learn

RandomForest

TF-IDF Vectorizer

FastAPI (API)

HTML + CSS (UI)

📊 Feature Engineering
Text Features

TF-IDF (max_features=500)

Captures key emotional words

Categorical Features

ambience_type

time_of_day

previous_day_mood

face_emotion_hint

reflection_quality

(Encoded using one-hot encoding)

Numerical Features

sleep_hours

duration_min

energy_level

stress_level

🤖 Models Used
Task	Model	Type
Emotional State	RandomForest	Classification
Intensity	RandomForest	Classification
Stress	RandomForest	Regression/Classification
Energy	RandomForest	Regression/Classification
Action	RandomForest	Classification
🧩 Why Classification for Intensity?

Intensity is discrete (1–5), so it is treated as a multi-class classification problem rather than regression.

🧠 Decision Engine (Core Innovation)

The system uses a hybrid approach:

1. ML Action Model

Learns patterns from pseudo-labels

2. Rule-Based Reasoning

Ensures logical consistency

Handles edge cases

3. Safety Overrides

High stress → grounding

Low energy → rest

🔍 Uncertainty Modeling

Confidence = max probability from model

Uncertain if confidence < 0.5

Enhancement:

A semantic calibration layer adjusts confidence based on text signals.

🧪 Ablation Study
Model	Performance
Text only	Moderate
Text + Metadata	Significantly better ✅

👉 Metadata (stress, energy, sleep) greatly improves predictions.

⚠️ Error Analysis
Common Failure Cases

Short Text

"ok", "fine"

No signal → low confidence

Ambiguous Emotion

"I feel weird"

Model confused between classes

Conflicting Signals

"tired but happy"

Mixed predictions

Noisy Labels

Same text → different labels in dataset

🧠 Improvements Applied

Semantic correction layer (text-aware)

Confidence calibration

Hybrid decision system

Safety override logic

📱 Edge / Deployment Plan
On-device deployment:

Replace RandomForest with:

LightGBM / XGBoost (optimized)

OR small neural model

Optimizations:

Reduce TF-IDF features

Quantization

Batch inference

Trade-offs:
Factor	Tradeoff
Accuracy	vs Speed
Model size	vs performance
🛠️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train models
python src/train.py
3. Run API
uvicorn app.main:app --reload
4. Open UI
http://127.0.0.1:8000
📁 Project Structure
Emotion-Analyser/
│
├── data/
├── models/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── app/
│   ├── main.py
│
├── templates/
├── static/
├── outputs/
└── README.md
🌟 Key Highlights

✅ Handles noisy real-world data
✅ Hybrid ML + reasoning system
✅ Uncertainty-aware predictions
✅ Product-oriented decision making
✅ End-to-end pipeline + UI