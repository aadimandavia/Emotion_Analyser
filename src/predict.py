
import pickle
import pandas as pd
from src.preprocess import preprocess_single

# Load models
model_state = pickle.load(open("models/state_model.pkl", "rb"))
model_intensity = pickle.load(open("models/intensity_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
cat_columns = pickle.load(open("models/cat_columns.pkl", "rb"))
model_stress = pickle.load(open("models/stress_model.pkl", "rb"))
model_energy = pickle.load(open("models/energy_model.pkl", "rb"))
model_action = pickle.load(open("models/action_model.pkl", "rb"))


def decide_action(state, intensity, stress, energy, time):

    time = str(time).lower()

    # -----------------------------
    # 1. STRESS OVERRIDE (safety first)
    # -----------------------------
    if stress >= 8:
        return "grounding", "now"

    if 6 <= stress < 8:
        return "box_breathing", "now" if intensity >= 4 else "within_15_min"

    # -----------------------------
    # 2. EMOTIONAL BASE ACTION
    # -----------------------------
    if state == "focused":
        action = "deep_work"
    elif state == "calm":
        action = "light_planning"
    elif state == "happy":
        action = "movement"
    elif state == "restless":
        action = "grounding"
    elif state == "overwhelmed":
        action = "box_breathing"
    else:
        action = "light_planning"


    if energy < 3 and state not in ["focused", "happy"]:
        action = "rest"


    if time == "night" and action == "deep_work":
        action = "journaling"


    if intensity >= 4:
        timing = "now"
    elif intensity == 3:
        timing = "within_15_min"
    else:
        timing = "later_today"

    return action, timing


def generate_message(state, action, intensity, confidence):

    # -----------------------------
    # Tone (emotion-aware)
    # -----------------------------
    if state in ["happy", "calm", "focused"]:
        tone = "This seems like a positive state."
    elif state in ["overwhelmed", "restless"]:
        tone = "It looks like things feel a bit heavy right now."
    elif intensity >= 4:
        tone = "This seems quite intense."
    elif intensity == 3:
        tone = "This looks noticeable."
    else:
        tone = "This seems mild."

    # -----------------------------
    # Confidence awareness
    # -----------------------------
    if confidence < 0.4:
        certainty = "I might be off, but"
    elif confidence < 0.7:
        certainty = "I may not be fully certain, but"
    else:
        certainty = "I'm fairly confident that"

    if action == "deep_work":
        action_phrase = "starting focused work"
    elif action == "rest":
        action_phrase = "taking a short rest"
    else:
        action_phrase = action

    # -----------------------------
    # Human-like phrasing
    # -----------------------------
    return f"{certainty} you're feeling {state}. {tone} A small step could help — try {action}."


def predict(input_dict):

    df = pd.DataFrame([input_dict])

    text = input_dict["journal_text"].lower()

    X_basic = preprocess_single(df, tfidf, cat_columns, include_targets=False)

    stress = int(model_stress.predict(X_basic)[0])
    energy = int(model_energy.predict(X_basic)[0])

    if any(word in text for word in ["energetic", "excited", "motivated"]):
        energy = max(energy, 7)

    if any(word in text for word in ["happy", "calm"]):
        stress = min(stress, 3)

    df["stress_level"] = stress
    df["energy_level"] = energy

    X = preprocess_single(df, tfidf, cat_columns, include_targets=True)

    state = model_state.predict(X)[0]
    intensity = int(model_intensity.predict(X)[0])

    probs = model_state.predict_proba(X)
    confidence = float(probs.max())

    positive_words = ["happy", "energetic", "excited", "motivated", "productive"]
    negative_words = ["tired", "exhausted", "stressed", "overwhelmed"]

    positive_score = sum(word in text for word in positive_words)
    negative_score = sum(word in text for word in negative_words)

    if positive_score >= 2:
        confidence = min(confidence + 0.4, 1.0)
    elif positive_score == 1:
        confidence = min(confidence + 0.2, 1.0)

    if negative_score >= 2:
        confidence = min(confidence + 0.3, 1.0)

    uncertain_flag = int(confidence < 0.5)

    action_rule, timing = decide_action(
        state,
        intensity,
        stress,
        energy,
        df["time_of_day"][0]
    )

    action_ml = model_action.predict(X)[0]

    if confidence > 0.7:
        action = action_ml
    else:
        action = action_rule

    if stress > 8:
        action = "grounding"
        timing = "now"

    message = generate_message(state, action, intensity, confidence)

    return {
        "state": state,
        "intensity": intensity,
        "confidence": round(confidence, 2),
        "uncertain": uncertain_flag,
        "stress": stress,
        "energy": energy,
        "action": action,
        "timing": timing,
        "message": message,
        "action_ml": action_ml,
        "action_rule": action_rule,
    }

#I feel really happy and energetic today, ready to do something productive