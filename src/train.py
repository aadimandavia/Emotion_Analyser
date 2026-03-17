from preprocess import preprocess
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def generate_action(row):

    if row["stress_level"] >= 8:
        return "grounding"

    elif row["stress_level"] >= 6:
        return "box_breathing"

    elif row["emotional_state"] == "focused" and row["energy_level"] > 6:
        return "deep_work"

    elif row["emotional_state"] == "happy":
        return "movement"

    elif row["energy_level"] < 3:
        return "rest"

    else:
        return "light_planning"
    


train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

X_train, X_test, tfidf, cat_columns = preprocess(train, test)

train["action_label"] = train.apply(generate_action, axis=1)

y_action = train["action_label"]

model_action = RandomForestClassifier(n_estimators=100, random_state=42)
model_action.fit(X_train, y_action)

y_state = train["emotional_state"]
y_intensity = train["intensity"]

# State model
model_state = RandomForestClassifier(n_estimators=100, random_state=42)
model_state.fit(X_train, y_state)

# Intensity model
model_intensity = RandomForestClassifier(n_estimators=100, random_state=42)
model_intensity.fit(X_train, y_intensity)

# Train stress model
y_stress = train["stress_level"]
model_stress = RandomForestClassifier(n_estimators=100, random_state=42)
model_stress.fit(X_train, y_stress)

# Train energy model
y_energy = train["energy_level"]
model_energy = RandomForestClassifier(n_estimators=100, random_state=42)
model_energy.fit(X_train, y_energy)


pickle.dump(model_state, open("models/state_model.pkl", "wb"))
pickle.dump(model_intensity, open("models/intensity_model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
pickle.dump(cat_columns, open("models/cat_columns.pkl", "wb"))
pickle.dump(model_stress, open("models/stress_model.pkl", "wb"))
pickle.dump(model_energy, open("models/energy_model.pkl", "wb"))
pickle.dump(model_action, open("models/action_model.pkl", "wb"))

print("Model saved successfully")