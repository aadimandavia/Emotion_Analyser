import pandas as pd
from src.predict import predict

test = pd.read_csv("data/Test.csv")
results = []

for i in range(len(test)):

    row = test.iloc[i]

    input_dict = {
        "journal_text": row["journal_text"],
        "ambience_type": row["ambience_type"],
        "duration_min": row["duration_min"],
        "sleep_hours": row["sleep_hours"],
        "time_of_day": row["time_of_day"],
        "previous_day_mood": row["previous_day_mood"],
        "face_emotion_hint": row["face_emotion_hint"],
        "reflection_quality": row["reflection_quality"]
    }

    output = predict(input_dict)

    results.append({
        "id": row["id"],
        "predicted_state": output["state"],
        "predicted_intensity": output["intensity"],
        "confidence": output["confidence"],
        "uncertain_flag": output["uncertain"],
        "what_to_do": output["action"],
        "when_to_do": output["timing"],
        "support_message": output["message"]
    })

df = pd.DataFrame(results)

df.to_csv("outputs/predictions.csv", index=False)