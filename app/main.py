from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    journal_text: str
    ambience_type: str
    duration_min: float
    sleep_hours: float
    time_of_day: str
    previous_day_mood: str
    face_emotion_hint: str
    reflection_quality: str


# ------------------------
# Routes
# ------------------------
@app.get("/")
def home():
    return {"message": "Emotion AI API is running "}


@app.post("/predict")
def get_prediction(data: UserInput):
    
    input_dict = data.dict()
    
    result = predict(input_dict)
    
    return result