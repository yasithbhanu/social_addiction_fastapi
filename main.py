from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import JSONResponse
from fastapi import Request, status
from fastapi.middleware.cors import CORSMiddleware


# Load the pre-trained model
model = joblib.load("social_media_addiction_model.pkl")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow localhost for testing and your deployed URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Custom 404 handler
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "The endpoint you requested was not found. Please check the URL and try again."},
    )

# Define the input schema using Pydantic
class PredictionInput(BaseModel):
    age: int
    daily_usage_time: float
    posts_per_day: float
    likes_received_per_day: float
    comments_received_per_day: float
    messages_sent_per_day: float
    gender: str
    platform: str
    dominant_emotion: str

# Function to preprocess the input data
def preprocess_input(data: PredictionInput):
    # Numerical features
    features = [
        data.age, 
        data.daily_usage_time, 
        data.posts_per_day, 
        data.likes_received_per_day, 
        data.comments_received_per_day, 
        data.messages_sent_per_day
    ]
    
    # Gender encoding (only one encoding for gender if "Female" was dropped during training)
    features += [1 if data.gender == "Male" else 0]  # Only include "Male" encoding

    # Platform encoding
    platforms = ["Instagram", "LinkedIn", "Snapchat", "Telegram", "Twitter", "Whatsapp"]
    features += [1 if data.platform == p else 0 for p in platforms]
    
    # Dominant emotion encoding
    emotions = ["Anxiety", "Boredom", "Happiness", "Neutral", "Sadness"]
    features += [1 if data.dominant_emotion == e else 0 for e in emotions]
    
    return np.array(features).reshape(1, -1)

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    # Preprocess the input data
    input_data = preprocess_input(data)
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Format the prediction result
    result = "At Risk of Addiction" if prediction[0] else "Not at Risk of Addiction"
    return {"prediction": result}
