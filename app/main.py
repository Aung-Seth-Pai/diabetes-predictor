# turn model into a FastAPI app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np

# define input data model
class PatientData(BaseModel):
    # attributes info ( features are already normalized )
    age: float  # age in years
    sex: float
    bmi: float  # body mass index
    bp: float   # avg blood pressure
    s1: float   # tc, total serum cholesterol
    s2: float   # ldl, low-density lipoproteins
    s3: float   # hdl, high-density lipoproteins
    s4: float   # tch, total cholesterol / HDL
    s5: float   # ltg, possibly log of serum triglycerides level
    s6: float   # glu, blood sugar level

    class Config:
        schema_extra = {
            "example": {
                "age": 0.05,
                "sex": 0.05,
                "bmi": 0.06,
                "bp": 0.02,
                "s1": -0.04,
                "s2": -0.04,
                "s3": -0.02,
                "s4": -0.01,
                "s5": 0.01,
                "s6": 0.02
            }
        }

# define output data model
class PredictionResponse(BaseModel):
    predicted_progression_score: float
    interpretation: str
    note: str

    class Config:
        schema_extra = {
            "example": {
                "predicted_progression_score": 150.0,
                "interpretation": "Moderate progression rate",
                "note": "Score represents disease progression one year after baseline"
            }
        }

# create FastAPI app
app = FastAPI(
    title="Diabetes Progression Predictor",
    description="API for predicting diabetes progression using 10 physiological features",
    version="1.0.0"
)

# load the trained model
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
model_path = os.path.join(model_dir, "diabetes_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)
# confirm model loaded
if model:
    print("Model loaded successfully from disk.")

# create prediction endpoint
@app.post("/predict", response_model=PredictionResponse,)
def predict_diabetes_progression(patient: PatientData):
    """
    Predict diabetes progression based on patient data.
    
    :param patient_data: PatientData object containing physiological features
    :return: Predicted diabetes progression value
    """

    # convert input data to numpy array
    input_data = np.array([
        patient.age, patient.sex, patient.bmi, patient.bp,
        patient.s1, patient.s2, patient.s3, patient.s4, patient.s5, patient.s6
    ]).reshape(1, -1)

    # make prediction
    try:
        prediction = model.predict(input_data)

        # basic validation 
        if prediction is None:
            raise ValueError("Prediction returned None.")
        if prediction.size == 0:
            raise ValueError("Prediction returned an empty result.")
        # check if prediction is a numpy array
        if not isinstance(prediction, np.ndarray):
            raise TypeError("Prediction is not a numpy array.")
        # ensure prediction is a 1D array
        if prediction.ndim > 1:
            prediction = prediction.flatten()
        print(prediction) # debug print

        pred_value = float(prediction[0])  # ensure it's a float

        # Warning for unusual values falling outside the range from sample data
        if pred_value < 25 or pred_value > 346:
            print(f"Warning: Prediction {pred_value} is outside observed range [25, 346]")
        
        return {
            "predicted_progression_score": round(pred_value, 2),
            "interpretation": get_interpretation(pred_value),
            "note": "Score represents disease progression one year after baseline"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def get_interpretation(prediction: float) -> str:
    """
    Interpret the predicted diabetes progression score.
    Note: The interpretation thresholds are only examples and should be defined based on clinical significance.
    In practice, this should be based on clinical guidelines or expert consensus.
    The thresholds used here are arbitrary and for demonstration purposes only.
    """
    if not isinstance(prediction, (int, float)):
        raise ValueError("Prediction must be a numeric value")
        
    # Using quartile-based interpretation (example thresholds)
    if prediction < 100:
        return "Lower quartile progression rate"
    elif prediction < 200:
        return "Moderate progression rate"
    elif prediction < 300:
        return "Higher quartile progression rate"
    else:
        return "Very high progression rate"

# create root endpoint
@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Diabetes Progression Predictor API!"}

# create health check endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is operational.
    """
    return {"status": "healthy", "model_loaded": "diabetes_progression_v1"}

# run the app using uvicorn
# To run the app, use the command:
# uvicorn main:app --reload --port 8000

# This will start the FastAPI server and you can access the API at localhost:8000.
# You can also access the interactive API documentation at:
# http://localhost:8000/docs
# Note: Make sure to have uvicorn installed in your environment.