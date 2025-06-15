# Diabetes Progression Predictor API

A FastAPI-based REST API that predicts diabetes disease progression using machine learning. The model is trained on the scikit-learn diabetes dataset and uses a Random Forest Regressor to make predictions.

## Features

- Predicts diabetes progression based on 10 physiological features
- RESTful API built with FastAPI
- Docker support for easy deployment
- Includes model training script
- Comprehensive input validation
- Health check endpoint

## Project Structure

```
diabetes-predictor/
├── app/
│   ├── __init__.py
│   └── main.py
├── models/
│   └── diabetes_model.pkl
├── Dockerfile
├── requirements.txt
├── train_model.py
└── README.md
```

## Prerequisites

- Python 3.12+
- Docker (optional)

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv diabetes-env
source diabetes-env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

1. Train the model (optional, model file is included):
```bash
python train_model.py
```

2. Start the FastAPI server:
```bash
cd app
uvicorn main:app --reload --port 8000
```

3. Access the API:
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Make predictions using the `/predict` endpoint

### Using Docker

1. Build the Docker image:
```bash
docker build -t diabetes-predictor .
```

2. Run the container:
```bash
docker run -p 8000:8000 diabetes-predictor
```

## API Endpoints

### POST /predict

Make a prediction for diabetes progression. Example request body:

```json
{
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
```

### GET /health

Check if the API is operational.

## Model Information

- Uses Random Forest Regressor
- Trained on the scikit-learn diabetes dataset
- Features are normalized
- Prediction range: 25-346 (disease progression score)

## License

[MIT License](LICENSE)
