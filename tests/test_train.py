import os 
import joblib
from src.train import train_model

def test_model_training():
    train_model()
    
    os.path.exists("models/model.pkl"),"Model file not found"
    model = joblib.load("models/model.pkl")
    assert hasattr(model,"predict"),"Model does nothave a predict method"



    

    