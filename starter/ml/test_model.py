from model import compute_model_metrics, inference
from data import process_data
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sample = {
"age":48,
"workclass": "State-gov",
"fnlgt": 327886,
"education": "Doctorate",
"education-num": 16,
"marital-status": "Divorced",
"occupation": "Prof-specialty",
"relationship": "Own-child",
"race": "White",
"sex": "Male",
"capital-gain": 0,
"capital-loss": 0,
"hours-per-week": 50,
"native-country": "United-States",
"salary": ">50K"}

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

data = pd.read_csv("../../census_mod.csv")
loaded_model = joblib.load('../rf_model.pkl')
loaded_encoder = joblib.load('../encoder.pkl')
loaded_lb = joblib.load('../label_enc.pkl')

train, test = train_test_split(data, test_size=0.20)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=loaded_encoder, lb=loaded_lb
)

pred = inference(loaded_model, X_test)


def test_save_model():
    '''
    Function tests save_model function
    '''

    assert os.file.isfile('rf_model.pkl')
    assert os.file.isfile('encoder.pkl')
    assert os.file.isfile('label_enc.pkl')

def test_predict():
    '''
    Function tests predict function
    '''
    
    assert isinstance(pred, np.array)


def test_compute_model_metrics():
    '''
    Function tests compute_model_metrics function
    '''

    prec, _, _ = compute_model_metrics(y_test, pred)
    assert isinstance(prec, float)