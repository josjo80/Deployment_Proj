from . import model
from . import data
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sample = {
    "age": 48,
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

data_samples = pd.read_csv("./census_mod.csv")
loaded_model = joblib.load('./starter/rf_model.pkl')
loaded_encoder = joblib.load('./starter/encoder.pkl')
loaded_lb = joblib.load('./starter/label_enc.pkl')

train, test = train_test_split(data_samples, test_size=0.20)

X_test, y_test, _, _ = data.process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=loaded_encoder, lb=loaded_lb
)

pred = model.inference(loaded_model, X_test)


def test_save_model():
    '''
    Function tests save_model function
    '''

    assert os.path.exists('./starter/rf_model.pkl')
    assert os.path.exists('./starter/encoder.pkl')
    assert os.path.exists('./starter/label_enc.pkl')


def test_predict():
    '''
    Function tests predict function
    '''

    assert isinstance(pred, np.ndarray)


def test_compute_model_metrics():
    '''
    Function tests compute_model_metrics function
    '''

    prec, _, _ = model.compute_model_metrics(y_test, pred)
    assert isinstance(prec, float)
