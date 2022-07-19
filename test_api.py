import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from main import app
import pandas as pd

client = TestClient(app)

def test_api_root():

    response = client.get('/')
    
    assert response.status_code == 200
    assert response.json() == {"Welcome to my Salary prediction model!"}

def test_sample_neg():

    samp = {"age": 37,
            "workclass": "Private",
            "fnlgt":284582,
            "education": "Masters",
            "education_num":14,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
    }

    #samp = pd.DataFrame.from_dict(samp)

    response = client.post('/prediction/', json=samp)

    assert response.status_code == 200
    assert response.json() == '<=50k'

def test_sample_pos():

    samp = {
            "age":48,
            "workclass": "State-gov",
            "fnlgt": 327886,
            "education": "Doctorate",
            "education_num": 16,
            "marital_status": "Divorced",
            "occupation": "Prof-specialty",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
    }

    #samp = pd.DataFrame.from_dict(samp)

    response = client.post('/prediction/', json=samp)

    assert response.status_code == 200
    assert response.json() == '>50k'