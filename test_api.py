import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_api_root():

    response = client.get('/')

    assert response.status_code == 200
    assert response.json() == {
        'welcome': 'Welcome to my Salary prediction model!'}


def test_sample_neg():

    samp = {"age": 37,
            "workclass": "Private",
            "fnlgt": 284582,
            "education": "9th",
            "education_num": 14,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 16,
            "native_country": "Jamaica",
            "salary": '<=50K'
            }

    response = client.post('prediction', json=samp)

    assert response.status_code == 200
    assert response.json() == '<=50K'


def test_sample_pos():

    samp = {
        "age": 48,
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
        "native_country": "United-States",
        "salary": ">50K"
    }

    response = client.post('prediction', json=samp)

    assert response.status_code == 200
    assert response.json() == '>50K'
