import requests
import json

data = {
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
    "salary": ">50K"}

r = requests.post("http://127.0.0.1:8000/prediction/",
                  data=json.dumps(data))
# r = requests.post("https://mldevopsdemo2.herokuapp.com/predict/", \
# data=json.dumps(data))

print(r.json())
