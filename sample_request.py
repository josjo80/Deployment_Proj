import requests
import json

data = {
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

r = requests.post("https://mldevopsdemo2.herokuapp.com/predict/", data=json.dumps(data))

print(r.json())