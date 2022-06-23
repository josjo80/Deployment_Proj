from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

import joblib

from starter.ml.data import process_data
from starter.ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Declare the data object with its components and their type.
class DataSample(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = {
            'example': {
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
                "native_country": "United-States",
                "salary": ">50K"}
        }

loaded_model = joblib.load('starter/rf_model.pkl')
loaded_encoder = joblib.load('starter/encoder.pkl')
loaded_lb = joblib.load('starter/label_encoder.pkl')

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

app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/prediction")
async def prediction(sample: DataSample):
    
    sample = {key.replace('_','-'): [value] for key, value in data.__dict__.items()}
    data_sample = pd.DataFrame.from_dict(sample)
    data_sample = process_data(data_sample, 
                                categorical_features=cat_features, 
                                label=None, 
                                training=False, 
                                encoder=loaded_encoder, 
                                lb=loaded_lb)
    item = inference(loaded_model, data_sample)

    item = lb.inverse_transform(item)

    return item[0]

@app.get("/")
async def welcome():
    return {"welcome": "Welcome to my Salary prediction model!"}