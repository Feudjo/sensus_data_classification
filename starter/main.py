# Put the code for your API here.
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from starter.train_model import cat_features
from starter.ml.data import process_data
from starter.ml.model import inference

class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship : str
    race: str
    sex: str
    capital_gain: float = Field(alias='capital-gain')
    capital_loss: float = Field(alias='capital-loss')
    hours_per_week: float = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


    class Config:
        allow_population_by_field_name = True,

        schema_extra ={
            'example':{
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship' : 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }




app = FastAPI()
@app.get("/")
async def home():
    return {"greeting": "Welcome!"}

@app.post('/predict')
def predict_salary(data: Person):

    model = joblib.load(os.path.join(os.getcwd(), 'model', 'clf_model.pkl'))
    encoder = joblib.load(os.path.join(os.getcwd(), 'model', 'encoder.pkl'))
    lb = joblib.load(os.path.join(os.getcwd(), 'model', 'lbibarizer.pkl'))

    data = data.dict()
    X = pd.DataFrame([data])
    X.rename({col: col.replace('_', '-') for col in X.columns} , axis='columns', inplace=True)
    X_test,  _, encoder, lb = process_data(X,
                          categorical_features=cat_features,
                          training=False,
                          encoder=encoder,
                          lb=lb)

    pred = inference(model, X_test)
    return {'salary_':f'{pred[-1]}'}





