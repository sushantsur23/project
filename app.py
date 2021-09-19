
import uvicorn
from fastapi import FastAPI
from linear import Linearassgn
import numpy as np
import pickle
import pandas as pd

# Create the app object
app = FastAPI()
pickle_in = open("linear_reg.pkl","rb")
linear=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Linear Regression Assignment'}

@app.post('/predict')
def predict_atmosphere_pres(data:Linearassgn):
    data = data.dict()
    PT=data['PT']
    RS=data['RS']
    Torque=data['Torque']
    TW=data['TW']
    TWF=data['TWF']
    HDF=data['HDF']
    PWF=data['PWF']
    OSF=data['OSF']
    RNF=data['RNF']
    prediction = linear.predict([PT, RS, Torque, TW, TWF, HDF, PWF, OSF, RNF])
    return {
        'prediction': prediction[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload