#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict(p_year, p_mileage, p_state, p_make, p_model):
    model = joblib.load(os.path.dirname(__file__) + '/vehicle_price_model.pkl')    
    encoder = joblib.load(os.path.dirname(__file__) + '/encoder.joblib')
    preprocessor = joblib.load(os.path.dirname(__file__) + '/preprocessor.joblib')

    record = pd.DataFrame(
        [[p_year, p_mileage, p_state, p_make, p_model]],
        columns=['Year', 'Mileage', 'State', 'Make', 'Model']
    )
  
    # Transform categorical values
    record_encoded = encoder.transform(record)    

    # Scale numerical values
    record_processed  = preprocessor.transform(record_encoded)
    
    # Make prediction
    p1 = model.predict(record_processed)[0]

    return p1

        