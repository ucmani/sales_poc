# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("knn_api")

# Create input/output pydantic models
input_model = create_model("knn_api_input", **{'id': 'c7c2e3ab-2b22-4302-a503-32726aa31590', 'first_channel': 'DIGITAL_MARKETING', 'first_activity_datetime': Timestamp('2022-02-19 00:00:00'), 'first_touch_point': 'REQUEST_A_DEMO', 'first_state': 'MADHYA_PRADESH', 'first_graduation_year': 2007, 'total_discovery_events_touched': 2, 'demo_watched': 0, 'demo_watched_date': '3/8/2023', 'enrolled_datetime': NaT, 'activity_duration_days': nan})
output_model = create_model("knn_api_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
