import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.Pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

##Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods = ['GET', 'POST'])
def predict_data_point():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            id = int(request.form.get('id')),
            gender=request.form.get('gender'),
            age=int(request.form.get('age')),
            hypertension=int(request.form.get('hypertension')),
            heart_disease=int(request.form.get('heart_disease')),
            ever_married=request.form.get('ever_married'),
            work_type=request.form.get('work_type'),
            Residence_type=request.form.get('Residence_type'),
            avg_glucose_level=float(request.form.get('avg_glucose_level')),
            bmi=float(request.form.get('bmi')),
            smoking_status=request.form.get('smoking_status'),
        )

        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred_results = predict_pipeline.predict(pred_df)
        return render_template('home.html', result = f'{'High' if pred_results[0] == 'yes' else 'Low'}')

if __name__ == '__main__':
    app.run(host = '0.0.0.0')