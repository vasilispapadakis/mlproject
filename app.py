from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = CustomData(
            gender = request.form.get('gender'),
            married = request.form.get('married'),
            dependents = request.form.get('dependents'),
            education = request.form.get('education'),
            self_employed = request.form.get('self_employed'),
            property_area = request.form.get('property_area'),
            credit_history = int(request.form.get('credit_history')),
            loan_amount_term = int(request.form.get('loan_amount_term')),
            loan_amount = float(request.form.get('loan_amount')),
            applicant_income = float(request.form.get('applicant_income')),
            coapplicant_income = float(request.form.get('coapplicant_income'))
        )
        data_df = data.get_data_as_df()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data_df)
        return render_template('home.html', prediction=prediction[0])
    else:
        return render_template('home.html')
            

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_data, test_data, _ =  data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTraining()
    model_trainer.initiate_model_training(train_data, test_data)
    app.run(host='0.0.0.0',port = 8000)