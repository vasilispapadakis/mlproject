import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, data):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_processed = preprocessor.transform(data)
            data_processed = data_processed.drop(columns=['Loan_Status'])
            prediction = model.predict(data_processed)
            return prediction
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)
    

class CustomData:
    """ Custom data for prediction, "Gender", "Married", "Dependents", "Education", "Self_Employed", 
    "Property_Area", "Credit_History", "Loan_Amount_Term", "LoanAmount", "ApplicantIncome", "CoapplicantIncome" """
    
    def __init__(self, gender: str, married: str, dependents: str, education: str, self_employed: str, 
                 property_area: str, credit_history: int, loan_amount_term: int, loan_amount: float, 
                 applicant_income: float, coapplicant_income: float):
        
        self.gender = gender
        self.married = married
        self.dependents = dependents
        self.education = education
        self.self_employed = self_employed
        self.property_area = property_area
        self.credit_history = credit_history
        self.loan_amount_term = loan_amount_term
        self.loan_amount = loan_amount
        self.applicant_income = applicant_income
        self.coapplicant_income = coapplicant_income
    
    def get_data_as_df(self):
        try:
            custom_data_dict = {
                "Gender": [self.gender],
                "Married": [self.married],
                "Dependents": [self.dependents],
                "Education": [self.education],
                "Self_Employed": [self.self_employed],
                "Property_Area": [self.property_area],
                "Credit_History": [self.credit_history],
                "Loan_Amount_Term": [self.loan_amount_term],
                "LoanAmount": [self.loan_amount],
                "ApplicantIncome": [self.applicant_income],
                "CoapplicantIncome": [self.coapplicant_income]
            }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            logging.error(f"Error in getting data as dataframe: {str(e)}")
            raise CustomException(e, sys)

