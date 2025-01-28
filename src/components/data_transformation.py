import sys 
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from src.logger import logging
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    """ Configuration for data transformation """
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

class CustomMapper(BaseEstimator, TransformerMixin):
    """Handles categorical data transformation"""
    def __init__(self,mappings):
        self.mappings = mappings

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.copy()
        for col, mapping in self.mappings.items():
            data[col] = data[col].map(mapping)
            if data[col].isnull().any():
                logging.warning(f"Null values found in column {col}")
                data.fillna({col: -2}, inplace=True)
        return data

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.copy()
        data.drop(columns=self.columns, inplace=True)
        return data
    
class DummyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns using one-hot encoding"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.copy()
        data = pd.get_dummies(data, columns=self.columns, drop_first=True)
        return data
    
class DataImputation(BaseEstimator, TransformerMixin):
    """Imputes missing values based on the number of unique values"""
    def __init__(self):
        pass
    
    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.copy()
        for feature in data.columns:
            if len(data[feature].value_counts()) < 5:
                imputer = IterativeImputer(estimator= RandomForestClassifier(), random_state=0)
            else:
                imputer = IterativeImputer(estimator= RandomForestRegressor(), random_state=0)
            data[feature] = imputer.fit_transform(data[feature].values.reshape(-1,1))
        return data

class DataTransformation:
    """ Orchestrates the data transformation pipeline """
    def __init__(self):
        self.config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        
        try:
            mappings = {
            'Gender': {'Male': 1, 'Female': -1},
            'Married': {'Yes': 1, 'No': -1},
            'Education': {'Graduate': 1, 'Not Graduate': -1},
            'Self_Employed': {'Yes': 1, 'No': -1},
            'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
            'Property_Area': {'Rural': 1, 'Semiurban': 2, 'Urban': 3},
            'Credit_History': {1.0: 1, 0.0: -1, np.nan: -1},
            }
            
            columns = [ "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Credit_History", "Loan_Status"]
    
            pipeline = Pipeline(
                steps=[
            ('dropper', ColumnDropper(['Loan_ID'])),
            ('mapper', CustomMapper(mappings)),
            ('encoder', DummyEncoder(columns)),
            ('imputer', DataImputation())
            ] )
            
            preprocessor = ColumnTransformer(
                [
                    ('pipeline', pipeline)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)     
  
    def save_preprocessor(self):
        """ Saves the preprocessor object """
        try:
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_path), exist_ok=True)
            joblib.dump(self.pipeline, self.config.preprocessor_obj_path)
            
            logging.info(f"Preprocessor saved at {self.config.preprocessor_obj_path}")
                    
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train and test data loaded")
            
            logging.info("Obtaining preprocessor object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Loan_Status"
            
            logging.info("Fitting preprocessor object")
            
            X_train = preprocessing_obj.fit_transform(train_df)
            X_test = preprocessing_obj.transform(test_df)
            
            logging.info(f" Saving preprocessor object at {self.config.preprocessor_obj_path}")
            
            save_obj(
                
                path = self.config.preprocessor_obj_path,
                obj = preprocessing_obj
                
            )
            
            return(X_train, X_test, self.config.preprocessor_obj_path)
        
        except Exception as e:
            raise CustomException(e,sys)
