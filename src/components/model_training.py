from dataclasses import dataclass
import os
import sys
from src.logger import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTraining:
    """ Class to train the model """
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
    
    def initiate_model_training(self, train_data, test_data):
        logging.info("Initiated model training")
        
        try:
            logging.info("Splitting data into features and target")
            X_train = train_data.drop(columns=['Loan_Status'])
            y_train = train_data['Loan_Status']
            X_test = test_data.drop(columns=['Loan_Status'])
            y_test = test_data['Loan_Status']
            
            model = RandomForestClassifier(random_state=1)
            model.fit(X_train, y_train)
            logging.info("Model trained")
            
            y_pred = model.predict(X_test)
            
            # Evaluate model
            logging.info("Evaluating model")
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model accuracy: {accuracy}")
            
            # Save model
            logging.info(f"Saving model at {self.model_training_config.trained_model_path}")
            save_obj(model, self.model_training_config.trained_model_path)
            logging.info("Model saved")
            
            return accuracy
                        
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e,sys)
    