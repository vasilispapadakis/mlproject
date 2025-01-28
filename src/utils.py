import os 
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill


def save_obj(obj, path):
    """ Save object to file """
    try:
       dir_path = os.path.dirname(path)
       os.makedirs(dir_path, exist_ok=True)
       
       with open(path, 'wb') as file:
           dill.dump(obj, file)
    
    except Exception as e:
        raise CustomException(e,sys)