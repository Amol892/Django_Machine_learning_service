import joblib
import pandas as pd
from django.core.cache import cache
import os


def load_and_cache_joblib(file_path):
    
    # generate unique cache key based on the file_path
    cache_key = f"joblib_{hash(file_path)}"
    
    # retrieve data from cache using cache_key
    
    data =cache.get(cache_key)
    
    if data is not None:
        return data
    
    # If data is None
    with open(file_path, 'rb') as fh:
        data = joblib.load(fh)
        
    # store data into cache
    
    cache.set(cache_key,data, timeout=None)
    
    return data


# extra tree classifier class
class ExtraTreesClassifier:
    
    def __init__(self):
        self.values_fill_missing =  load_and_cache_joblib(os.path.join('research', 'train_mode.joblib'))
        self.encoders = load_and_cache_joblib(os.path.join('research', 'encoders.joblib'))
        self.model = load_and_cache_joblib(os.path.join('research','random_forest.joblib'))
        
        
    def preprocessing(self,input_data):
        # JSON to DatFrame convertion with single row having index 0
        input_data = pd.DataFrame(input_data, index=[0])
        
        # Filling missing values
        input_data = input_data.fillna(self.values_fill_missing)
        
        
        # converting categorical values to numerical values
        cat_columns = ['workclass','education','marital-status','occupation','relationship','race','gender','native-country']
        for column in cat_columns:
            categorical_converter = self.encoders[column]
            input_data[column]=categorical_converter.transform(input_data[column])

        return input_data

        # predicting using trained model
    def predict(self, input_data):
        
        return self.model.predict_proba(input_data)
    
    def postprocessing(self, input_data):
        label = "<=50K"
        if input_data[1] > 0.5:
            label = ">50K"
        return {"probability": input_data[1], "label": label, "status": "OK"}
    
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
            print(prediction)
        except Exception as e:
            print(str(e))
            return {"status": "Error", "message": str(e)}

        return prediction