import joblib
import pandas as pd
from django.core.cache import cache
import os


def load_and_cache_joblib(file_path):
    # generate a unique cache key based on the file_path
    cache_key = f"pickle_{hash(file_path)}"
    
    # Try to retrieve data from the cache
    data = cache.get(cache_key)
        
    if data is not None:
        return data
    
    # If data is None 
    with open(file_path, 'rb') as fh:
        data = joblib.load(fh)

    # store the data in the cache for subsequence request
    
    cache.set(cache_key, data, timeout=None)
    
    return data
    

    
class RandomForestClassifier:
    
    def __init__(self):
        
        self.values_fill_missing =  load_and_cache_joblib(os.path.join('research', 'train_mode.joblib'))
        self.encoders = load_and_cache_joblib(os.path.join('research', 'encoders.joblib'))
        self.model = load_and_cache_joblib(os.path.join('research','random_forest.joblib'))
        
        
    def preprocessing(self, input_data):
        
        # JSON to DataFrame convertion
        input_data = pd.DataFrame(input_data, index=[0])
        
        # Filling missing values
        print(self.values_fill_missing)
        input_data = input_data.fillna(self.values_fill_missing)
        print('1->',input_data)
        # Converting categorical values to numerical values
        
        cat_columns = ['workclass','education','marital-status','occupation','relationship','race','gender','native-country']
        for column in cat_columns:
            categorical_converter = self.encoders[column]
            input_data[column]=categorical_converter.transform(input_data[column])
        print('2->',input_data)
        return input_data
    
    # predicting using trained model
    def predict(self, input_data):
        
        return self.model.predict_proba(input_data)
    
    def postprocessing(self, input_data):
        label = "<=50K"
        print('4->',input_data)
        if input_data[1] > 0.5:
            
            label = ">50K"
        return {"probability": input_data[1], "label": label, "status": "OK"}
    
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            print('5->',self.predict(input_data))
            prediction = self.predict(input_data)[0]  # only one sample
            print('6->',prediction)
            prediction = self.postprocessing(prediction)
            print(prediction)
        except Exception as e:
            print(str(e))
            return {"status": "Error", "message": str(e)}

        return prediction