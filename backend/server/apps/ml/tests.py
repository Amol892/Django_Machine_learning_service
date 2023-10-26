from django.test import TestCase
from apps.ml.income_classifier.random_forest import RandomForestClassifier
from apps.ml.registry import MLRegistry
import inspect
class MLTests(TestCase):
    def test_rf_algorithum(self):
        input_data = {
            "age": 37,
            "workclass": "Private",
            "fnlwgt": 34146,
            "education": "HS-grad",
            "educational-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Craft-repair",
            "relationship": "Husband",
            "race": "White",
            'gender': "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 68,
            "native-country": "United-States"
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('<=50K', response['label'])
        
        

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        
        endpoint_name = "income_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Amol"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(RandomForestClassifier)
        print(algorithm_code)
        
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
        