import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from heart_disease_prediction.utils import load_pkl



class ModelEstimator:
    def __init__(self, model_path, preprocessor_path, test_data):
        self.preprocessor = load_pkl(preprocessor_path)
        self.model = load_pkl(model_path)
        self.test_data = test_data

    def predict(self, x_data):
        y_pred = self.model.predict(x_data)
        return y_pred

    def transform(self, x_data):
        return self.preprocessor.transform(x_data)


    def initiate_estimator(self):
        # Transform the test data using the preprocessor
        X_test_transformed = self.transform(self.test_data)

        # Make predictions
        y_pred = self.predict(X_test_transformed)
        return y_pred





