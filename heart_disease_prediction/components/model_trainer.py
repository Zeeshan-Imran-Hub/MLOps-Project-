import os
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from heart_disease_prediction.entity.config_entity import ModelTrainerConfig
from heart_disease_prediction.entity.artifact_entity import ModelTrainerArtifact
from heart_disease_prediction.constants import transformed_test_file_path, transformed_train_file_path

from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        self.model_trainer_config = model_trainer_config

    def read_data(self):
        train_df = pd.read_csv(transformed_train_file_path)
        test_df = pd.read_csv(transformed_test_file_path)
        return train_df, test_df

    def evaluate_clf(self, true, predicted):
        acc = accuracy_score(true, predicted)
        f1 = f1_score(true, predicted)
        precision = precision_score(true, predicted)
        recall = recall_score(true, predicted)
        roc_auc = roc_auc_score(true, predicted)
        return acc, f1, precision, recall, roc_auc

    def load_models_from_yaml(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            model_configs = yaml.safe_load(file)['model_selection']
        models = {}
        search_params = {}
        for module in model_configs:
            module_info = model_configs[module]
            model_class = getattr(__import__(module_info['module'], fromlist=[module_info['class']]),
                                  module_info['class'])
            model_instance = model_class(**module_info['params'])
            models[module_info['class']] = model_instance
            search_params[module_info['class']] = module_info['search_param_grid']
        return models, search_params

    def tune_hyperparameters(self, X_train, y_train, models, search_params):
        best_params = {}
        for model_name, model in models.items():
            param_grid = search_params[model_name]
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=0, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_params[model_name] = grid_search.best_params_
            models[model_name] = grid_search.best_estimator_
        return models, best_params

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        best_model = None
        best_accuracy = 0
        best_model_details = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc, f1, precision, recall, roc_auc = self.evaluate_clf(y_test, y_pred)

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_model_details = {
                    "best_model_name": model_name,
                    "Accuracy": acc,
                    "f1_score": f1,
                    "Precision": precision,
                    "Recall": recall,
                    "Roc_Auc": roc_auc
                }

        return best_model, best_model_details

    def initiate_model_trainer(self):
        train_df, test_df = self.read_data()

        X_train = train_df.drop(columns=['stroke'])
        y_train = train_df['stroke']
        X_test = test_df.drop(columns=['stroke'])
        y_test = test_df['stroke']

        models, search_params = self.load_models_from_yaml(self.model_trainer_config.model_config_file_path)
        models, best_params = self.tune_hyperparameters(X_train, y_train, models, search_params)
        best_model, best_model_details = self.evaluate_models(X_train, y_train, X_test, y_test, models)

        if best_model_details['Accuracy'] >= self.model_trainer_config.expected_accuracy:
            model_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as model_file:
                pickle.dump(best_model, model_file)

            report = {
                'best_model_name': best_model_details["best_model_name"],
                'Accuracy': float(best_model_details["Accuracy"]),
                'f1_score': float(best_model_details["f1_score"]),
                'Precision': float(best_model_details["Precision"]),
                'Recall': float(best_model_details["Recall"]),
                'Roc_Auc': float(best_model_details["Roc_Auc"]),
                'best_params': best_params[best_model_details["best_model_name"]]
            }
            report_path = self.model_trainer_config.model_report_file_path
            with open(report_path, 'w') as report_file:
                yaml.dump(report, report_file)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_path,
                report_artifact=report_path
            )

            return model_trainer_artifact
        else:
            raise Exception("No model achieved the expected accuracy.")
