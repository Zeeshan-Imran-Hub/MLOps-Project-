import os
import yaml
import pandas as pd
from heart_disease_prediction.entity.artifact_entity import DataValidationArtifact
from heart_disease_prediction.entity.config_entity import DataValidationConfig
from heart_disease_prediction.constants import schema_file_path, test_file_path
from heart_disease_prediction.utils import load_schema


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig = DataValidationConfig()):
        self.data_validation_config = data_validation_config
        self.schema_file_path = schema_file_path
        self.test_file_path = test_file_path
        self.schema = load_schema(self.schema_file_path)
        self.data = pd.read_csv(test_file_path)
        self.data = pd.read_csv(test_file_path).drop(columns=['_id'])

    def _load_schema(self) -> dict:
        with open(self.schema_file_path, 'r') as file:
            schema = yaml.safe_load(file)
        return schema

    def validate_column_names(self) -> bool:
        schema_columns = [list(col.keys())[0] for col in self.schema['columns']]
        data_columns = self.data.columns.tolist()
        if set(schema_columns) == set(data_columns):
            return True
        else:
            print("Column name validation failed")
            print("Expected columns:", schema_columns)
            print("Found columns:", data_columns)
            return False

    def validate_numerical_columns(self) -> bool:
        schema_numerical_columns = set(self.schema['numerical_columns'])
        data_numerical_columns = set(self.data.select_dtypes(include=['int64', 'float64']).columns)
        if schema_numerical_columns == data_numerical_columns:
            return True
        else:
            print("Numerical column validation failed")
            print("Expected numerical columns:", schema_numerical_columns)
            print("Found numerical columns:", data_numerical_columns)
            return False

    def validate_categorical_columns(self) -> bool:
        schema_categorical_columns = set(self.schema['categorical_columns'])
        data_categorical_columns = set(self.data.select_dtypes(include=['object']).columns)
        if schema_categorical_columns == data_categorical_columns:
            return True
        else:
            print("Categorical column validation failed")
            print("Expected categorical columns:", schema_categorical_columns)
            print("Found categorical columns:", data_categorical_columns)
            return False

    def initiate_data_validation(self) -> DataValidationArtifact:
        column_validation_status = self.validate_column_names()
        numerical_validation_status = self.validate_numerical_columns()
        categorical_validation_status = self.validate_categorical_columns()

        validation_status = column_validation_status and numerical_validation_status and categorical_validation_status
        if validation_status:
            message = "The data validation has been successful"
        else:
            message = "The data validation was not successful"
            if not column_validation_status:
                message += " due to column name validation failure."
            if not numerical_validation_status:
                message += " due to numerical column validation failure."
            if not categorical_validation_status:
                message += " due to categorical column validation failure."

        report = {
            'validation_status': validation_status,
            'column_validation_status': column_validation_status,
            'numerical_validation_status': numerical_validation_status,
            'categorical_validation_status': categorical_validation_status,
            'message': message
        }

        report_dir_path = self.data_validation_config.data_validation_dir
        os.makedirs(report_dir_path, exist_ok=True)
        report_file_path = self.data_validation_config.report_file_path
        with open(report_file_path, 'w') as report_file:
            yaml.dump(report, report_file)

        print(message)
        return DataValidationArtifact(validation_status, message, report_file_path)
