
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()


DATABASE_NAME = "heart_disease"

COLLECTION_NAME = "stroke_data"

#MONGODB_URL_KEY = os.environ.get("MONGODB_URL")


ARTIFACT_DIR: str = "artifact"


TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "data.csv"
MODEL_FILE_NAME = "model.pkl"

schema_file_path = "C:/Users/ASUS/Desktop/MLOps-Project-/config/schema.yaml"
test_file_path = "C:/Users/ASUS/Desktop/MLOps-Project-/artifact/07_30_2024_15_16_08/data_ingestion/ingested/test.csv"
train_file_path = "C:/Users/ASUS/Desktop/MLOps-Project-/artifact/07_30_2024_15_16_08/data_ingestion/ingested/train.csv"

transformed_test_file_path = "C:/Users/ASUS/Desktop/MLOps-Project-/artifact/08_06_2024_16_16_56/data_transformation/ingested/transformed_test.csv"
transformed_train_file_path = "C:/Users/ASUS/Desktop/MLOps-Project-/artifact/08_06_2024_16_16_56/data_transformation/ingested/transformed_train.csv"



"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "stroke_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


'''
Data Validation related constants start with DATA_VALIDATION VAR NAME
'''

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"


'''
Data Transform related constants start with DATA_VALIDATION VAR NAME
'''
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
TRANSFORMED_TRAIN_FILE_NAME: str = "transformed_train.csv"
TRANSFORMED_TEST_FILE_NAME: str = "transformed_test.csv"


'''
MODEL TRAINER related constant start with MODEL_TRAINER var name
'''
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_REPORT_FILE_NAME: str = "model_report.yaml"
