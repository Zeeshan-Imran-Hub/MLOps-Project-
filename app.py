import os
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

dataIngestion = DataIngestion(DataIngestionConfig())
di_art = dataIngestion.initiate_data_ingestion()
print(di_art)

