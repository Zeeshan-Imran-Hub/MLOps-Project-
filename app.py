import os
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_initiate_data_ingestion():
    # Initialize the DataIngestionConfig with paths
    data_ingestion_config = DataIngestionConfig(
        feature_store_file_path="feature_store/test_feature_store.csv",
        training_file_path="ingested/test_train.csv",
        testing_file_path="ingested/test_test.csv",
        train_test_split_ratio=0.2
    )

    # Create an instance of DataIngestion
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

    # Test initiate_data_ingestion method
    try:
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        if (os.path.exists(data_ingestion_artifact.trained_file_path) and
                os.path.exists(data_ingestion_artifact.test_file_path)):
            print("initiate_data_ingestion passed.")
        else:
            print("initiate_data_ingestion failed: Train or test file not created.")
    except Exception as e:
        print(f"initiate_data_ingestion failed: {e}")

if __name__ == "__main__":
    test_initiate_data_ingestion()
