import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from heart_disease_prediction.constants import DATABASE_NAME, COLLECTION_NAME
from pymongo import MongoClient

from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact


# from heart_disease_prediction.exception import heart_disease_prediction_exception
# from heart_disease_prediction.logger import logging

import certifi

ca = certifi.where()

class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            print("Exception Occured: ", e)
        # raise heart_disease_prediction_exception(e, sys)

    def import_data_as_dataframe(self) -> DataFrame:
        """
        Method Name :   import_data_as_dataframe
        Description :   This method imports data from MongoDB and loads it into a DataFrame

        Output      :   DataFrame containing the data from MongoDB
        """
        try:
            mongodb_url = os.getenv('MONGODB_URL')
            if not mongodb_url:
                raise ValueError("MONGODB_URL_KEY environment variable not defined")
            client = MongoClient(mongodb_url, tlsCAFile=ca)
            database = client[DATABASE_NAME]
            collection = database[COLLECTION_NAME]
            data = list(collection.find())
            if not data:
                raise Exception("No data found in the collection.")
            dataframe = pd.DataFrame(data)

            return dataframe
        except Exception as e:
            print(pd.DataFrame())
            print("Exception 1 Occured: ", e)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file

        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
           # logging.info(f"Exporting data from mongodb")
            ### create a class method which reads the data from db and returns the dataframe
            #### Your code below
            dataframe = self.import_data_as_dataframe()

           # logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
          #  logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            print("Exception Occured: ", e)
        # raise heart_disease_prediction_exception(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
      #  logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
       #     logging.info("Performed train test split on the dataframe")
        #    logging.info(   "Exited split_data_as_train_test method of Data_Ingestion class")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

         #   logging.info(f"Exporting train and test file path.")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

          #  logging.info(f"Exported train and test file path.")
        except Exception as e:
            print("Exception 2 Occured: ", e)
    #  raise heart_disease_prediction_exception(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline

        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        #logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            #logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            # logging.info("Performed train test split on the dataset")

           # logging.info(    "Exited initiate_data_ingestion method of Data_Ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_path=self.data_ingestion_config.feature_store_file_path,
            )

            # logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            print("Exception Occured: ", e)
         # raise heart_disease_prediction_exception(e, sys) from e

