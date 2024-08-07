'''from heart_disease_prediction.components.data_transformation import DataTransformation
from heart_disease_prediction.entity.config_entity import DataTransformConfig

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initiate Data Transformation
data_transformation = DataTransformation(DataTransformConfig())
dt_art = data_transformation.initiate_data_transformation()
print(dt_art)


from heart_disease_prediction.components.model_trainer import ModelTrainer
from heart_disease_prediction.entity.config_entity import ModelTrainerConfig

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initiate Model Trainer
model_trainer_config = ModelTrainerConfig()
model_trainer = ModelTrainer(model_trainer_config)
model_trainer_artifact = model_trainer.initiate_model_trainer()
print(model_trainer_artifact)
'''



from heart_disease_prediction.components.model_estimator import ModelEstimator
import pandas as pd


model_path = 'C:/Users/ASUS/Desktop/MLOps-Project-/artifact/08_07_2024_12_22_06/model_trainer/trained_model/model.pkl'
preprocessor_path = 'C:/Users/ASUS/Desktop/MLOps-Project-/artifact/08_06_2024_16_16_56/data_transformation/preprocessor.pkl'
test_data_path = 'C:/Users/ASUS/Desktop/MLOps-Project-/artifact/07_30_2024_15_16_08/data_ingestion/ingested/test.csv'


test_data = pd.read_csv(test_data_path)
X_test = test_data.drop(columns=['stroke'])  # Adjust column name as needed
y_test = test_data['stroke']

model_estimator = ModelEstimator(model_path, preprocessor_path, X_test)
print(model_estimator.initiate_estimator())

