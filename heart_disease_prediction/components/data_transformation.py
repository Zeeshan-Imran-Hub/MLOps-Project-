import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from heart_disease_prediction.constants import test_file_path, train_file_path
from heart_disease_prediction.entity.config_entity import DataTransformConfig
from heart_disease_prediction.entity.artifact_entity import DataTransformArtifact


class DataTransformation:
    def __init__(self, data_transform_config: DataTransformConfig = DataTransformConfig()):
        self.data_transform_config = data_transform_config
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.transformed_train_file_path = data_transform_config.transformed_training_file_path
        self.transformed_test_file_path = data_transform_config.transformed_testing_file_path
        self.preprocessor_file_path = os.path.join(data_transform_config.data_transform_dir, 'preprocessor.pkl')
        os.makedirs(os.path.dirname(self.transformed_train_file_path), exist_ok=True)

    def read_data(self):
        print(f"Reading train data from: {self.train_file_path}")
        print(f"Reading test data from: {self.test_file_path}")

        train_df = pd.read_csv(self.train_file_path)
        test_df = pd.read_csv(self.test_file_path)
        return train_df, test_df

    def transform_data(self, df):
        x = df.copy()
        x.drop(columns=['_id', 'id'], inplace=True)
        y = x.pop('stroke')
        cat_features = [feature for feature in x.columns if x[feature].dtype == 'object']
        num_features = [feature for feature in x.columns if feature not in cat_features]
        transform_features = ['avg_glucose_level', 'bmi']

        imputer = KNNImputer(n_neighbors=5)
        x[['bmi']] = imputer.fit_transform(x[['bmi']])

        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder(sparse=False)
        transform_pipeline = Pipeline(steps=[('transformer', PowerTransformer(method='yeo-johnson'))])

        preprocessor = ColumnTransformer([
            ('StandardScaler', numeric_transformer, num_features),
            ('OneHotEncoder', oh_transformer, cat_features),
            ('Transformer', transform_pipeline, transform_features)
        ])

        x_transformed = preprocessor.fit_transform(x)

        # Get transformed feature names
        num_feature_names = num_features
        cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(cat_features)
        transform_feature_names = transform_features

        all_feature_names = np.concatenate([num_feature_names, cat_feature_names, transform_feature_names])

        return x_transformed, y, all_feature_names, preprocessor

    def balance_data(self, x, y):
        smt = SMOTEENN(random_state=42, sampling_strategy='minority')
        x_resampled, y_resampled = smt.fit_resample(x, y)
        return x_resampled, y_resampled

    def save_transformed_data(self, x_train, y_train, x_test, y_test, feature_names, preprocessor):
        train_combined = np.hstack((x_train, y_train.values.reshape(-1, 1)))
        test_combined = np.hstack((x_test, y_test.values.reshape(-1, 1)))

        # Convert to DataFrame with column names
        train_df = pd.DataFrame(train_combined, columns=np.append(feature_names, 'stroke'))
        test_df = pd.DataFrame(test_combined, columns=np.append(feature_names, 'stroke'))

        # Save to CSV
        train_df.to_csv(self.transformed_train_file_path, index=False, header=True)
        test_df.to_csv(self.transformed_test_file_path, index=False, header=True)

        # Save preprocessor object
        with open(self.preprocessor_file_path, 'wb') as filehandler:
            pickle.dump(preprocessor, filehandler)

    def initiate_data_transformation(self) -> DataTransformArtifact:
        train_df, test_df = self.read_data()

        x_train, y_train, feature_names, preprocessor = self.transform_data(train_df)
        x_test, y_test, _, _ = self.transform_data(test_df)

        x_train_resampled, y_train_resampled = self.balance_data(x_train, y_train)
        x_test_resampled, y_test_resampled = self.balance_data(x_test, y_test)

        self.save_transformed_data(x_train_resampled, y_train_resampled, x_test_resampled, y_test_resampled,
                                   feature_names, preprocessor)
        print("Data transformation, balancing, and preprocessor saving completed successfully")

        return DataTransformArtifact(
            transformed_trained_file_path=self.transformed_train_file_path,
            transformed_test_file_path=self.transformed_test_file_path
        )
