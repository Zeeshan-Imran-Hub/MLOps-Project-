from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str
    feature_store_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    report_file_path: str

@dataclass
class DataTransformArtifact:
    transformed_trained_file_path: str
    transformed_test_file_path: str
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    report_artifact: str


