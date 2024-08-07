
import yaml
import pickle

def load_schema(schema_file_path: str) -> dict:
    with open(schema_file_path, 'r') as file:
        schema = yaml.safe_load(file)
    return schema

def load_pkl(file_path):
    """Load an object from a pickle file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)