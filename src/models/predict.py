# Contents of /mlops-product-classification/mlops-product-classification/src/models/predict.py

import joblib
import pandas as pd


class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, input_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(input_data)


if __name__ == "__main__":
    import json
    import sys

    # Load input data from JSON file
    input_file = sys.argv[1]
    with open(input_file, "r") as f:
        input_data = pd.DataFrame(json.load(f))

    # Initialize the predictor
    predictor = ModelPredictor(model_path="path/to/your/model.pkl")

    # Make predictions
    predictions = predictor.predict(input_data)

    # Output predictions
    print(predictions.tolist())
