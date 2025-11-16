import pandas as pd
from src.build_features import build_features

def test_build_features_output_columns():
    df = pd.read_csv("data/raw/train.csv")
    transformed = build_features(df)

    expected_cols = ["TotalSF", "TotalBathrooms", "Age", "HasGarage", "Has2ndFloor", "Remodeled"]
    
    for col in expected_cols:
        assert col in transformed.columns
