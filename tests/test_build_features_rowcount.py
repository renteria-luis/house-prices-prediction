import pandas as pd
from src.build_features import build_features

def test_build_features_preserves_row_count():
    df = pd.read_csv("data/raw/train.csv")
    transformed = build_features(df)

    assert len(df) == len(transformed), "Row count changed after feature engineering"