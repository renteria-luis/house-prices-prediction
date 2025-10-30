import numpy as np
import pandas as pd

def build_features(df):
    # Ordinal maps
    qual_map = {np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    fin_map = {np.nan: 0, 'Rec': 1, 'BLQ': 2, 'LwQ': 3, 'ALQ': 4, 'Unf': 3, 'GLQ': 4}

    ord_cols = {
        'ExterQual': qual_map,
        'ExterCond': qual_map,
        'BsmtQual': qual_map,
        'BsmtCond': qual_map,
        'HeatingQC': qual_map,
        'KitchenQual': qual_map,
        'GarageQual': qual_map,
        'GarageCond': qual_map,
        'BsmtFinType1': fin_map,
    }

    # Apply ordinal mapping
    for col, mapping in ord_cols.items():
        if col in df.columns:
            df[f'{col}_ord'] = df[col].map(mapping)

    df_ords_qual = df.loc[:, 'ExterQual_ord':'GarageCond_ord']

    # Engineered features
    df['TotalSF'] = df['GrLivArea'] + 0.8 * df['TotalBsmtSF'] + df['LotFrontage'].fillna(0) * 0.1
    df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 0.8 * (df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']))
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['PorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['ComponentsQual'] = df_ords_qual.mean(axis=1).round(3)
    df['HasBsmt'] = df['BsmtQual'].notna().astype(int)
    df['HasGarage'] = df['GarageQual'].notna().astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasPool'] = (df['PoolQC'] == 'Ex').astype(int)
    df['Remodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['MoSold_cat'] = df['MoSold'].map({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'})

    # Final feature selection
    cat_features = ['MSZoning', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Foundation', 'BsmtExposure', 'GarageType', 'GarageFinish', 'MoSold_cat']

    num_features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 
                    'BsmtFinType1_ord', 'TotalSF', 'TotalBathrooms', 'Age', 'PorchSF', 'ComponentsQual', 'HasBsmt', 'HasGarage', 'Has2ndFloor', 'HasPool', 'Remodeled']

    selected_features = cat_features + num_features

    return df[selected_features]
