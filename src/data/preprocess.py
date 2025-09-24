import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONT_COLS = [
    "age",
    "ocular_pressure",
    "MD",
    "PSD",
    "cornea_thickness",
    "RNFL4.mean",
]

CAT_COLS = ["RL", "GHT"]

def build_preprocessor():
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONT_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop",
    )
    return ct

def split_X_y(df: pd.DataFrame):
    y = df["glaucoma"].astype(int).values
    X = df.drop(columns=["glaucoma"])
    return X, y