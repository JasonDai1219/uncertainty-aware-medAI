from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "ds_whole.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def split_and_save(test_size=0.2, calib_size=0.2, random_state=42):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV)
    y = df["glaucoma"].astype(int)
    X = df.drop(columns=["glaucoma"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + calib_size),
        stratify=y,
        random_state=random_state
    )

    rel = calib_size / (test_size + calib_size)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp,
        test_size=rel,
        stratify=y_temp,
        random_state=random_state
    )

    pd.concat([X_train, y_train], axis=1).to_pickle(PROCESSED_DIR / "train.pkl")
    pd.concat([X_calib, y_calib], axis=1).to_pickle(PROCESSED_DIR / "calib.pkl")
    pd.concat([X_test, y_test], axis=1).to_pickle(PROCESSED_DIR / "test.pkl")

    # print("Saved splits to:", PROCESSED_DIR)
    # print("Class counts (train/calib/test):",
    #       pd.Series(y_train).value_counts().to_dict(),
    #       pd.Series(y_calib).value_counts().to_dict(),
    #       pd.Series(y_test).value_counts().to_dict())

if __name__ == "__main__":
    split_and_save()