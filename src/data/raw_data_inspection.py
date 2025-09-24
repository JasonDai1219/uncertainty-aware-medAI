from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "ds_whole.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def inspect_raw_data(path: Path):
    df = pd.read_csv(path)

    print("Data Info")
    print(df.info(), "\n")

    print("Missing Values Info")
    print(df.isnull().sum(), "\n")

    print("Basic Summary")
    print(df.describe(include="all"), "\n")

    print("Label_distribution")
    print(df["glaucoma"].value_counts(normalize=True), "\n")

    return df

if __name__ == "__main__":
    print("Reading:", RAW_CSV)
    df = inspect_raw_data(RAW_CSV)