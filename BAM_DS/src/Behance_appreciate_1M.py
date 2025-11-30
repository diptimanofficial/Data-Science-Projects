import pandas as pd

dir_path = r"C:\Users\vikra\OneDrive\Desktop\Diptiman_DS\BAM_DS"

RAW = fr"{dir_path}\data_raw\Behance_appreciate_1M"
OUT = fr"{dir_path}\data_processed\appreciates.parquet"

def run():
    print("Loading appreciates...")

    df = pd.read_csv(
    RAW,
    sep=" ",
    names=["user_id", "item_id", "timestamp"],
    dtype={"item_id": str}
)

    print(df.head())
    df.to_parquet(OUT)
    print("Saved to: ", OUT)

if __name__=="__main__":
    run()
