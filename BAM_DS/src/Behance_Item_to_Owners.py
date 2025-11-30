import pandas as pd
import os
from Behance_appreciate_1M import dir_path

RAW = fr"{dir_path}\data_raw\Behance_Item_to_Owners"
OUT = fr"{dir_path}\data_processed\item_owners.parquet"

def run():
    print("Loading item --> owner mappings...")

    #Ensure output directory exists
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df = pd.read_csv(
    RAW,
    sep=" ",
    names=["item_id", "owner_id"],
    dtype={"item_id": str}
)

    print(df.head())
    df.to_parquet(OUT)
    print("Saved to: ", OUT)

if __name__ == "__main__":
    run()


