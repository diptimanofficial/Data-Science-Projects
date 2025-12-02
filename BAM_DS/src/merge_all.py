import pandas as pd
import os
from Behance_appreciate_1M import dir_path


#Paths
appreciates = fr"{dir_path}\data_processed\appreciates.parquet"
owners = fr"{dir_path}\data_processed\item_owners.parquet"
features = fr"{dir_path}\data_processed\image_features.parquet"

Out = fr"{dir_path}\data_processed\final_dataset.parquet"

def run():
    print("\n================")
    print("Loading processed datasets....")
    print("\n================")

    df_app = pd.read_parquet(appreciates)
    df_own = pd.read_parquet(owners)
    df_feat = pd.read_parquet(features)

    df_app["item_id"] = df_app["item_id"].astype(str)
    df_own["item_id"] = df_own["item_id"].astype(str)
    df_feat["item_id"] = df_feat["item_id"].astype(str)

    print(f"Appreciates: {df_app.shape}")
    print(f"Owners:      {df_own.shape}")
    print(f"Features:    {df_feat.shape}")

    # 1. Aggregate appreciates
    print("\nAggregating appreciates...")

    df_app["count"] = 1

    popularity = df_app.groupby("item_id").agg(
        total_appreciates=("count", "sum"), 
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max")
    ).reset_index()

    print("Popularity table: ", popularity.shape)

    # 2. Merge: owners + popularity
    print("\nMerging owners + popularity")

    merged = df_own.merge(popularity, on="item_id", how="left")

    print("After merging owners + popularity: ", merged.shape)

    # 3. Merged with visual features
    print("\nMerging with visual features (4096-dim vectors)...")

    merged = merged.merge(df_feat, on="item_id", how="left")
    print("After merging features: ", merged.shape)

    print("How many feature rows matched:", merged["f_0"].notna().sum())


    # 4. Final cleaning
    print("\nCleaning final dataset...")

    # Items with no appreciates get 0
    merged["total_appreciates"] = merged["total_appreciates"].fillna(0)

    # Replace missing timestamps with -1 (never appreciated)
    merged["first_timestamp"] = merged["first_timestamp"].fillna(-1)
    merged["last_timestamp"] = merged["last_timestamp"].fillna(-1)

    merged = merged.dropna(subset=["f_0"])

    print("After cleaning: ", merged.shape)

    # 5. Save final dataset
    print("\nSaving final dataset...")

    out_dir = os.path.dirname(Out)
    os.makedirs(out_dir, exist_ok=True)

    merged.to_parquet(Out)

    print(f"\nDone! Final dataset saved to: \n{Out}")
    print("\nColumns: ", len(merged.columns))
    print("Preview:\n", merged.head())

if __name__ == "__main__":
    run()