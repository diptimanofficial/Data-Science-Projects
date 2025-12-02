import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Behance_appreciate_1M import dir_path


# Path to your final dataset
DATA = fr"{dir_path}\data_processed\final_dataset.parquet"

def run():
    print("RUN FUNCTION STARTED")
    print("Loading final dataset...")
    df = pd.read_parquet(DATA)

    # Only keep the 4096 visual embedding columns
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    x = df[feature_cols].values

    print("Shape of visual feature matrix:", x.shape)

    # Standardize Data
    print("Standardizing features...")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # PCA to 2 dimensions
    print("Running PCA...")
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    df["pca1"] = x_pca[:, 0]
    df["pca2"] = x_pca[:, 1]

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.scatter(
        df["pca1"],
        df["pca2"],
        c=df["total_appreciates"],
        cmap="viridis",
        s=5,
        alpha=0.7
    )

    plt.colorbar(label="Total Appreciates")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Artwork Visual Embeddings (Colored by popularity)")
    plt.show()

    # Save the PCA Output
    OUT = fr"{dir_path}\data_processed\final_dataset_with_pca.parquet"
    df.to_parquet(OUT)
    print(f"PCA-enhanced dataset saved to:\n{OUT}")


if __name__ == "__main__":
    run()
