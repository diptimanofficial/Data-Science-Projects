import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from Behance_appreciate_1M import dir_path

# Enable interactive mode
plt.ion()

DATA = fr"{dir_path}\data_processed\final_dataset_with_pca.parquet"

def run():
    print("Loading dataset...")
    df = pd.read_parquet(DATA)

    print("Computing creator average popularity...")
    creator_pop = (
        df.groupby("owner_id")["total_appreciates"]
        .mean()
        .rename("owner_avg_popularity")
    )

    df = df.merge(creator_pop, on="owner_id", how="left")

    print("Running multivariate regression...")
    X = df[["pca1", "pca2", "owner_avg_popularity"]]
    y = df["total_appreciates"]

    model = LinearRegression()
    model.fit(X, y)

    print("\n========== REGRESSION 2: Visual Style + Creator Influence ==========")
    print(f"Coefficient PCA1:               {model.coef_[0]:.4f}")
    print(f"Coefficient PCA2:               {model.coef_[1]:.4f}")
    print(f"Creator Avg Popularity Coeff:   {model.coef_[2]:.4f}")
    print(f"Intercept:                      {model.intercept_:.4f}")
    print(f"R² Score:                       {model.score(X, y):.4f}")
    print("====================================================================\n")

    # ================= VISUALIZATION (INTERACTIVE) =================
    print("Generating interactive Actual vs Predicted plot...")

    y_pred = model.predict(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(y, y_pred, alpha=0.3, s=5, label="Artworks")

    # 45-degree reference line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             color="red", linewidth=2, label="Perfect Prediction")

    plt.xlabel("Actual Popularity (Total Appreciates)")
    plt.ylabel("Predicted Popularity")
    plt.title("Actual vs Predicted Popularity\n(Multivariate Regression Model)")
    plt.legend()

    # Interactive display
    plt.show()

if __name__ == "__main__":
    run()
