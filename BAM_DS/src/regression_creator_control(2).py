import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from Behance_appreciate_1M import dir_path

DATA = fr"{dir_path}\data_processed\final_dataset_with_pca.parquet"

def run():
    print("Loading dataset...")
    df = pd.read_parquet(DATA)

    # Compute creator popularity
    creator_pop = (
        df.groupby("owner_id")["total_appreciates"]
        .mean()
        .rename("owner_avg_popularity")
    )
    df = df.merge(creator_pop, on="owner_id", how="left")

    # Multivariate regression
    X = df[["pca1", "pca2", "owner_avg_popularity"]]
    y = df["total_appreciates"]

    model = LinearRegression()
    model.fit(X, y)

    print("\n========== REGRESSION 2 RESULTS ==========")
    print(f"PCA1 Coefficient:               {model.coef_[0]:.4f}")
    print(f"PCA2 Coefficient:               {model.coef_[1]:.4f}")
    print(f"Creator Avg Popularity Coeff:   {model.coef_[2]:.4f}")
    print(f"Intercept:                      {model.intercept_:.4f}")
    print(f"R² Score:                       {model.score(X, y):.4f}")
    print("===========================================\n")

    # ================= VISUALIZATION ===================
    print("Generating Actual vs Predicted plot...")

    y_pred = model.predict(X)

    # ====== BLACK THEME FIGURE ======
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("black")
    ax = plt.gca()
    ax.set_facecolor("black")

    # White gridlines
    ax.grid(color="white", alpha=0.25)

    # White spines
    for spine in ax.spines.values():
        spine.set_color("white")

    # White tick labels
    ax.tick_params(colors="white")

    # Scatter
    ax.scatter(y, y_pred, alpha=0.3, s=6, color="#8FE3E9", label="Artworks")

    # Perfect prediction line
    ax.plot([y.min(), y.max()], [y.min(), y.max()],
            color="red", linewidth=3, label="Perfect Prediction")

    # Labels
    ax.set_xlabel("Actual Popularity (Total Appreciates)", color="white")
    ax.set_ylabel("Predicted Popularity", color="white")
    ax.set_title("Actual vs Predicted Popularity\n(Multivariate Regression)", color="white")

    # Legend styling
    legend = ax.legend()
    plt.setp(legend.get_texts(), color="white")
    legend.get_frame().set_edgecolor("white")

    # ====== SAVE OUTPUTS ======
    SAVE_DIR = fr"{dir_path}\BAM_DS\visualisations"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save plot
    fig_path = os.path.join(SAVE_DIR, "regression2_creator_control_plot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to: {fig_path}")

    # Save regression results
    csv_path = os.path.join(SAVE_DIR, "regression2_creator_control_output.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "value"])
        writer.writerow(["intercept", model.intercept_])
        writer.writerow(["coef_pca1", model.coef_[0]])
        writer.writerow(["coef_pca2", model.coef_[1]])
        writer.writerow(["coef_creator_avg_popularity", model.coef_[2]])
        writer.writerow(["r2_score", model.score(X, y)])

    print(f"Saved regression output to: {csv_path}")

    plt.close()

if __name__ == "__main__":
    run()
