import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from Behance_appreciate_1M import dir_path

DATA = fr"{dir_path}\data_processed\final_dataset_with_pca.parquet"

def run():
    print("Loading data...")
    df = pd.read_parquet(DATA)

    # X = PCA1 only
    X = df[["pca1"]].values   # shape: (n,1)
    y = df["total_appreciates"].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    print("\n===== Regression: Popularity ~ PCA1 =====")
    print(f"Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R² Score: {model.score(X, y):.4f}")
    print("==========================================\n")

    # Make predictions for the regression line
    x_line = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_line = model.predict(x_line)

    # ==== BLACK THEME FIGURE ====
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("black")
    ax = plt.gca()
    ax.set_facecolor("black")

    # White gridlines
    ax.grid(color="white", alpha=0.25)

    # White axes spines
    for spine in ax.spines.values():
        spine.set_color("white")

    # White tick labels
    ax.tick_params(colors="white")

    # Scatter plot
    ax.scatter(df["pca1"], df["total_appreciates"],
               alpha=0.2, s=5, color="#8FE3E9", label="Artworks")

    # Regression line
    ax.plot(x_line, y_line, color="red", linewidth=3, label="Regression Line")

    # Axis labels + title
    ax.set_xlabel("PCA1 (Visual Style Dimension 1)", color="white")
    ax.set_ylabel("Total Appreciates (Popularity)", color="white")
    ax.set_title("Relationship Between Visual Style (PCA1) and Popularity", color="white")

    # Legend styling
    legend = ax.legend()
    plt.setp(legend.get_texts(), color="white")
    legend.get_frame().set_edgecolor("white")

    # ==== VISUALISATIONS FOLDER ====
    SAVE_DIR = fr"{dir_path}\visualisations"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save plot
    fig_path = os.path.join(SAVE_DIR, "regression1_pca1_plot.png")
    plt.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    print(f"Saved plot to: {fig_path}")

    # Save regression outputs to CSV
    csv_path = os.path.join(SAVE_DIR, "regression1_pca1_output.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "value"])
        writer.writerow(["intercept", model.intercept_])
        writer.writerow(["slope_pca1", model.coef_[0]])
        writer.writerow(["r2_score", model.score(X, y)])

    print(f"Saved regression output to: {csv_path}")

    plt.close()

if __name__ == "__main__":
    run()
