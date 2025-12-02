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

    # Compute artwork age
    df["age"] = df["last_timestamp"] - df["first_timestamp"]
    df["age"] = df["age"].clip(lower=1)   # avoid zeros or negatives

    # X = age only
    X = df[["age"]].values        # shape: (n,1)
    y = df["total_appreciates"].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    print("\n===== Regression #3: Popularity ~ Age (Time Bias) =====")
    print(f"Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept:          {model.intercept_:.4f}")
    print(f"R² Score:           {model.score(X, y):.4f}")
    print("========================================================\n")

    # Prepare regression line
    x_line = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_line = model.predict(x_line)

    # Scatter plot (downsample for speed)
    if len(df) > 5000:
        df_plot = df.sample(5000, random_state=42)
    else:
        df_plot = df

    # =====================================================
    # BLACK THEME VISUALIZATION (consistent with others)
    # =====================================================
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("black")    # full canvas background black

    ax = plt.gca()
    ax.set_facecolor("black")           # inside-plot background black

    # White gridlines
    ax.grid(color="white", alpha=0.25)

    # White axis spines
    for spine in ax.spines.values():
        spine.set_color("white")

    # White tick labels
    ax.tick_params(colors="white")

    # Scatter plot
    ax.scatter(
        df_plot["age"], 
        df_plot["total_appreciates"],
        alpha=0.25, s=6, color="#8FE3E9",
        label="Artworks (sampled)"
    )

    # Regression line (red)
    ax.plot(x_line, y_line, color="red", linewidth=3, label="Regression Line")

    # Labels + title (white)
    ax.set_xlabel("Artwork Age (Exposure Time)", color="white")
    ax.set_ylabel("Total Appreciates (Popularity)", color="white")
    ax.set_title("Regression #3: Effect of Age on Popularity (Time Bias)", color="white")

    # Legend
    legend = ax.legend()
    plt.setp(legend.get_texts(), color="white")
    legend.get_frame().set_edgecolor("white")

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================
    SAVE_DIR = fr"{dir_path}\visualisations"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save plot
    fig_path = os.path.join(SAVE_DIR, "regression3_age_plot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {fig_path}")

    # Save regression results
    csv_path = os.path.join(SAVE_DIR, "regression3_age_output.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "value"])
        writer.writerow(["intercept", model.intercept_])
        writer.writerow(["slope_age", model.coef_[0]])
        writer.writerow(["r2_score", model.score(X, y)])

    print(f"Saved regression output to: {csv_path}")

    plt.close()


if __name__ == "__main__":
    run()
