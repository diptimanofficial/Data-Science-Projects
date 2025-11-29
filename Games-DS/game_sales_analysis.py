import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
from openpyxl import Workbook

plt.style.use("dark_background")

def set_axes_white(ax):
    """Makes axes, labels, and ticks white."""
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

dir_path = r"C:\Users\vikra\OneDrive\Desktop\Diptiman_DS\Games_DS"
os.chdir(dir_path)
print("CHANGED WORKING DIRECTORY TO:", os.getcwd())

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Step 1: Loading Data
df = pd.read_csv(f"{dir_path}\games_march2025_cleaned.csv", low_memory=False)

print("\n=== Dataset Loaded Successfully ===")
print("Shape: ", df.shape)
print("\nColumns: ")
print(df.columns.tolist())

print("\n=== First 5 Rows===")
print(df.head())

print("\n== Data Types===")
print(df.dtypes)

print("\n=== Numeric Summary Stats ===")
print(df.describe().T)

print("n=== Missing Values (%) ===")
print((df.isnull().mean() * 100).round(2))


# Step 2: Extracting Numeric Owners from String
owners = df["estimated_owners"].str.replace(",", "", regex=False)

split_owners = owners.str.split(" - ")

df["owners_low"] = pd.to_numeric(split_owners.str[0], errors="coerce")
df["owners_high"] = pd.to_numeric(split_owners.str[1], errors="coerce")

#finding mean owners 
df["owners_mid"] = (df["owners_low"] + df["owners_high"]) / 2

print("\n=== Owners Extracted===")
print(df[["estimated_owners", "owners_low", "owners_high", "owners_mid"]].head(10))


# Step 3: Building Regression Dataset
predictors = [
    "price",
    "required_age",
    "dlc_count",
    "metacritic_score",
    "achievements",
    "recommendations",
    "user_score",
    "positive",
    "negative",
    "average_playtime_forever",
    "average_playtime_2weeks",
    "median_playtime_forever",
    "median_playtime_2weeks",
    "discount",
    "peak_ccu",
    "pct_pos_total",
    "num_reviews_total",
    "pct_pos_recent",
    "num_reviews_recent"
]

# Removing score_rank (nearly all missing)
model_df = df[predictors + ["owners_mid"]].dropna()

print("\n=== Regression Dataset Ready===")
print("Shape: ", model_df.shape)
print("\nPreview: ")
print(model_df.head())


# Step 4: Running sales regression 
x = model_df[predictors]
y = model_df["owners_mid"]

# Add a constant
x = sm.add_constant(x)

# Fit Ordinary Least Squares model 
sales_model = sm.OLS(y, x).fit()

# Print full summary
print("\n=== Sales Regression Results ===")
print(sales_model.summary())

# Exporting regression results
coef_table = pd.DataFrame({
    "variable"  : sales_model.params.index,
    "coef"      : sales_model.params, 
    "std_err"   : sales_model.bse, 
    "t_stat"    : sales_model.tvalues, 
    "p_value"   : sales_model.pvalues.values, 
    "ci_lower"  : sales_model.conf_int()[0].values,
    "ci_upper"  : sales_model.conf_int()[1].values
})

coef_table.to_csv("steam_regression_coefficients.csv", index=False)
print("Exported: steam_regression_coefficients.csv")

#Step 5: Creating log variables
log_vars = [
    "price",
    "required_age",
    "dlc_count",
    "metacritic_score",
    "achievements",
    "recommendations",
    "user_score",
    "positive",
    "negative",
    "average_playtime_forever",
    "average_playtime_2weeks",
    "median_playtime_forever",
    "median_playtime_2weeks",
    "discount",
    "peak_ccu",
    "pct_pos_total",
    "num_reviews_total",
    "pct_pos_recent",
    "num_reviews_recent",
    "owners_mid"
]

df_log = df.copy()

# Create log columns safely
for col in log_vars:
    df_log["log_" + col] = np.log(df_log[col] + 1)

# Use stable predictors
clean_log_predictors = [
    "log_price",
    "log_required_age",
    "log_metacritic_score",
    "log_user_score",
    "log_positive",
    "log_negative",
    "log_average_playtime_forever",
    "log_average_playtime_2weeks",
    "log_peak_ccu"
]

df_log = df.copy()

# Generate ONLY the log columns we need
for col in ["price", "required_age", "metacritic_score", "user_score",
            "positive", "negative", "average_playtime_forever",
            "average_playtime_2weeks", "peak_ccu", "owners_mid"]:
    df_log["log_" + col] = np.log(df_log[col] + 1)

log_predictors = clean_log_predictors
log_df = df_log[["log_owners_mid"] + log_predictors].replace([np.inf, -np.inf], np.nan).dropna()

x_log = sm.add_constant(log_df[log_predictors])
y_log = log_df["log_owners_mid"]

log_model = sm.OLS(y_log, x_log).fit()
print(log_model.summary())

# Exporting to Excel 
log_coef_table = pd.DataFrame({
    "variable": log_model.params.index,
    "coef": log_model.params.values,
    "std_err": log_model.bse.values,
    "t_stat": log_model.tvalues.values,
    "p_value": log_model.pvalues.values,
    "ci_lower": log_model.conf_int()[0].values,
    "ci_upper": log_model.conf_int()[1].values
})

log_coef_table.to_csv("steam_log_regression_coefficients.csv", index=False)

print("Exported: steam_log_regression_coefficients.csv")


# Step 6: Making visualisations

# Chart Style
plt.style.use("dark_background")

def set_axes_white(ax):
    """Makes axes, ticks, and labels white."""
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


def plot_linear_relationship(df, x_col, y_col, save_path):
    """Black + Red color scheme for Linear Regression."""
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=df, x=x_col, y=y_col,
        color="red", alpha=0.35, ax=ax
    )

    sns.regplot(
        data=df, x=x_col, y=y_col,
        scatter=False,
        color="red",
        line_kws={'linewidth': 2},
        ax=ax
    )

    ax.set_title(f"Linear Regression: {x_col} vs {y_col}", fontsize=16)
    set_axes_white(ax)

    plt.savefig(os.path.join(charts_dir, save_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_log_relationship(df, x_col, y_col, save_path):
    """Black + Blue color scheme for Log-Log Regression."""
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=df, x=x_col, y=y_col,
        color="dodgerblue", alpha=0.35, ax=ax
    )

    sns.regplot(
        data=df, x=x_col, y=y_col,
        scatter=False,
        color="dodgerblue",
        line_kws={'linewidth': 2},
        ax=ax
    )

    ax.set_title(f"Log-Log Regression: {x_col} vs {y_col}", fontsize=16)
    set_axes_white(ax)

    plt.savefig(os.path.join(charts_dir, save_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Create charts directory
charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)
print(f"Charts will be saved in: {charts_dir}/")

#LINEAR REGRESSION PLOTS
plot_linear_relationship(df, "price", "owners_mid", "linear_price_vs_owners.png")
plot_linear_relationship(df, "positive", "owners_mid", "linear_positive_vs_owners.png")

# Predicted vs Actual (Linear)
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x=sales_model.predict(x), y=df["owners_mid"], color="red", alpha=0.35, ax=ax)
ax.set_title("Predicted vs Actual (Linear Model)")
ax.set_xlabel("Predicted Owners")
ax.set_ylabel("Actual Owners")
set_axes_white(ax)
plt.savefig(os.path.join(charts_dir, "linear_predicted_vs_actual.png"), dpi=300, bbox_inches="tight")
plt.close()


# LOG-LOG REGRESSION PLOTS
plot_log_relationship(log_df, "log_price", "log_owners_mid", "log_price_vs_log_owners.png")
plot_log_relationship(log_df, "log_positive", "log_owners_mid", "log_positive_vs_log_owners.png")

# Log Residuals
residuals_log = y_log - log_model.predict(x_log)

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x=log_model.predict(x_log), y=residuals_log,
                color="dodgerblue", alpha=0.35, ax=ax)
ax.axhline(0, color='red')
ax.set_title("Residuals (Log-Log Model)")
ax.set_xlabel("Predicted log(owners)")
ax.set_ylabel("Residuals")
set_axes_white(ax)
plt.savefig(os.path.join(charts_dir, "log_residuals.png"), dpi=300, bbox_inches="tight")
plt.close()

# Log Predicted vs Actual
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x=log_model.predict(x_log), y=y_log,
                color="dodgerblue", alpha=0.35, ax=ax)
sns.lineplot(x=y_log, y=y_log, color='red', ax=ax)
ax.set_title("Predicted vs Actual (Log-Log Model)")
ax.set_xlabel("Predicted log(owners)")
ax.set_ylabel("Actual log(owners)")
set_axes_white(ax)
plt.savefig(os.path.join(charts_dir, "log_predicted_vs_actual.png"), dpi=300, bbox_inches="tight")
plt.close()

# COMBINED PLOTS

# Combined Linear vs Log Price
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(x=df["price"], y=df["owners_mid"], color="red", alpha=0.3, ax=axes[0])
sns.regplot(x=df["price"], y=df["owners_mid"], scatter=False, color="red", ax=axes[0])
axes[0].set_title("Linear: Price vs Owners")
set_axes_white(axes[0])

sns.scatterplot(x=log_df["log_price"], y=log_df["log_owners_mid"], color="dodgerblue", alpha=0.3, ax=axes[1])
sns.regplot(x=log_df["log_price"], y=log_df["log_owners_mid"], scatter=False, color="dodgerblue", ax=axes[1])
axes[1].set_title("Log-Log: log(price) vs log(owners)")
set_axes_white(axes[1])

plt.savefig(os.path.join(charts_dir, "combined_price_linear_vs_log.png"), dpi=300, bbox_inches="tight")
plt.close()


# Combined Residuals
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(x=sales_model.predict(x), 
                y=df["owners_mid"] - sales_model.predict(x),
                color="red", alpha=0.3, ax=axes[0])
axes[0].axhline(0, color="white", linewidth=1)
axes[0].set_title("Residuals: Linear Model")
set_axes_white(axes[0])

sns.scatterplot(x=log_model.predict(x_log), 
                y=residuals_log,
                color="dodgerblue", alpha=0.3, ax=axes[1])
axes[1].axhline(0, color="white", linewidth=1)
axes[1].set_title("Residuals: Log-Log Model")
set_axes_white(axes[1])

plt.savefig(os.path.join(charts_dir, "combined_residuals_linear_vs_log.png"), dpi=300, bbox_inches="tight")
plt.close()

# Step 7: GTA 6 prediction model

# GTA 5 key stats
gta5_stats = pd.DataFrame({
    "metric": [
        "launch_price",
        "required_age",
        "metacritic_score",
        "user_score",
        "positive_reviews_month1",
        "negative_reviews_month1",
        "month1_sales"
    ],
    "value": [
        59.99,
        17,
        97,
        8.3,
        450000,
        20000,
        13000000
    ] 
})

# GTA 6 stats assumption
gta6_assumptions = pd.DataFrame({
    "metric": [
        "launch_price",
        "required_age",
        "metacritic_score",
        "user_score",
        "positive_reviews_month1",
        "negative_reviews_month1",
        "peak_ccu_month1"
    ],
    "value": [
        69.99,
        17,
        97,
        8.5,
        650000,
        30000,
        500000
    ]
})

gta6_features = pd.DataFrame({
    "price": [69.99],
    "required_age": [17],
    "metacritic_score": [97],
    "user_score": [8.5],
    "positive": [650000],
    "negative": [30000],
    "peak_ccu": [500000]
})

# For columns missing in model, filling with 0
for col in x.columns:
    if col not in gta6_features.columns:
        gta6_features[col] = 0

gta6_input = gta6_features[x.columns]

gta6_features = sm.add_constant(gta6_input)

# Predict Month-1 GTA 6 sales
gta6_pred_linear = sales_model.predict(gta6_input)[0]

gta6_forecast = pd.DataFrame({
    "Metric": ["Predicted Month-1 Sales (Linear Model)"],
    "Value": [gta6_pred_linear]
})

output_path = "gta6_sales_forecast_linear.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    gta5_stats.to_excel(writer, sheet_name="GTA5_Launch_Stats", index=False)
    gta6_assumptions.to_excel(writer, sheet_name="GTA6_Assumptions", index=False)
    gta6_forecast.to_excel(writer, sheet_name="GTA6_Linear_Forecast", index=False)

print(f"\nExported successfully: {output_path}")
print("\n=== GTA 6 Forecast (Linear Model) ===")
print(gta6_forecast)