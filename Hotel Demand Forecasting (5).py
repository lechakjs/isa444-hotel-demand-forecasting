#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-06T04:18:11.486Z
"""

# # Section 0: Import Data


# > import pandas as pd
# 
# df = pd.read_parquet("sample_hotels.parquet")
# df.head() 


!pip install statsforecast mlforecast xgboost neuralforecast utilsforecast lightgbm


# 1. Imports

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoARIMA

from mlforecast import MLForecast
from xgboost import XGBRegressor

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS   # your version uses these

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global settings
H = 28           # forecast horizon (days)
N_WINDOWS = 5    # 5-fold time-series CV

plt.rcParams["figure.figsize"] = (10, 5)


# # Section 1: Load and Inspect Data


import pandas as pd

df = pd.read_parquet("sample_hotels.parquet")
df.head()

df.dtypes

df.columns.tolist()


# # Section 2: Build the Time Series Data


ts_df = (
    df[["unique_id", "ds", "y"]]
    .sort_values(["unique_id", "ds"])
)

ts_df.head()


ts_df = (
    df[["unique_id", "ds", "y"]]
    .sort_values(["unique_id", "ds"])
)

ts_df.head()



!pip install statsforecast

# Now, import the necessary modules
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoARIMA

# Define models
sf_models = [
    Naive(),
    SeasonalNaive(season_length=7),
    AutoETS(),
    AutoARIMA()
]

# Create StatsForecast object (no df here)
sf = StatsForecast(
    models=sf_models,
    freq="D",
    n_jobs=-1
)

# 5-fold cross-validation (df passed here)
sf_cv = sf.cross_validation(
    df=ts_df,
    h=28,
    n_windows=5,
    step_size=28,
    refit=True,
    level=[]
)

sf_cv.head()

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape   # ← use these


sf_model_names = ["Naive", "SeasonalNaive", "AutoETS", "AutoARIMA"]

metrics_sf = evaluate(
    sf_cv,
    metrics=[mae, rmse, mape],
    models=sf_model_names,
    id_col="unique_id",
    time_col="ds",
    target_col="y",
)

metrics_sf


from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape

sf_model_names = ["Naive", "SeasonalNaive", "AutoETS", "AutoARIMA"]

metrics_sf = evaluate(
    sf_cv,
    metrics=[mae, rmse, mape],
    models=sf_model_names,
    id_col="unique_id",
    time_col="ds",
    target_col="y"
)


# Names of the model columns in sf_cv
sf_model_names = ["Naive", "SeasonalNaive", "AutoETS", "AutoARIMA"]

metrics_sf = evaluate(
    sf_cv,
    metrics=[mae, rmse, mape],     # use the custom metric functions
    models=sf_model_names,         # tell evaluate which columns are models
    id_col="unique_id",
    time_col="ds",
    target_col="y",
)

metrics_sf.head()


# # Step 3: Metrics


import numpy as np
import pandas as pd

def compute_metrics_group(group, pred_col):
    y_true = group["y"]
    y_pred = group[pred_col]
    
    me = (y_true - y_pred).mean()
    mae = (y_true - y_pred).abs().mean()
    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    mape = ((y_true - y_pred).abs() / y_true.replace(0, np.nan)).mean() * 100
    
    return pd.Series({
        "ME": me, 
        "MAE": mae, 
        "RMSE": rmse, 
        "MAPE": mape
    })


rows = []
for m in sf_models:
    mname = m.__class__.__name__
    pred_col = mname
    
    tmp = sf_cv[["unique_id", "ds", "y", pred_col]].dropna()
    
    metrics = tmp.groupby("unique_id").apply(
        lambda g: compute_metrics_group(g, pred_col)
    ).reset_index()
    
    metrics["model"] = mname
    rows.append(metrics)

sf_metrics_by_hotel = pd.concat(rows, ignore_index=True)
sf_metrics_by_hotel.head()


idx = sf_metrics_by_hotel.groupby("unique_id")["MAE"].idxmin()
sf_winners = sf_metrics_by_hotel.loc[idx, ["unique_id", "model"]]

sf_winners["model"].value_counts()


import os
os.makedirs("results", exist_ok=True)

sf_metrics_by_hotel.to_csv("results/sf_cv_metrics_by_hotel.csv", index=False)
sf_winners.to_csv("results/sf_classical_mae_winners.csv", index=False)


sf_metrics_by_hotel.head()

sf_winners["model"].value_counts()

!pip install mlforecast xgboost lightgbm


from mlforecast import MLForecast
from xgboost import XGBRegressor

mlf = MLForecast(
    models=[
        XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="reg:squarederror"
        )
    ],
    freq="D",
    lags=[1, 2, 7, 14]  # standard useful lags
)


# Cross-validation for XGBoost / MLForecast
mlf_cv = mlf.cross_validation(
    df=ts_df,
    h=28,
    n_windows=5,
    step_size=28,
)

mlf_cv.head()


ml_cv = mlf.cross_validation(
    df=ts_df,
    n_windows=5,
    h=28,
    step_size=28
)

ml_cv.head()


pred_col = "XGBRegressor"

tmp = ml_cv[["unique_id", "ds", "y", pred_col]].dropna()

ml_metrics_by_hotel = tmp.groupby("unique_id").apply(
    lambda g: compute_metrics_group(g, pred_col)
).reset_index()

ml_metrics_by_hotel["model"] = pred_col

ml_metrics_by_hotel.head()


ml_metrics_by_hotel.to_csv("results/ml_cv_metrics_by_hotel.csv", index=False)


ml_cv.head()

ml_metrics_by_hotel.head()

# # Step 4: Classical Models (StatsForecast)


!pip install statsforecast


# Step 4 — Classical Models (StatsForecast)

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoARIMA

# Define classical models
sf_models = [
    Naive(),
    SeasonalNaive(season_length=7),  # Weekly seasonality
    AutoETS(),
    AutoARIMA()
]

# Create StatsForecast object
sf = StatsForecast(
    models=sf_models,
    freq='D',
    n_jobs=-1
)

# Run 5-fold cross-validation (h = 28-day forecast, step = 28)
sf_cv = sf.cross_validation(
    df=ts_df,
    n_windows=5,
    h=28,
    step_size=28
)

sf_cv.head()


import numpy as np
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape  # 👈 built-in metric functions

# List of model columns in sf_cv
sf_models = ["Naive", "SeasonalNaive", "AutoETS", "AutoARIMA"]

metrics_sf = evaluate(
    sf_cv,
    metrics=[mae, rmse, mape],   # 👈 pass the FUNCTIONS, not strings
    models=sf_models,            # 👈 tell it which columns are model forecasts
    id_col="unique_id",
    time_col="ds",
    target_col="y",
)

metrics_sf


from utilsforecast.losses import mae, rmse, mape
metrics=[mae, rmse, mape]
models=["Naive","SeasonalNaive","AutoETS","AutoARIMA"]


# # Step 5: Machine Learning Models (MLForecast – XGBoost)


ts_df.head()



!pip install mlforecast xgboost lightgbm


from mlforecast import MLForecast
from xgboost import XGBRegressor
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape
import pandas as pd
import numpy as np


ml_models = {
    'xgboost': XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0
    )
}

mlf = MLForecast(
    models=ml_models,        # 👈 gives the column name 'xgboost'
    freq='D',
    lags=[1, 2, 7, 14, 28],  # past values
    date_features=['dayofweek', 'week', 'month']  # calendar effects
)


mlf_cv = mlf.cross_validation(
    df=ts_df,
    n_windows=5,
    h=28,
    step_size=28,
)

mlf_cv.head()


ml_model_names = ['xgboost']   # matches the column in mlf_cv

metrics_ml = evaluate(
    mlf_cv,
    metrics=[mae, rmse, mape],
    models=ml_model_names,
    id_col='unique_id',
    time_col='ds',
    target_col='y'
)

metrics_ml


def mean_error(y, y_hat):
    return np.mean(y_hat - y)

rows = []
y_true = mlf_cv['y'].values

for model in ml_model_names:
    y_hat = mlf_cv[model].values
    rows.append({
        'metric': 'ME',
        'model': model,
        'value': mean_error(y_true, y_hat)
    })

me_ml = pd.DataFrame(rows)
me_ml


# # Step 6: Deep Learning Models (NeuralForecast)


!pip install neuralforecast


from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape

import numpy as np
import pandas as pd


nf_models = [
    NBEATS(h=28, input_size=196),
    NHITS(h=28, input_size=196)
]


nf = NeuralForecast(
    models=nf_models,
    freq='D'
)

nf_cv = nf.cross_validation(
    df=ts_df,
    n_windows=5,
    h=28,
    step_size=28
)

nf_cv.head()


deep_models = ["NBEATS", "NHITS"]

metrics_deep = evaluate(
    nf_cv,
    metrics=[mae, rmse, mape],
    models=deep_models,
    id_col="unique_id",
    time_col="ds",
    target_col="y"
)

metrics_deep


def mean_error(y, y_hat):
    return np.mean(y_hat - y)

rows = []
y_true = nf_cv["y"].values

for m in deep_models:
    y_hat = nf_cv[m].values
    rows.append({
        "metric": "ME",
        "model": m,
        "value": mean_error(y_true, y_hat)
    })

me_deep = pd.DataFrame(rows)
me_deep


# 


sf_models      = ["Naive", "SeasonalNaive", "AutoETS", "AutoARIMA"]
ml_models      = ["xgboost"]          # whatever your MLForecast column is called
deep_models    = ["NBEATS", "NHITS"]
all_models     = sf_models + ml_models + deep_models

sf_cols   = ["unique_id", "ds", "cutoff", "y"] + sf_models
ml_cols   = ["unique_id", "ds", "cutoff"] + ml_models
deep_cols = ["unique_id", "ds", "cutoff"] + deep_models

cv_all = (
    sf_cv[sf_cols]
    .merge(mlf_cv[ml_cols],  on=["unique_id", "ds", "cutoff"], how="left")
    .merge(nf_cv[deep_cols], on=["unique_id", "ds", "cutoff"], how="left")
)

cv_all.head()

import numpy as np
import pandas as pd

def compute_metrics_overall(cv_df, model_names):
    rows = []
    y = cv_df["y"].values

    for m in model_names:
        y_hat = cv_df[m].values
        error = y_hat - y

        me   = np.mean(error)
        mae_ = np.mean(np.abs(error))
        rmse_ = np.sqrt(np.mean(error**2))
        mape_ = np.mean(np.abs(error / y)) * 100  # assumes y has no zeros

        rows.append({
            "model": m,
            "ME":   me,
            "MAE":  mae_,
            "RMSE": rmse_,
            "MAPE": mape_
        })

    return pd.DataFrame(rows)


metrics_all = compute_metrics_overall(cv_all, all_models)
metrics_all.sort_values("MAE")


def mae_by_series(cv_df, model_names):
    rows = []
    for uid, grp in cv_df.groupby("unique_id"):
        y = grp["y"].values
        for m in model_names:
            y_hat = grp[m].values
            mae_ = np.mean(np.abs(y_hat - y))
            rows.append({"unique_id": uid, "model": m, "MAE": mae_})
    return pd.DataFrame(rows)

mae_series = mae_by_series(cv_all, all_models)
mae_series.head()


winners_mae = (
    mae_series
    .sort_values(["unique_id", "MAE"])
    .groupby("unique_id")
    .first()
    .reset_index()
)

winner_counts_mae = winners_mae["model"].value_counts()
winner_counts_mae


cv_all.to_csv("results/cv_all_models.csv", index=False)
metrics_all.to_csv("results/metrics_all_models.csv", index=False)
mae_series.to_csv("results/mae_by_series.csv", index=False)
winners_mae.to_csv("results/mae_winners_by_series.csv", index=False)
winner_counts_mae.to_frame("count").to_csv("results/mae_winner_counts.csv")


import matplotlib.pyplot as plt

def plot_forecasts_for_hotel(hotel_id, models_to_plot=None):
    # If no specific models listed, plot all
    if models_to_plot is None:
        models_to_plot = all_models

    # Filter the combined CV dataframe
    sub = cv_all[cv_all["unique_id"] == hotel_id].sort_values("ds")

    plt.figure(figsize=(12, 5))
    
    # Plot actuals
    plt.plot(sub["ds"], sub["y"], label="Actual", linewidth=3, color="black")

    # Plot each model’s forecasts
    for m in models_to_plot:
        if m in sub.columns:
            plt.plot(sub["ds"], sub[m], label=m, alpha=0.7)

    plt.title(f"Forecast vs Actuals for {hotel_id}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_forecasts_for_hotel("hotel_0")
plot_forecasts_for_hotel("hotel_77")


# # Highlights Top Models


plot_forecasts_for_hotel("hotel_0", models_to_plot=["xgboost", "AutoARIMA", "NHITS"])


# 


!pip install python-dotenv


from dotenv import load_dotenv
load_dotenv()

import os
key = os.getenv("NIXTLA_API_KEY")
print("Loaded:", key[:8] + "*****")

!pip install nixtla


from nixtla import NixtlaClient

nixtla_client = NixtlaClient(api_key=key)

import pandas as pd

# Load the hotel dataset
df = pd.read_parquet("sample_hotels.parquet")
df.head()



ts_complete = (
    ts_df
    .sort_values(["unique_id", "ds"])
    .set_index("ds")                                  # use ds as index
    .groupby("unique_id", group_keys=False)           # group by hotel
    .apply(lambda g: g.asfreq("D"))                   # daily frequency
    .reset_index()                                    # back to columns: unique_id, ds, y
)

ts_complete.head()


# Make 100% sure there are NO NaNs in y for any hotel
ts_complete["y"] = (
    ts_complete
    .groupby("unique_id")["y"]
    .transform(
        lambda s: (
            s.interpolate(limit_direction="both")  # smooth internal gaps
             .ffill()                              # fill any remaining with last value
             .bfill()                              # and next value if at the start
        )
    )
)

# As a last resort (in case an entire series was NaN), fill any leftover with 0
ts_complete["y"] = ts_complete["y"].fillna(0)

# Sanity check
print("NaNs remaining in y:", ts_complete["y"].isna().sum())


from nixtla import NixtlaClient
nixtla_client = NixtlaClient(api_key=key)


# Ensure y is numeric
ts_complete["y"] = pd.to_numeric(ts_complete["y"], errors="coerce")

# (re-fill just in case converting to numeric produced new NaNs)
ts_complete["y"] = (
    ts_complete
    .groupby("unique_id")["y"]
    .transform(
        lambda s: (
            s.interpolate(limit_direction="both")
             .ffill()
             .bfill()
        )
    )
)
ts_complete["y"] = ts_complete["y"].fillna(0)

print("NaNs in y after numeric conversion:", ts_complete["y"].isna().sum())


import pandas as pd
import numpy as np

# Keep only the three columns we need
tgpt_df = ts_complete[["unique_id", "ds", "y"]].copy()

# 1. Force ds to proper datetime
tgpt_df["ds"] = pd.to_datetime(tgpt_df["ds"], errors="coerce")

# 2. Force y to numeric
tgpt_df["y"] = pd.to_numeric(tgpt_df["y"], errors="coerce")

# 3. Drop rows where ds or y is completely unusable
tgpt_df = tgpt_df.dropna(subset=["ds", "y"])

print(tgpt_df.dtypes)
print("Any NaNs in y?", tgpt_df["y"].isna().sum())
print("Any NaNs in ds?", tgpt_df["ds"].isna().sum())


# Look for rows where y was originally non-numeric text
mask_weird_y = ts_complete["y"].apply(lambda v: isinstance(v, str))
print("Rows where original y is a string:", mask_weird_y.sum())
ts_complete.loc[mask_weird_y, ["unique_id", "ds", "y"]].head()


# Inspect the types inside unique_id
tgpt_df['uid_type'] = tgpt_df['unique_id'].map(type)
print(tgpt_df['uid_type'].value_counts())
tgpt_df[['unique_id', 'uid_type']].head()



if 'uid_type' in tgpt_df.columns:
    tgpt_df = tgpt_df.drop(columns=['uid_type'])

# Force unique_id to string
tgpt_df['unique_id'] = tgpt_df['unique_id'].astype(str)

# Double-check
print(tgpt_df.dtypes)
print(tgpt_df['unique_id'].map(type).value_counts())


import pandas as pd
import numpy as np

# Start from the daily-complete dataframe (the one you called ts_complete)
tgpt_df = ts_complete[["unique_id", "ds", "y"]].copy()

# 1. Force correct types
tgpt_df["unique_id"] = tgpt_df["unique_id"].astype(str)
tgpt_df["ds"]        = pd.to_datetime(tgpt_df["ds"])
tgpt_df["y"]         = pd.to_numeric(tgpt_df["y"], errors="coerce")

# 2. Remove any accidental duplicate id-date rows
dupes = tgpt_df.duplicated(subset=["unique_id", "ds"])
print("Duplicate id-date rows:", dupes.sum())
tgpt_df = tgpt_df[~dupes]

# 3. Fill missing y for each hotel (interpolate, then forward/back fill)
tgpt_df = tgpt_df.sort_values(["unique_id", "ds"])

tgpt_df["y"] = (
    tgpt_df
    .groupby("unique_id")["y"]
    .transform(lambda s: s.interpolate(limit_direction="both").ffill().bfill())
)


# 4. Final sanity checks
print(tgpt_df.dtypes)
print("Any NaNs in y?", tgpt_df["y"].isna().sum())


H = 28
N_WINDOWS = 5

min_points = N_WINDOWS * H + 1   # 141 points per hotel

counts = tgpt_full.groupby("unique_id").size()
valid_ids = counts[counts >= min_points].index

print("Total hotels:", counts.shape[0])
print("Hotels with enough history:", len(valid_ids))

tgpt_long = (
    tgpt_full[tgpt_full["unique_id"].isin(valid_ids)]
    .sort_values(["unique_id", "ds"])
    .reset_index(drop=True)
)


H = 28           # 4 weeks ahead
N_WINDOWS = 5    # 5-fold time-series CV

tgpt_cv = nixtla_client.cross_validation(
    df=tgpt_long,      # <-- use the cleaned daily data
    h=H,
    freq="D",
    n_windows=N_WINDOWS,
)

tgpt_cv.head()


import numpy as np
import pandas as pd

def me(y, yhat):
    return np.mean(y - yhat)

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def mape(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    mask = y != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y[mask] - yhat[mask]) / y[mask]))

# overall TimeGPT metrics across all hotels & CV windows
tgpt_metrics_overall = pd.Series({
    "ME":   me(tgpt_cv["y"], tgpt_cv["TimeGPT"]),
    "MAE":  mae(tgpt_cv["y"], tgpt_cv["TimeGPT"]),
    "RMSE": rmse(tgpt_cv["y"], tgpt_cv["TimeGPT"]),
    "MAPE": mape(tgpt_cv["y"], tgpt_cv["TimeGPT"]),
})

tgpt_metrics_overall


# long-format row for overall TimeGPT metrics
tgpt_long = (
    tgpt_metrics_overall
    .reset_index()
    .rename(columns={"index": "metric", 0: "value"})
)
tgpt_long["model"] = "TimeGPT"





tgpt_long


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=088240b9-4bb0-48b0-97d5-633dac7933c1' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>