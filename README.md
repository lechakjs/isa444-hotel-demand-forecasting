# ISA 444: Hotel Demand Forecasting Project

This repository contains my full forecasting project for ISA 444.  
I completed **Option 1: Hotel Demand Forecasting** using the `sample_hotels.parquet` dataset.

---

## Project Objectives
- Forecast daily hotel demand for 17 properties  
- Forecast horizon: **28 days (h = 28)**  
- Perform **5-fold non-overlapping time-series cross-validation**  
- Compare:
  - Naive + Seasonal Naive  
  - AutoETS & AutoARIMA (StatsForecast)  
  - XGBoost (MLForecast)  
  - AutoNBEATS & AutoNHITS (NeuralForecast)  
  - TimeGPT (Nixtla)  
- Compute ME, MAE, RMSE, MAPE  
- Count model “wins” (lowest error per series)  
- Produce forecast vs actual plots  
- Export all evaluation results to CSVs  

---

## Models Implemented

### Classical Forecasting (StatsForecast)
- Naive  
- Seasonal Naive (7-day seasonality)  
- AutoETS  
- AutoARIMA  

### Machine Learning (MLForecast)
- XGBoostRegressor  
- Lag features, rolling statistics, calendar features  

### Deep Learning (NeuralForecast)
- AutoNBEATS  
- AutoNHITS  

### TimeGPT
- Global pretrained model  
- Applied with 5-fold cross-validation and evaluation metrics  

---

## Repository Contents

### Notebook
- `Hotel Demand Forecasting.ipynb`  
  Contains the full analysis, modeling, cross-validation, and plots.

### Results (CSV Outputs)
- `cv_all_models.csv`  
- `metrics_all_models.csv`  
- `mae_by_series.csv`  
- `mae_winner_counts.csv`  
- `mae_winners_by_series.csv`  
- `ml_cv_metrics_by_hotel.csv`  
- `sf_classical_mae_winners.csv`  
- `sf_cv_metrics_by_hotel.csv`  

These files are created programmatically during the notebook run.

---


