Household Energy Usage Forecast


Domain: Energy & Utilities

Tools Used: Python · Scikit-learn · lightgbm · Matplotlib · Seaborn · Pandas · NumPy

Problem Statement
In today's energy-driven world, managing household energy efficiently is essential for both consumers and energy providers. This project focuses on building a predictive machine learning model that accurately forecasts household energy consumption using historical data.
Accurate predictions can help:

Reduce costs and optimize energy usage for households.

Improve demand forecasting for energy providers.

Detect anomalies like faults or unexpected usage.

Enable smart grid integration and promote sustainability.

Approach
A complete **end-to-end machine learning project** that predicts **Global Active Power** usage from household power consumption data.  
This project includes **data preprocessing**, **model training & optimization**, and an **interactive Streamlit web app** for real-time prediction.

---
## Project Overview

This project predicts `Global_active_power` using a variety of electrical features such as voltage, reactive power, and sub-metering data.
Trained with 6 Regressor Algorithm and got the best algorithm for this dataset.
It uses a **Random Forest Regressor** trained on preprocessed data, with scaling and feature engineering.

---

## 🧩 Key Components

1. **Data Preprocessing (`preprocessing.ipynb`)**
   - Handles missing values
   - Applies `np.log1p()` to skewed sub-metering columns
   - Scales numeric features using `StandardScaler`
   - Saves the fitted `scaler.pkl` for later use

2. **Model Training (`model_training.ipynb`)**
   - Trains **RandomForestRegressor**
   - Uses **GridSearchCV** for hyperparameter optimization
   - Evaluates model performance using R², RMSE, and MAE
   - Saves best model as `random_forest_model.pkl`

3. **Deployment (Streamlit App – `deployment.py`)**
   - Interactive prediction UI
   - Supports two input modes:
     -**Generate Random values**
     -**User Input**
   - Automatically computes `Is_peak_hour` and `Is_daytime` from `Hour`
   - Displays raw → log → scaled transformation and prediction
   - Shows feature importances




