Domain: Energy & Utilities
Tools Used: Python · Scikit-learn · XGBoost · Matplotlib · Seaborn · Pandas · NumPy

Problem Statement
In today's energy-driven world, managing household energy efficiently is essential for both consumers and energy providers. This project focuses on building a predictive machine learning model that accurately forecasts household energy consumption using historical data.
Accurate predictions can help:

Reduce costs and optimize energy usage for households.

Improve demand forecasting for energy providers.

Detect anomalies like faults or unexpected usage.

Enable smart grid integration and promote sustainability.

Approach
1. Data Understanding & Exploration
Loaded and analyzed time-series household energy consumption data.

Conducted EDA to identify usage trends, seasonal patterns, and outliers.

2. Data Preprocessing
Handled missing values and ensured data consistency.

Extracted datetime features (hour, day, weekday, month).

Engineered features such as:

Daily average consumption

Peak hour indicators

Rolling averages (3h, 6h, 1d)

Applied feature scaling for model performance.

3. Feature Engineering
Selected and transformed key features influencing power usage.

Designed temporal and contextual features for better prediction accuracy.

4. Modeling
Trained and evaluated multiple regression models:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

K-Nearest Neighbors Regressor

Neural Network (MLP Regressor)

5. Model Evaluation
Models were evaluated using:

R² Score

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

Best Model: Neural Network (MLP Regressor)

