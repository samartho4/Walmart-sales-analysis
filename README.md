# Walmart Sales Forecasting

This project focuses on predicting Walmart store sales using historical data, with the goal of identifying patterns and insights that impact retail performance. The dataset includes store-level weekly sales data, along with features such as holidays, temperature, fuel prices, and unemployment rates.

## Objectives:
To predict future sales for Walmart stores.
To analyze factors influencing sales trends, including holiday markdowns, weather conditions, and economic indicators.
## Features of the Dataset:
Store: Store number.
Dept: Department number.
Date: Week of the sales.
Weekly_Sales: Sales for the given department in the given store.
IsHoliday: Boolean indicating if the week includes a major holiday.
Additional economic and weather-related features are provided.
## Analysis and Approach:
### Exploratory Data Analysis (EDA):
Identification of trends in sales.
Analysis of the impact of holidays and markdowns on sales.
Correlation analysis to uncover key relationships between features.
### Modeling and Forecasting:
Data preprocessing and feature engineering to handle missing values and encode categorical features.
Time-series analysis for understanding seasonality.
Implementation of machine learning models to predict sales.
### Evaluation Metrics:
Weighted Mean Absolute Error (WMAE) to account for the impact of holiday sales on prediction accuracy.
## Results:
The analysis provided valuable insights into sales patterns and drivers. Holiday periods were found to have a significant impact on sales, necessitating their accurate identification and handling in the predictive models.

### Tools and Libraries:
Python: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Kaggle environment for dataset exploration and modeling
