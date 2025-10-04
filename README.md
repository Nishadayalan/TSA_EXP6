# Ex.No: 6               HOLT WINTERS METHOD
### Date: 04-10-2025



### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load dataset
data = pd.read_csv("/content/Airplane_Crashes_and_Fatalities_Since_1908_20190820105639.csv", parse_dates=['Date'], index_col='Date')
print("\nFirst 5 rows of dataset:\n", data.head())

# 2. Resample monthly (count crashes per month)
data_monthly = data.resample('MS').size().to_frame(name='Crashes')
print("\nMonthly resampled data (first 5 rows):\n", data_monthly.head())

# 3. Plot original data
data_monthly.plot(title="Monthly Plane Crashes")
plt.show()

# 4. Scaling data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data.plot(title="Scaled Data (0-1)")
plt.show()

# 5. Seasonal decomposition
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

# 6. Train-Test Split
scaled_data = scaled_data + 1   # multiplicative seasonality requires positive values
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# 7. Holt-Winters Model Training
model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions = model.forecast(steps=len(test_data))

# Plot prediction vs actual
ax = train_data.plot(label="Train")
test_predictions.plot(ax=ax, label="Predictions")
test_data.plot(ax=ax, label="Test")
ax.legend()
ax.set_title("Holt-Winters Forecast vs Actual (Plane Crashes)")
plt.show()

# 8. Model Performance Metrics
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
mae = mean_absolute_error(test_data, test_predictions)
print("\nModel Performance Metrics:")
print("RMSE:", rmse)
print("MAE :", mae)
print("Std Dev of scaled data:", scaled_data.std())
print("Mean of scaled data   :", scaled_data.mean())

# 9. Final Model (on full dataset) and Forecast
final_model = ExponentialSmoothing(
    scaled_data, trend='add', seasonal='mul', seasonal_periods=12
).fit()
final_predictions = final_model.forecast(steps=12)  # next 12 months

# Plot final forecast
ax = scaled_data.plot(label="Original")
final_predictions.plot(ax=ax, label="Forecast")
ax.legend()
ax.set_title("Final Forecast (Next Year - Plane Crashes)")
plt.show()
```



### OUTPUT:

<img width="830" height="612" alt="image" src="https://github.com/user-attachments/assets/ce0c356b-5f2b-4b3e-bed7-19c9f6aeb291" />

<img width="823" height="612" alt="image" src="https://github.com/user-attachments/assets/0236aea4-7fd7-40b4-a2fd-a8deac0955cd" />


TEST_PREDICTION:

<img width="892" height="651" alt="image" src="https://github.com/user-attachments/assets/261d51b5-7cf6-4cff-9af9-be22e4bae4cb" />

<img width="775" height="613" alt="image" src="https://github.com/user-attachments/assets/da8cc4fc-652f-4c85-8bb8-a79a7c201647" />



FINAL_PREDICTION:

<img width="815" height="608" alt="image" src="https://github.com/user-attachments/assets/04d687cb-2c4e-4c37-b6d2-7fa4f814c50e" />

<img width="728" height="155" alt="image" src="https://github.com/user-attachments/assets/fd9053b9-d896-450c-afd5-afca8867d5b6" />




### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
