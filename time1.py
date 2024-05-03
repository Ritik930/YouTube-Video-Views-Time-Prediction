import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("E:\Table data.csv")

# Extract the actual upload time
actual_upload_time = data['Video publish time']
print("Actual Upload Time:", actual_upload_time)

# Extract features for predicting views
X_views = data[['Watch time (hours)', 'Subscribers', 'Impressions', 'Impressions click-through rate (%)']]
X_views['Video publish time'] = pd.to_datetime(data['Video publish time'])
X_views['Day of week'] = X_views['Video publish time'].dt.dayofweek
X_views['Hour of day'] = X_views['Video publish time'].dt.hour
X_views.drop(columns=['Video publish time'], inplace=True)

# Target variable for predicting views
y_views = data['Views']

# Split the data into training and testing sets for predicting views
X_train_views, X_test_views, y_train_views, y_test_views = train_test_split(X_views, y_views, test_size=0.2, random_state=42)

# Initialize SimpleImputer with strategy='mean' to impute missing values for predicting views
imputer_views = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both training and testing data for predicting views
X_train_views_imputed = imputer_views.fit_transform(X_train_views)
X_test_views_imputed = imputer_views.transform(X_test_views)

# Initialize the Random Forest regressor model for predicting views
model_views = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using imputed data for predicting views
model_views.fit(X_train_views_imputed, y_train_views)

# Make predictions on the testing set using the imputed data for predicting views
predictions_views_imputed = model_views.predict(X_test_views_imputed)

# Evaluate the model performance using the imputed data for predicting views
mae_views_imputed = mean_absolute_error(y_test_views, predictions_views_imputed)
print("Mean Absolute Error for views with imputed data:", mae_views_imputed)

# Example usage: Predict optimal upload time for a new video
new_video_features = [[500, 1000, 20000, 2.5, 3, 12]]  # Example features for Watch time, Subscribers, Impressions, CTR, Day of week, Hour of day
predicted_views = model_views.predict(new_video_features)
print("Predicted views for the new video:", predicted_views)

# Extract features for predicting the target publish time
X_time = data[['Watch time (hours)', 'Subscribers', 'Impressions', 'Impressions click-through rate (%)']]
X_time['Video publish time'] = pd.to_datetime(data['Video publish time'])
X_time['Day of week'] = X_time['Video publish time'].dt.dayofweek
X_time['Hour of day'] = X_time['Video publish time'].dt.hour
X_time.drop(columns=['Watch time (hours)', 'Subscribers', 'Impressions', 'Impressions click-through rate (%)', 'Video publish time'], inplace=True)

# Target variable for predicting the target publish time
y_time = data['Video publish time'].astype('int64') // 10**9  # Convert timestamp to Unix time

# Split the data into training and testing sets for predicting the target publish time
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

# Initialize SimpleImputer with strategy='mean' to impute missing values for predicting the target publish time
imputer_time = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both training and testing data for predicting the target publish time
X_train_time_imputed = imputer_time.fit_transform(X_train_time)
X_test_time_imputed = imputer_time.transform(X_test_time)

# Initialize the Random Forest regressor model for predicting the target publish time
model_time = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using imputed data for predicting the target publish time
model_time.fit(X_train_time_imputed, y_train_time)

# Make predictions on the testing set using the imputed data for predicting the target publish time
predictions_time_imputed = model_time.predict(X_test_time_imputed)

# Evaluate the model performance using the imputed data for predicting the target publish time
mae_time_imputed = mean_absolute_error(y_test_time, predictions_time_imputed)
print("Mean Absolute Error for target publish time with imputed data:", mae_time_imputed)

# Example usage: Predict the target publish time for a new video
new_video_features_time = [[500, 1000, 20000, 2.5, 3]]  # Example features for Watch time, Subscribers, Impressions, CTR, Day of week
predicted_publish_time = model_time.predict(new_video_features_time)
print("Predicted target publish time for the new video:", predicted_publish_time)
