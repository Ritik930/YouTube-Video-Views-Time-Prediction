# YouTube-Video-Views-Time-Prediction

This project aims to predict the number of views for YouTube videos based on various features such as watch time, subscribers, impressions, and more. Additionally, it captures the time of prediction to provide users with more context about when the prediction was made.

## Dataset

The dataset `Table data.csv` contains the following columns:

- **Content**: Description of the video content.
- **Video title**: Title of the YouTube video.
- **Video publish time**: Time when the video was published.
- **Views**: Number of views for the video.
- **Watch time (hours)**: Total watch time of the video in hours.
- **Subscribers**: Number of subscribers at the time of video publication.
- **Impressions**: Number of impressions the video received.
- **Impressions click-through rate (%)**: Click-through rate (CTR) of the impressions.

## Usage

1. **Data Preparation**: Ensure you have the dataset `Table data.csv` with the required features.
2. **Model Training and Prediction**: Run the script `predict_views.py` to train the prediction model and make predictions for new videos.
3. **Input New Video Features**: Follow the instructions in the script to input the features of the new video for which you want to predict the views.
4. **View Prediction Results**: The script will output the predicted number of views and the time of prediction, providing both the prediction and the time it was made.

## Example

Here's an example of how to use the prediction model:

```python
# Load the model and make predictions
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Assuming the video features are stored in a list named `new_video_features`
model = RandomForestRegressor()
predicted_views = model.predict(new_video_features)

# Capture the current time as the prediction time
prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Display prediction results
print("Predicted views for the new video:", predicted_views)
print("Prediction time:", prediction_time)

