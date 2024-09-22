import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (ensure the correct path)
model_path = '../data/sales_prediction_deeplearning_model_20240922_122652.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# Load the scalers (you need to save these scalers during your training phase)
scaler_X = MinMaxScaler(feature_range=(-1, 1))  # Load your trained scaler for input data
scaler_y = MinMaxScaler(feature_range=(-1, 1))  # Load your trained scaler for output data

# Define input data schema using Pydantic
class PredictionInput(BaseModel):
    features: list[float]

# API root endpoint (health check)
@app.get("/")
def read_root():
    return {"message": "ML Model Serving API is up and running"}

# Endpoint to make multi-step predictions (e.g., 42 days or 6 weeks ahead)
@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        # Function to predict future sales using the LSTM model
        def predict_future_sales(model, last_n_days, scaler_y, forecast_horizon=42):
            future_sales = []

            current_input = last_n_days  # Start with the last 7 days

            for _ in range(forecast_horizon):
                # Predict the next day's sales (scaled)
                next_pred_scaled = model.predict(current_input)

                # Ensure that next_pred_scaled has the correct shape for inverse_transform (should be 2D)
                next_pred_scaled = next_pred_scaled.reshape(-1, 1)

                # Invert scaling to get the actual sales
                next_pred = scaler_y.inverse_transform(next_pred_scaled)

                # Append the predicted sales
                future_sales.append(next_pred[0][0])

                # Update the input: drop the first day and add the new predicted day
                # The model expects input shape (1, lag, 1), so we adjust accordingly
                next_pred_scaled = next_pred_scaled.reshape(1, 1, 1)
                current_input = np.append(current_input[:, 1:, :], next_pred_scaled, axis=1)

            return np.array(future_sales)  # Return the future predictions

        # Main function to run the prediction process
        def main(model, scaler_y, input_data, forecast_horizon=42):
            # Ensure the input data is a NumPy array
            input_data = np.array(input_data)

            # Reshape the input data to match the model's input expectations
            input_data = input_data.reshape(1, len(input_data), 1)

            # Predict the future sales iteratively
            predicted_sales = predict_future_sales(model, input_data, scaler_y, forecast_horizon=forecast_horizon)

            return predicted_sales

        # Example usage:
        predicted_sales = main(model, scaler_y, input_data.features)  # Use the input data from the request

        return {"predictions": predicted_sales}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error occurred: {str(e)}")