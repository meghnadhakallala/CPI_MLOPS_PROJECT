# api.py

from fastapi import FastAPI
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()

# Use CPU for deployment
device = torch.device("cpu")

# ----------------------------
# LSTM Model Definition
# This must match training
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


# ----------------------------
# Load Trained Model
# ----------------------------
model = LSTMModel().to(device)
model.load_state_dict(torch.load("data/model/lstm_model_v1.pt", map_location=device))
model.eval()


# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict(values: list[float]):
    """
    Input: sequence of CPI values
    Example: [210.5, 211.2, 212.1, 213.0, ...]
    """

    arr = np.array(values, dtype=np.float32)
    arr = arr.reshape(1, len(values), 1)  # (batch, seq_len, features)

    x = torch.tensor(arr).to(device)

    with torch.no_grad():
        preds = model(x).numpy().tolist()[0]

    return {
        "trend_prediction": preds[0],
        "acceleration_prediction": preds[1],
        "regime_prediction": preds[2]
    }


# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {"message": "LSTM CPI Predictor API is running"}
