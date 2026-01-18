# CPI_MLOPS_PROJECT

**CPI_MLOPS_PROJECT** is a machine learning project that predicts macroeconomic trends using a trained LSTM model. The project demonstrates end-to-end MLOps workflow: from training a model to building a REST API for real-time predictions, making it ready for deployment.

---

##  Project Overview

This project predicts the following economic indicators based on past observations:

- **Trend Prediction:** Predicts whether the target variable is rising, stable, or falling.
- **Acceleration Prediction:** Estimates short-term changes in trends.
- **Regime Prediction:** Classifies the economic regime based on historical patterns.

The LSTM model captures temporal dependencies in the data, and the API exposes endpoints for easy integration with front-end applications or other services.

---

## Features

- Pre-trained **LSTM model** for real-time predictions
- REST API built using **FastAPI**
- Fully containerizable and ready for cloud deployment (Render, Heroku, etc.)
- Supports JSON input for predictions
- Modular folder structure for maintainability

---

## LSTM Model & PyTorch Highlights

This project demonstrates **practical deep learning skills**:

- **Model Architecture:**
  - **LSTM (Long Short-Term Memory)** to capture temporal dependencies in economic time series.
  - **Input Size:** Number of features per time step.
  - **Hidden Size:** Dimension of the internal memory vector (hidden state), controlling memory capacity.
  - **Output Layer (Fully Connected):** Maps hidden states to trend, acceleration, and regime predictions.

- **State Management:**
  - Saved model using `torch.save(model.state_dict(), "lstm_model_v1.pt")`
  - Architecture is defined in code; the **state dict contains learned weights and biases** for deployment.

- **Advanced PyTorch Techniques:**
  - Handling sequence input for time series forecasting.
  - Managing **hidden and cell states** in recurrent models.
  - Forward pass with FC layers to generate predictions.
  - Loading state dicts correctly for production inference without retraining.

- **ML/DL Skills Showcased:**
  - Time series prediction using deep learning.
  - Understanding of **internal memory (hidden and cell states)** in LSTMs.
  - Model serialization for deployment in real-world pipelines.
  - Integration of classical ML concepts with deep learning.




