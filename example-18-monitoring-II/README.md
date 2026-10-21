# Machine Learning Prediction Service

A simple machine learning prediction service using Python and FastAPI. This project demonstrates how to serve a pre-trained machine learning model as a REST API service with logging and monitoring capabilities.

## Project Overview

This project contains:

- **Machine Learning Model**: A pre-trained linear regression model (y = 2x) as `model.pkl`
- **API Service**: FastAPI-based REST API with prediction and health check endpoints
- **Logging System**: Request logging with latency tracking to `logs/model_service.log`
- **Test Client**: Simple batch prediction test script

## Prerequisites

- Python 3.12
- uv (recommended for dependency management)

## Setup and Usage

### 1. Install Dependencies

Using uv (recommended):

```bash
uv sync
```

This will create a virtual environment and install all required dependencies

### 2. Activate the Virtual Environment

```bash
source .venv/bin/activate
# .venv\Scripts\activate (on Windows)
```

### 3. Start the API Server

```bash
uvicorn app:app --port 8000
```

The server will start on `http://localhost:8000` with the following endpoints:

- `GET /health` - Health check endpoint
- `POST /predict` - Prediction endpoint (accepts JSON with `x` value)

### 4. Test the API

In another terminal, run the test client:

```bash
python test.py
```

This will send a batch of 10 test inputs (1.5, 2.5, ..., 10.5) to the prediction endpoint and display:

- Individual predictions for each input
- Total time taken for the batch

## Logging

The application logs all requests and predictions to:

- Console (stdout)
- `logs/model_service.log` file
