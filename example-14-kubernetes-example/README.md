# Machine Learning Prediction Service

A simple machine learning prediction service using Python and FastAPI. This project demonstrates how to build and deploy a machine learning model as a REST API service within a containerized environment orchestrated by Kubernetes.

## Project Overview

This project contains:

- **Machine Learning Model**: A simple linear regression model (y = 2x) for predictions
- **API Service**: FastAPI-based REST API with prediction and health check endpoints
- **Docker Support**: Complete containerization setup
- **Kubernetes Deployment**: YAML configurations for container orchestration
- A Simple test client for API validation

## Prerequisites

- Docker (for containerization)
- kind (Kubernetes in Docker)
- `kubectl` (for Kubernetes management)

## Step-by-Step Setup and Usage

### 1. Install Dependencies

Using uv (recommended):

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
# .venv\Scripts\activate (on Windows)
```

### 2. Train the Machine Learning Model

```bash
python train.py
```

This will:

- Generate sample training data (X: [1,2,3,4,5], y: [2,4,6,8,10])
- Train a simple linear regression model (y = 2x)
- Save the model to `model.pkl`
- Display model coefficients and intercept

### 3. Test the API Locally (Optional)

Start the FastAPI server locally:

```bash
uvicorn app:app --port 8000
```

In another terminal, test the API:

```bash
python test.py
```

### 4. Build Docker Image

Make sure you have Docker installed and running. Then build the Docker image:

```bash
docker build -t k-demo .
```

Run and test the Docker container locally (optional).

### 5. Deploy to Kubernetes

1. Load the Docker image into the kind cluster:

```bash
kind load docker-image k-demo --name kind
```

Or push the image to Docker Hub and pull it from there.

#### 6. Deploy the Application

1. Apply the Kubernetes configurations:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

2. Verify the deployment:

```bash
kubectl get pods
kubectl get services
```

3. Check pod logs:

```bash
kubectl logs -l app=k-demo
```

### 6. Access the Service

#### Port Forwarding (for testing)

```bash
kubectl port-forward service/k-demo 8000:80
```

Then access the API through `test.py`.
