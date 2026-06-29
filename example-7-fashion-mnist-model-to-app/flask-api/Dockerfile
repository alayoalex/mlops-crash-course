# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy app files
COPY . .

# Copy model file
COPY model.onnx .

# Expose port 5000
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
