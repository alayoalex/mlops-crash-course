from utils import transform_image
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("model.onnx")

def get_prediction(image_bytes):
    image_array = transform_image(image_bytes=image_bytes)
    outputs = ort_session.run(None, {'input': image_array})
    _, y_hat = outputs[0].max(), outputs[0].argmax()
    return y_hat
