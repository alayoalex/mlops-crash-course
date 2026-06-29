import numpy as np
from PIL import Image
import io

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0
    # img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(np.expand_dims(img_array, axis=0), axis=0)
    return img_array


def open_image(image_path="../data/examples/Sandal_0.png"):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return image_bytes
