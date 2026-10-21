import requests
import time

# Batch of 10 values to send as input
data_batch = [{"x": x}
            for x in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]]

responses = []
start = time.time()
session = requests.Session()
for data in data_batch:
    response = session.post(
        "http://localhost:8000/predict",
        json=data
    )
    responses.append(response.json())

end = time.time()

for i, resp in enumerate(responses):
    print(f"Response {i + 1}: {resp}")

print(f"Total time taken for batch: {end - start:.4f} seconds")
