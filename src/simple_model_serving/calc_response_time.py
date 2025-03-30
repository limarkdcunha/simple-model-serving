import requests
import time

URL = "http://127.0.0.1:8000/predict"  # Change this if needed
NUM_REQUESTS = 100

# Sample input for prediction
payload = {"text": "This is a sample input for testing."}

response_times = []

for i in range(NUM_REQUESTS):
    start_time = time.perf_counter()  # Start time
    response = requests.post(URL, json=payload, timeout=10)
    end_time = time.perf_counter()  # End time

    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    response_times.append(elapsed_time)

# Compute Average Response Time
average_time = sum(response_times) / NUM_REQUESTS
print(f"\n🔹 Average Response Time: {average_time:.2f} ms")
