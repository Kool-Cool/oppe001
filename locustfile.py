from locust import HttpUser, task, between
import random

class FraudApiUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        payload = {
            "transaction_id": random.randint(1, 1000),
            "Time": random.random(),
            "V1": random.random(),
            "V2": random.random(),
            "V3": random.random(),
            "V4": random.random(),
            "V5": random.random(),
            "V6": random.random(),
            "V7": random.random(),
            "V8": random.random(),
            "V9": random.random(),
            "V10": random.random(),
            "V11": random.random(),
            "V12": random.random(),
            "V13": random.random(),
            "V14": random.random(),
            "V15": random.random(),
            "V16": random.random(),
            "V17": random.random(),
            "V18": random.random(),
            "V19": random.random(),
            "V20": random.random(),
            "V21": random.random(),
            "V22": random.random(),
            "V23": random.random(),
            "V24": random.random(),
            "V25": random.random(),
            "V26": random.random(),
            "V27": random.random(),
            "V28": random.random(),
            "Amount": random.uniform(1, 1000)
        }
        self.client.post("/predict", json=payload)
