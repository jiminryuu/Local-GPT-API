from locust import HttpUser, task, between
import random

class TextGenerationUser(HttpUser):
    # Set the wait time between requests (in seconds)
    wait_time = between(1, 3)

    # Define the endpoint URL
    @task
    def generate_text(self):
        # Prepare the payload for the text generation request
        prompt = "Once upon a time, in a faraway kingdom..."
        max_length = random.randint(50, 100)
        temperature = random.uniform(0.7, 1.0)
        
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature
        }
        
        # Make the POST request to the /generate endpoint
        self.client.post("/generate", json=payload)