import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGE_PATH = Path("data/raw/diabetic_retinopathy/100_left.jpeg")

class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""
    
    wait_time = between(1, 2)

    @task
    def health(self):
        self.client.get("/health")

    @task
    def classify_image(self):
        with IMAGE_PATH.open("rb") as img:
            self.client.post(
                "/classify",
                files={
                    "file": (
                        IMAGE_PATH.name,
                        img,
                        "image/jpeg",
                    )
                },
            )