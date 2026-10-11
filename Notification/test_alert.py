import os
import io
import requests
import imageio
import numpy as np

TOPIC = "incident_alert_shubh"  # Match the topic you created in the app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, "sample.gif")

# Create a sample animated GIF if none exists
if not os.path.exists(FILE_PATH):
    print(f"Creating test GIF at '{FILE_PATH}'...")
    test_frames = [
        np.zeros((120, 160, 3), dtype=np.uint8) + int(i * 25)
        for i in range(10)
    ]
    imageio.mimsave(FILE_PATH, test_frames, format="GIF", fps=10)

with open(FILE_PATH, "rb") as f:
    response = requests.put(
        f"https://ntfy.sh/{TOPIC}",
        data=f,
        headers={
            "Title": "🚨 Suspicious Activity Detected!".encode("utf-8"),
            "Message": "Movement detected in Camera Feed 1".encode("utf-8"),
            "Priority": "high",
            "Tags": "warning,rotating_light",
            "Filename": "incident.gif"
        }
    )

print("Status code:", response.status_code)
if response.status_code == 200:
    print("Alert sent successfully to ntfy.sh!")
else:
    print("Response text:", response.text)