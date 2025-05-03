import cv2
import requests
from parking_management import ParkingManagement
import parking_management
from datetime import datetime

print(dir(parking_management))

# API endpoint to send parking status
API_URL = 'http://127.0.0.1:5000/update'
from flask import Flask
import threading

# Create a shared buffer for frame data
latest_frame = None

# Video capture
cap = cv2.VideoCapture("carPark.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
#video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parkingmanager = ParkingManagement(
    model="best (4).pt",  # path to model file
    json_file="bounding_boxes.json",  # path to parking annotations file
    device="cuda",  # device to run the model on (cpu or cuda)
)

# Track last known parking status
last_occupied = 0
last_available = 0

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    #im01 = cv2.resize(im0, (1080, 600))    
    results = parkingmanager(im0)

    # Display the processed frame
    cv2.imshow("im0", results.plot_im)
    # Optionally: show original resized frame
    # cv2.imshow("im0", im01)

    # Check occupancy status
    current_occupied = parkingmanager.pr_info['Occupancy']
    current_available = parkingmanager.pr_info['Available']

    # Only update the API if status changed
    if current_occupied != last_occupied or current_available != last_available:
        last_occupied = current_occupied
        last_available = current_available

        try:
            payload = {
                "occupied": current_occupied,
                "available": current_available,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print("API updated successfully.")
            else:
                print(f"API update failed. Status code: {response.status_code}")
        except Exception as e:
            print(f"Failed to send API update: {e}")

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
    # Update shared frame
    #latest_frame = results.plot_im.copy()
    
cap.release()
cv2.destroyAllWindows()  # destroy all opened windows