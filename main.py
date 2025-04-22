import cv2
import json
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
from ultralytics import solutions
import time 

parkingmanager = solutions.ParkingManagement(
    model="yolov8s.pt",
    json_file="bounding_boxes.json",
)

model = YOLO("yolov8s.pt")

with open("bounding_boxes.json", "r") as f:
    parking_zones = json.load(f)

def is_box_in_zone(box, zone_pts):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cv2.pointPolygonTest(np.array(zone_pts, dtype=np.int32), (cx, cy), False) >= 0

cap = cv2.VideoCapture(1)
assert cap.isOpened(), "❌ Error reading video file"

window = tk.Tk()

target_width = 821
target_height = 400

status_dict = {}

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame_resized = cv2.resize(frame, (target_width, target_height))

    results = model(frame_resized)[0]

    car_boxes = [
        box.xyxy[0].cpu().numpy()
        for box in results.boxes
        if int(box.cls[0]) in [2, 3, 5, 7]
    ]

    for zone_index, zone in enumerate(parking_zones):
        zone_id = zone_index + 1
        pts = zone["points"]
        occupied = False

        for box in car_boxes:
            if is_box_in_zone(box, pts):
                occupied = True
                break

        status_dict[f"spot{zone_id}"] = "occupied" if occupied else "available"
    status = status_dict.copy()
    with open("status.json", "w") as f:
        json.dump(status, f)
    print("Updated:", status)
    for box in car_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image)

    label.config(image=photo)
    label.image = photo

    window.geometry(f"{image.width}x{image.height}")

    window.after(1, update_frame)

label = tk.Label(window)
label.pack()

update_frame()

window.mainloop()

cap.release()