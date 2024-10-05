import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np

# Function to capture mouse movement and print coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Create a window named 'RGB'
cv2.namedWindow('RGB')
# Set the mouse callback function to capture mouse movement
cv2.setMouseCallback('RGB', RGB)

# Load COCO class names from a file
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv11 model
model = YOLO("yolo11n.pt")

# Open the video capture (0 for webcam)
cap = cv2.VideoCapture(0)
count = 0

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 1020x500
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the bounding boxes, class IDs, track IDs, and confidence scores
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores
       
        # Loop through each detected object and draw bounding boxes and labels
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)  # Draw track ID
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)  # Draw class name

    # Display the frame in the 'RGB' window
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

