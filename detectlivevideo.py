from ultralytics import YOLO
import cv2 as cv
import numpy as np

model = YOLO("yolov8x.pt")
cap = cv.VideoCapture(0)
conf_threshold, iou_threshold = 0.5, 0.45

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    results = model(frame, conf=conf_threshold, iou=iou_threshold)
    
    # Log results for debugging
    print(f"Detections: {[cls.name for cls in results[0].boxes.cls]}")
    
    annotated_frame = results[0].plot()
    cv.imshow('YOLOv8x Detection', annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

'''from ultralytics import YOLO
import cv2 as cv
import numpy as np

# Load the pre-trained YOLOv8 model (medium size)
model = YOLO("yolov8x.pt")

# Initialize video capture from webcam (0) or video file
cap = cv.VideoCapture(0)  # Use 0 for webcam, or path to video file for a video

# Define parameters for detection
conf_threshold = 0.5  # Confidence threshold for detection
iou_threshold = 0.45  # IoU threshold for NMS

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Perform object detection on the frame with adjusted thresholds
    results = model(frame, conf=conf_threshold, iou=iou_threshold)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv.imshow('YOLOv8m Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()'''