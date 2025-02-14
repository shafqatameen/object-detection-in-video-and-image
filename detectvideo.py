from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("data/video/cars on road1.mp4")  # Load a video

model = YOLO("yolov8n.pt") 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO
    annotated_frame = results[0].plot()  # Draw boxes
    
    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
