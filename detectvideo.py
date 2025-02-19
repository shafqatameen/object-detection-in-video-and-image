
from ultralytics import YOLO
import cv2 as cv

# Load the video
cap = cv.VideoCapture("data/video/cars on road1.mp4")

# Load YOLO model
model = YOLO("yolov8s.pt")

# Get video propertiescl
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
video_writer = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()  # Draw detections

    # Save the frame
    video_writer.write(annotated_frame)

    # Display the frame
    cv.imshow("YOLO Detection", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
video_writer.release()
cv.destroyAllWindows()

print("Video saved as output.mp4")

