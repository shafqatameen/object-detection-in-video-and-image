
from ultralytics import YOLO
import cv2

# Load the video
cap = cv2.VideoCapture("data/video/cars on road1.mp4")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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
    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Video saved as output.mp4")

