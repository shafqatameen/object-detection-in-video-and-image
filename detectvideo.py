/*************  ✨ Codeium Command 🌟  *************/
"""
README FOR THIS FILE

This script performs object detection on a video file using the YOLOv8 model. 
The video is processed frame by frame, with detected objects annotated on each frame.
The annotated video is saved to an output file, and the video is also displayed in 
real-time with annotations.

Requirements:
- Python 3.x
- OpenCV
- ultralytics library (for YOLO model)

How to Run:
1. Ensure that the required libraries are installed.
2. Set the correct path for the input video file.
3. Run the script.
4. The output video will be saved in the current working directory as "output.mp4".

"""

/*************  ✨ Codeium Command 🌟  *************/
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


/******  567a63f6-c823-4bcc-90e8-4d9e4334e519  *******/