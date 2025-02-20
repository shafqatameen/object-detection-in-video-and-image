from ultralytics import YOLO
import cv2 as cv

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8s.pt")  # 'n' stands for nano (small and fast)

# Load an 
image_path = "data/img/000031.png"  # Change to your image path
image = cv.imread(image_path)

# Run object detection
results = model(image)

# Display the results
for r in results:
    r.show()  # Shows the image with detections

# Save output image
results[0].save(filename="output.jpg")
