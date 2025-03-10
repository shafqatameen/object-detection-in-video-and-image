# Object Detection in Images and Videos using YOLOv8

## README for Image Detection
This Python script detects objects in an image using the YOLOv8 model.

### Features:
- Loads an image and applies object detection using YOLOv8.
- Displays the results using Matplotlib.
- Saves the output image with detected objects.

### Requirements:
- Python 3.x
- OpenCV
- ultralytics library (for YOLO model)
- Matplotlib

### How to Run:
1. Ensure that the required libraries are installed:
   ```bash
   pip install ultralytics opencv-python matplotlib
   ```
2. Set the correct path for the input image file.
3. Run the script using:
   ```bash
   python detectimg.py
   ```
4. The output image will be saved as `output.png` in the current working directory.

---

## README for Video Detection
This script performs object detection on a video file using the YOLOv8 model. The video is processed frame by frame, with detected objects annotated on each frame.

### Features:
- Processes a video and applies object detection on each frame.
- Displays the video with real-time annotations.
- Saves the annotated video as an output file.

### Requirements:
- Python 3.x
- OpenCV
- ultralytics library (for YOLO model)

### How to Run:
1. Ensure that the required libraries are installed:
   ```bash
   pip install ultralytics opencv-python
   ```
2. Set the correct path for the input video file.
3. Run the script using:
   ```bash
   python detectvideo.py
   ```
4. The output video will be saved as `output.mp4` in the current working directory.

