# object-detection-in-video-and-image


. README FOR IMAGE
. This program is a Python script that detects objects in an image using the YOLOv8 model.
. It loads the image, detects objects, displays the results, and saves the output image.
. The model is downloaded from the ultralytics model zoo.
. The image is loaded using OpenCV.
. The model is run on the image using the ultralytics YOLO class.
. The results are displayed using Matplotlib.
. The output image is saved using OpenCV.
. The program can be run using Python.
. The program requires the ultralytics library.
. The program requires OpenCV.
. The program requires Matplotlib.

Requirements:
- Python 3.x
- OpenCV
- ultralytics library (for YOLO model)

How to Run:
1. Ensure that the required libraries are installed.
2. Set the correct path for the input IMAGE file.
3. Run the script.
4. The output IMAGE will be saved in the current working directory as "output.png".


-----------------------------------------README FOR VIDEO-----------------------------------------

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