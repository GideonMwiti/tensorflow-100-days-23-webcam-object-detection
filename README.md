# Project 23: Real-Time Object Detection with Webcam

Use TensorFlow Hub's SSD MobileNet V2 model to perform object detection on a live webcam feed using OpenCV.

## Requirements
- OpenCV (`pip install opencv-python`)
- Working webcam (locally)

## How it Works
1. **Webcam Capture**: Uses `cv2.VideoCapture(0)` to grab frames.
2. **Inference Pipeline**:
   - Converts frames from OpenCV's default BGR to RGB.
   - Converts to a `uint8` tensor with batch dimension `[1, H, W, 3]`.
   - Runs inference on the pre-trained SSD MobileNet V2 model loaded from TensorFlow Hub (`https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2`).
3. **Bounding Boxes & Text Overlay**: Draws bounding boxes and COCO category labels on each frame using OpenCV's `cv2.rectangle` and `cv2.putText` functions.
4. **Display**: Displays the annotated live video stream in a GUI window named `Real-Time Object Detection`.
5. **Exit**: Releases webcam and closes windows when the user presses `q`.

## Headless Fallback Mode
When run in a headless environment where no physical camera is accessible:
- The script automatically detects that `VideoCapture(0).isOpened()` is False.
- It falls back to processing a static test image.
- It runs the SSD MobileNet V2 model, applies the OpenCV drawings, saves the annotated image as `webcam_detection_results.png`, and exits.

## How to Run
Run the detection script:
```powershell
python main.py
```

## Results
The script saves the annotated image `webcam_detection_results.png` in fallback mode:

![Webcam Object Detection Results](webcam_detection_results.png)

## Repository Details
- **Name**: `tensorflow-100-days-23-webcam-object-detection`
- **Description**: Real-time object detection on live webcam feed using SSD MobileNet V2 from TF Hub and OpenCV drawing overlays, with a headless fallback pipeline.
- **Topics**: `tensorflow`, `deep-learning`, `object-detection`, `ssd-mobilenet`, `opencv`, `webcam`, `real-time`
