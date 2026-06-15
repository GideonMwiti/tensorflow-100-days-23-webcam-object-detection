import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import urllib.request
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Hardcoded COCO label map for SSD models (1-indexed)
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

def draw_detections(frame, detection_boxes, detection_classes, detection_scores, threshold=0.50):
    """Draws bounding boxes and labels on an OpenCV image frame."""
    height, width = frame.shape[:2]
    
    # Filter valid detections
    valid_indices = np.where(detection_scores > threshold)[0]
    
    # Color palette (BGR representation for OpenCV)
    # Blue, Green, Red, Orange, Purple, Cyan
    colors = [
        (232, 115, 26),   # #1a73e8 (Google Blue) -> (B, G, R)
        (62, 142, 30),     # #1e8e3e (Google Green) -> (B, G, R)
        (37, 48, 217),     # #d93025 (Google Red) -> (B, G, R)
        (0, 116, 227),     # #e37400 (Google Orange) -> (B, G, R)
        (176, 39, 156),    # #9c27b0 (Purple) -> (B, G, R)
        (212, 188, 0)      # #00bcd4 (Cyan) -> (B, G, R)
    ]
    
    for count, idx in enumerate(valid_indices):
        box = detection_boxes[idx]
        class_id = detection_classes[idx]
        score = detection_scores[idx]
        
        class_name = COCO_LABELS.get(class_id, f"Class {class_id}")
        
        # Bounding box coordinates: [ymin, xmin, ymax, xmax] (normalized)
        ymin, xmin, ymax, xmax = box
        
        # Convert to absolute coordinates in pixels
        x_min_px = int(xmin * width)
        y_min_px = int(ymin * height)
        x_max_px = int(xmax * width)
        y_max_px = int(ymax * height)
        
        color = colors[count % len(colors)]
        
        # Draw bounding box rectangle
        cv2.rectangle(frame, (x_min_px, y_min_px), (x_max_px, y_max_px), color, thickness=3)
        
        # Create tag label
        label_text = f"{class_name} ({score * 100:.1f}%)"
        
        # Calculate size of text box
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw small background rectangle for text label
        cv2.rectangle(
            frame,
            (x_min_px, y_min_px - text_h - 10),
            (x_min_px + text_w + 10, y_min_px),
            color,
            thickness=cv2.FILLED
        )
        
        # Overlay white text label on top of background rectangle
        cv2.putText(
            frame,
            label_text,
            (x_min_px + 5, y_min_px - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        
        print(f"Detected: {class_name:<15} | Score: {score * 100:.2f}% | Box: {box}")

    return frame

def run_headless_fallback(detector):
    """Processes a static image and saves output if webcam is not present."""
    print("\nRunning in Headless Fallback Mode...")
    
    # 1. Download COCO test image
    image_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image1.jpg"
    image_path = "test_image.jpg"
    
    print(f"Downloading fallback test image: {image_url}...")
    try:
        urllib.request.urlretrieve(image_url, image_path)
    except Exception as e:
        print(f"Failed to download image: {e}")
        # Create a basic synthetic fallback image with Pillow
        image_path = "fallback_image.jpg"
        img = Image.new("RGB", (640, 480), color=(135, 206, 235))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 320, 640, 480], fill=(34, 139, 34)) # Grass
        draw.ellipse([120, 240, 220, 340], fill=(220, 20, 60), outline=(0,0,0), width=3) # Red ball
        draw.ellipse([420, 220, 540, 340], fill=(30, 144, 255), outline=(0,0,0), width=3) # Blue ball
        img.save(image_path)
        
    # Load using OpenCV (loads as BGR)
    frame = cv2.imread(image_path)
    
    # 2. Preprocess: convert BGR to RGB, add batch dim, convert to uint8
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    # 3. Run Inference
    print("Running object detection inference...")
    detections = detector(image_tensor)
    
    detection_boxes = detections["detection_boxes"][0].numpy()
    detection_classes = detections["detection_classes"][0].numpy().astype(int)
    detection_scores = detections["detection_scores"][0].numpy()
    
    # 4. Draw annotations using OpenCV
    annotated_frame = draw_detections(frame, detection_boxes, detection_classes, detection_scores)
    
    # 5. Save output
    output_path = "webcam_detection_results.png"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Fallback object detection output saved to: {output_path}")

def main():
    print("====================================================")
    print("Project 23: Real-Time Object Detection with Webcam")
    print("Goal: Use SSD MobileNet V2 with OpenCV to run live detection")
    print("====================================================\n")

    # 1. Load SSD MobileNet V2 from TF Hub
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    print(f"Loading SSD MobileNet V2 detector from TF Hub: {model_url}...")
    
    try:
        detector = hub.load(model_url)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Kaggle Models redirect
        fallback_url = "https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1"
        print(f"Attempting to load fallback model: {fallback_url}...")
        detector = hub.load(fallback_url)
        print("Fallback model loaded successfully.")

    # 2. Try to initialize Webcam capture
    print("\nAttempting to open webcam camera feed...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not detected (common in headless/sandboxed environment).")
        # Run headless fallback
        run_headless_fallback(detector)
    else:
        print("Webcam initialized successfully.")
        print("Press 'q' key in the camera window to exit live object detection.")
        
        # Check if running headlessly or in a non-interactive shell (stdin redirected)
        import sys
        is_headless = not sys.stdin.isatty() or os.environ.get('TF_100_DAYS_HEADLESS') == '1'
        
        frame_count = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read webcam frame. Exiting loop.")
                break
                
            # Preprocess: convert BGR to RGB, add batch dim, convert to uint8
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
            image_tensor = tf.expand_dims(image_tensor, axis=0)
            
            # Run Inference
            detections = detector(image_tensor)
            
            detection_boxes = detections["detection_boxes"][0].numpy()
            detection_classes = detections["detection_classes"][0].numpy().astype(int)
            detection_scores = detections["detection_scores"][0].numpy()
            
            # Draw detections
            annotated_frame = draw_detections(frame, detection_boxes, detection_classes, detection_scores)
            
            if is_headless:
                frame_count += 1
                if frame_count >= 5:
                    print("Automated test run: successfully processed 5 frames. Exiting loop.")
                    # Save a sample frame to verify output
                    output_path = "webcam_detection_results.png"
                    cv2.imwrite(output_path, annotated_frame)
                    print(f"Fallback object detection output saved to: {output_path}")
                    break
            else:
                # Display the resulting frame in GUI
                cv2.imshow('Real-Time Object Detection (SSD MobileNet V2)', annotated_frame)
                
                # Press Q on keyboard to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exited webcam loop by user request.")
                    break
                
        # Release the capture and windows
        cap.release()
        cv2.destroyAllWindows()
        
    print("\n====================================================")

if __name__ == "__main__":
    main()
