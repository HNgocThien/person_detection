import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Configuration
MODEL_PATH = r"D:\THIEN_PROJECT\person_detection\models\yolov8n.pt"  # Path to YOLOv8n model
DEFAULT_VIDEO_PATH = r"D:\THIEN_PROJECT\person_detection\sample_video.mp4"  # Default video path
DEFAULT_IMAGE_PATH = r"D:\THIEN_PROJECT\person_detection\datasets\images\train\000000000077.jpg"  # Default image path
INPUT_TYPE = "image"  # Options: "webcam", "video", "image"

# Load YOLOv8n model
model = YOLO(MODEL_PATH)

# Define class names (COCO dataset, class 0 is 'person')
class_names = model.names

# Function to process frame and draw bounding boxes
def process_frame(frame):
    # Perform inference
    results = model(frame)[0]
    
    # Initialize person count
    person_count = 0
    
    # Process detection results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        # Only process 'person' class (class ID 0 in COCO)
        if cls == 0:
            person_count += 1
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display person count on frame
    cv2.putText(frame, f"People: {person_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, person_count

# Function to list available devices
def list_available_devices(max_index=10):
    available_devices = []
    for i in range(max_index):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG]:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                available_devices.append((i, backend))
                print(f"Device {i} is available with backend {backend}")
                cap.release()
    return available_devices

# Function to initialize video capture with retry mechanism
def init_video_capture(input_source=0, backends=[cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG]):
    # Determine input source based on INPUT_TYPE
    if isinstance(input_source, str):
        source = input_source
    else:
        if INPUT_TYPE == "video":
            source = DEFAULT_VIDEO_PATH
        elif INPUT_TYPE == "image":
            source = DEFAULT_IMAGE_PATH
        else:  # webcam
            source = input_source

    # Try different device indices and backends if input_source is an integer (webcam)
    if isinstance(source, int):
        for backend in backends:
            for index in [source, 0, 1, 2, -1]:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    print(f"Successfully opened device index {index} with backend {backend}")
                    # Set resolution to improve compatibility
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return cap
                cap.release()
        print("Error: Could not open any webcam. Available devices:")
        list_available_devices()
        return None
    
    # Handle video or image file
    for backend in backends:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"Successfully opened input source {source} with backend {backend}")
            return cap
        cap.release()
    print(f"Error: Could not open input source: {source}")
    return None

# Main function to handle webcam, video, or image input
def main(input_source=0, backends=[cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG]):
    # Initialize video capture
    cap = init_video_capture(input_source, backends)
    if cap is None:
        sys.exit(1)
    
    # Check if input is an image
    if isinstance(input_source, str) and input_source.lower().endswith(('.png', '.jpg', '.jpeg')) or INPUT_TYPE == "image":
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read image.")
            cap.release()
            sys.exit(1)
        
        # Process and display image
        processed_frame, person_count = process_frame(frame)
        cv2.imshow("YOLOv8 Person Detection", processed_frame)
        print(f"Detected {person_count} people in the image.")
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()
        cap.release()
        return
    
    # Process video or webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Check webcam or video source.")
            break
        
        # Process frame and get person count
        processed_frame, person_count = process_frame(frame)
        
        # Display the frame
        cv2.imshow("YOLOv8 Person Detection", processed_frame)
        
        # Print person count to console
        print(f"Detected {person_count} people in frame.")
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use INPUT_TYPE to determine input source
    # Can override with command-line input_source: main("path/to/video.mp4", [cv2.CAP_FFMPEG, cv2.CAP_DSHOW])
    try:
        main(0, [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Available devices:")
        list_available_devices()
        sys.exit(1)