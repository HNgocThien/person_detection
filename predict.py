import time
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Configuration
MODEL_PATH = r"D:\THIEN_PROJECT\person_detection\models\best.pt"  # Path to model
DEFAULT_VIDEO_PATH = r"rtsp://akacamai:fpt12345@103.176.147.25:8554/camera_146"  # Default video path
DEFAULT_IMAGE_PATH = r"D:\THIEN_PROJECT\person_detection\datasets\images\test\z6676591869971_e6e2ccb0404b4ef1b98f7a6f84d0275a.jpg"  # Default image path
INPUT_TYPE = "video"  # Options: "webcam", "video", "image"

# Load model
model = YOLO(MODEL_PATH)

# Define class names (COCO dataset, class 0 is 'person')
class_names = model.names

# Định nghĩa polygon vùng cảnh báo (theo thứ tự các điểm)
ALERT_POLYGON = np.array([
    [724, 507],
    [684, 994],
    [1297, 951],
    [1161, 465]
], np.int32)

def is_in_polygon(bbox_center, polygon):
    return cv2.pointPolygonTest(polygon, bbox_center, False) >= 0

# Function to process frame and draw bounding boxes
def process_frame(frame):
    # Vẽ vùng cảnh báo
    cv2.polylines(frame, [ALERT_POLYGON], isClosed=True, color=(0, 0, 255), thickness=3)

    # ĐO THỜI GIAN INFERENCE
    start_time = time.perf_counter()
    results = model(frame)[0]
    infer_time = (time.perf_counter() - start_time) * 1000  # ms

    person_count = 0
    alert = False

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)

        if cls == 0:  # Chỉ quan tâm class 'person'
            person_count += 1
            # Tính tâm bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # Kiểm tra có nằm trong vùng cảnh báo không
            if is_in_polygon(center, ALERT_POLYGON):
                color = (0, 0, 255)  # Đỏ
                alert = True
            else:
                color = (0, 255, 0)  # Xanh lá

            # Vẽ bbox và tâm
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, center, 5, color, -1)

    # Hiển thị cảnh báo nếu có người vào vùng đỏ
    if alert:
        cv2.putText(frame, "CANH BAO: Co nguoi vao vung do!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    # Hiển thị số người
    cv2.putText(frame, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if alert else (0, 255, 0), 2)
    # Hiển thị thời gian inference
    cv2.putText(frame, f"Infer: {infer_time:.1f} ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    # Log ra console
    print(f"Inference time: {infer_time:.1f} ms, People: {person_count}, Alert: {alert}")

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
        display_frame = cv2.resize(processed_frame, (960, 540))
        cv2.imshow("YOLOv8 Person Detection", display_frame)
        print(f"Detected {person_count} people in the image.")
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()
        cap.release()
        return
    
    # Process video or webcam
    frame_idx = 0  # Thêm biến đếm frame
    fps_counter = 0
    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Check webcam or video source.")
            break

        # Chỉ xử lý frame chẵn
        if frame_idx % 2 == 0:
            processed_frame, person_count = process_frame(frame)

            # Resize frame trước khi hiển thị
            display_frame = cv2.resize(processed_frame, (960, 540))
            cv2.imshow("YOLOv8 Person Detection", display_frame)

            print(f"Detected {person_count} people in frame.")

        frame_idx += 1  # Tăng biến đếm frame
        fps_counter += 1
        # Tính FPS
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            print(f"FPS: {fps_counter} frame/s")
            fps_counter = 0
            fps_start_time = current_time
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