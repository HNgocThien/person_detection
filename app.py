import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load model
MODEL_PATH = r"D:\THIEN_PROJECT\person_detection\models\best.pt"
model = YOLO(MODEL_PATH)

# Function to process image
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Không thể đọc ảnh: {image_path}")
        return None

    results = model.predict(source=image, save=False)
    for result in results:
        if result.boxes is not None:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Streamlit app
def main():
    st.title("Test YOLOv8 Model")
    uploaded_file = st.file_uploader("Tải lên một ảnh", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_path = f"temp_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(Image.open(image_path), caption="Ảnh gốc", use_container_width=True)

        processed_image = process_image(image_path)
        if processed_image is not None:
            st.image(processed_image, caption="Kết quả nhận diện", use_container_width=True)

if __name__ == "__main__":
    main()