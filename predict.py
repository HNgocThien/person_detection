from ultralytics import YOLO
import cv2
import sys
import os
import numpy as np

def predict_image(image_path, model):
    # In thông báo đang dự đoán
    print(f"🔍 Đang dự đoán: {image_path}")
    image = cv2.imread(image_path)
    results = model.predict(source=image, save=False)

    for result in results:
        if result.masks is not None:  # Kiểm tra xem có mặt nạ phân đoạn hay không
            masks = result.masks.data.cpu().numpy()  # Lấy mặt nạ phân đoạn
            boxes = result.boxes.xyxy.cpu().numpy()   # Lấy hộp giới hạn
            scores = result.boxes.conf.cpu().numpy()  # Lấy điểm tin cậy
            labels = result.boxes.cls.cpu().numpy()   # Lấy nhãn lớp

            # Tạo lớp phủ màu cho phân đoạn
            overlay = np.zeros_like(image)
            
            for mask, box, score, label in zip(masks, boxes, scores, labels):
                # Chuyển đổi kích thước mặt nạ
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
                
                # Tạo màu ngẫu nhiên cho mặt nạ
                color = np.random.randint(0, 255, (3), dtype=np.uint8)
                
                # Áp dụng mặt nạ lên lớp phủ
                overlay[mask] = color
                
                # Vẽ hộp giới hạn và nhãn
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{model.names[int(label)]} {score:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Trộn ảnh gốc với lớp phủ
            alpha = 0.4  # Hệ số trong suốt
            image = cv2.addWeighted(image, 1, overlay, alpha, 0)

    # Lưu kết quả
    os.makedirs("results", exist_ok=True)
    output_name = os.path.basename(image_path)
    output_path = os.path.join("results", output_name)
    cv2.imwrite(output_path, image)
    # In thông báo đã lưu
    print(f"✅ Đã lưu vào: {output_path}")

def main(path, model_path=r"D:\THIEN_PROJECT\person_detection\runs\weights\best.pt"):
    model = YOLO(model_path)

    if os.path.isfile(path):
        predict_image(path, model)
    elif os.path.isdir(path):
        for file_name in os.listdir(path):
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(path, file_name)
                predict_image(image_path, model)
    else:
        # In thông báo lỗi nếu đường dẫn không hợp lệ
        print(f"❌ Đường dẫn không hợp lệ: {path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # In hướng dẫn sử dụng
        print("Cách dùng: python predict_segment.py đường_dẫn_đến_ảnh_hoặc_thư_mục")
    else:
        main(sys.argv[1])