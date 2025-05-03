import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image
import shutil

# Hàm dự đoán cho một ảnh
def predict_image(image_path, model):
    # In thông báo đang dự đoán
    st.write(f"🔍 Đang dự đoán: {image_path}")
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
    os.makedirs("ket_qua", exist_ok=True)
    output_name = os.path.basename(image_path)
    output_path = os.path.join("ket_qua", output_name)
    cv2.imwrite(output_path, image)
    # In thông báo đã lưu
    st.write(f"✅ Đã lưu vào: {output_path}")
    return image, output_path

# Ứng dụng Streamlit
def main():
    st.title("Ứng Dụng Nhận Diện Phân Đoạn YOLO")
    st.write("Tải lên một ảnh hoặc thư mục chứa ảnh để nhận diện phân đoạn.")

    # Nhập đường dẫn mô hình
    model_path = st.text_input("Đường dẫn đến file mô hình YOLO (.pt)", 
                              r"D:\THIEN_PROJECT\person_detection\runs\weights\best.pt")
    
    # Tải mô hình
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return

    # Tùy chọn tải lên
    upload_type = st.radio("Chọn loại tải lên:", ("Tệp ảnh", "Thư mục chứa ảnh"))

    if upload_type == "Tệp ảnh":
        uploaded_file = st.file_uploader("Chọn một tệp ảnh (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Lưu ảnh tạm thời
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Hiển thị ảnh gốc
            st.subheader("Ảnh Gốc")
            st.image(Image.open(image_path), use_column_width=True)

            # Dự đoán và hiển thị kết quả
            with st.spinner("Đang xử lý..."):
                result_image, output_path = predict_image(image_path, model)
            
            # Hiển thị ảnh kết quả
            st.subheader("Kết Quả Nhận Diện")
            st.image(Image.open(output_path), use_column_width=True)

            # Xóa tệp tạm
            os.remove(image_path)

    else:  # Thư mục chứa ảnh
        uploaded_folder = st.file_uploader("Tải lên thư mục chứa ảnh (zip)", type=["zip"])
        
        if uploaded_folder is not None:
            # Giải nén thư mục
            temp_dir = "temp_folder"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            zip_path = os.path.join(temp_dir, uploaded_folder.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_folder.getbuffer())
            
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Tìm tất cả ảnh trong thư mục
            image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if not image_files:
                st.warning("❌ Không tìm thấy ảnh trong thư mục!")
            else:
                st.subheader("Kết Quả Nhận Diện")
                for file_name in image_files:
                    image_path = os.path.join(temp_dir, file_name)
                    
                    # Hiển thị ảnh gốc
                    st.write(f"**Ảnh Gốc: {file_name}**")
                    st.image(Image.open(image_path), use_column_width=True)
                    
                    # Dự đoán và hiển thị kết quả
                    with st.spinner(f"Đang xử lý {file_name}..."):
                        result_image, output_path = predict_image(image_path, model)
                    
                    # Hiển thị ảnh kết quả
                    st.write(f"**Kết Quả: {file_name}**")
                    st.image(Image.open(output_path), use_column_width=True)
            
            # Xóa thư mục tạm
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()