from ultralytics import YOLO
import os
import sys
from typing import Optional

class YOLOTrainer:
    """Class để quản lý quá trình huấn luyện YOLO"""
    
    def __init__(self):
        # Cấu hình đường dẫn và tham số
        self.pretrained_model_path = "./yolo11n-seg.pt"
        self.data_yaml_path = "./datasets/data.yaml"
        self.epochs = 100
        self.batch_size = 16
        self.model: Optional[YOLO] = None

    def check_files(self) -> bool:
        """Kiểm tra sự tồn tại của các file cần thiết"""
        for file_path, file_desc in [
            (self.pretrained_model_path, "mô hình pretrained"),
            (self.data_yaml_path, "cấu hình dữ liệu data.yaml")
        ]:
            if not os.path.exists(file_path):
                print(f"Lỗi: Không tìm thấy tệp {file_desc}: {file_path}")
                return False
        return True

    def load_model(self) -> bool:
        """Load model YOLO pretrained"""
        try:
            print(f"Đang load model pretrained từ: {self.pretrained_model_path}")
            self.model = YOLO(self.pretrained_model_path)
            print("Model pretrained đã được load thành công!")
            return True
        except Exception as e:
            print(f"Lỗi khi load model: {e}")
            return False

    def train_model(self) -> None:
        """Huấn luyện mô hình"""
        if not self.model:
            print("Lỗi: Model chưa được load!")
            return

        print("\n--- Bắt đầu quá trình huấn luyện ---")
        print(f"Dataset YAML: {self.data_yaml_path}")
        print(f"Số epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")

        try:
            results_train = self.model.train(
                data=self.data_yaml_path,
                epochs=self.epochs,
                batch=self.batch_size,
                verbose=True
            )
            print("\n--- Kết quả huấn luyện ---")
            print(results_train)
            print("Quá trình huấn luyện hoàn tất!")
        except Exception as e:
            print(f"Lỗi trong quá trình huấn luyện: {e}")
            sys.exit(1)

    def validate_model(self) -> None:
        """Đánh giá mô hình"""
        if not self.model:
            print("Lỗi: Model chưa được load!")
            return

        print("\n--- Bắt đầu quá trình đánh giá (Validation) ---")
        try:
            results_val = self.model.val()
            print("\n--- Kết quả đánh giá (Validation) ---")
            print(results_val)
            print("Quá trình đánh giá hoàn tất!")
        except Exception as e:
            print(f"Lỗi trong quá trình đánh giá: {e}")
            sys.exit(1)

    def export_model(self) -> None:
        """Xuất mô hình sang ONNX"""
        if not self.model:
            print("Lỗi: Model chưa được load!")
            return

        print("\n--- Bắt đầu xuất mô hình sang ONNX ---")
        try:
            success = self.model.export(format="onnx")
            print("Xuất mô hình sang ONNX thành công!" if success else "Xuất mô hình thất bại.")
        except Exception as e:
            print(f"Lỗi khi xuất ONNX: {e}")
            sys.exit(1)

    def run(self) -> None:
        """Chạy toàn bộ pipeline"""
        if not self.check_files():
            sys.exit(1)
        if not self.load_model():
            sys.exit(1)
        
        self.train_model()
        self.validate_model()
        self.export_model()
        print("\n--- Hoàn thành toàn bộ quá trình ---")

def main():
    trainer = YOLOTrainer()
    trainer.run()

if __name__ == "__main__":
    main()