from ultralytics import YOLO
import os
import sys
import yaml
import torch
from typing import Optional

class YOLOTrainerPro:
    """YOLOTrainer Pro: Class nâng cấp với tối ưu mạnh mẽ"""

    def __init__(self):
        self.pretrained_model_path = "./yolo11n-seg.pt"
        self.data_yaml_path = "./datasets/data.yaml"
        self.output_dir = "./runs"

        self.epochs = 100
        self.batch_size = 16
        self.img_size = 640
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[YOLO] = None

    def check_files(self) -> bool:
        """Check pretrained model & data.yaml"""
        for file_path, desc in [
            (self.pretrained_model_path, "pretrained model"),
            (self.data_yaml_path, "data.yaml config")
        ]:
            if not os.path.exists(file_path):
                print(f"❌ Missing {desc}: {file_path}")
                return False
        return True

    def verify_dataset(self):
        """Verify dataset structure + label format"""
        dataset_dir = os.path.dirname(self.data_yaml_path)
        for split in ["train", "val"]:
            img_dir = os.path.join(dataset_dir, split, "images")
            label_dir = os.path.join(dataset_dir, split, "labels_det")

            if not os.path.exists(img_dir) or not os.path.exists(label_dir):
                raise FileNotFoundError(f"❌ Missing {split} data dirs: {img_dir} or {label_dir}")

            imgs = set(f.split('.')[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg')))
            labels = set(f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt'))

            if imgs != labels:
                raise ValueError(f"❌ Mismatch between {split} images and labels")

            for label_file in os.listdir(label_dir):
                if label_file.endswith('.txt'):
                    with open(os.path.join(label_dir, label_file)) as f:
                        lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            raise ValueError(f"❌ Invalid label format in {label_file}: {line}")
                        cls, x, y, w, h = map(float, parts)
                        if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            raise ValueError(f"❌ Invalid values in {label_file}: {line}")
        print("✅ Dataset verified!")

    def load_model(self) -> bool:
        try:
            print(f"🔄 Loading pretrained model: {self.pretrained_model_path}")
            self.model = YOLO(self.pretrained_model_path)
            print("✅ Model loaded!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def train_model(self) -> None:
        if not self.model:
            print("❌ Model not loaded!")
            return

        print(f"\n🚀 Starting training on {self.device}...")

        training_params = {
            "data": self.data_yaml_path,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.img_size,
            "device": self.device,
            "project": self.output_dir,
            "name": "yolov8n_custom",
            "exist_ok": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "patience": 10,
            "augment": True,
            "mosaic": 1.0,
            "mixup": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 2.0,
            "perspective": 0.0001,
            "flipud": 0.5,
            "fliplr": 0.5,
        }

        results = self.model.train(**training_params)
        print("✅ Training done!")

    def validate_model(self) -> None:
        if not self.model:
            print("❌ Model not loaded!")
            return

        print("\n🔍 Running validation...")
        metrics = self.model.val()
        print(f"✅ Validation mAP@50: {metrics.box.map50:.4f}, mAP@50:95: {metrics.box.map:.4f}")

    def export_model(self) -> None:
        if not self.model:
            print("❌ Model not loaded!")
            return

        print("\n📦 Exporting model to ONNX...")
        try:
            success = self.model.export(format="onnx")
            print("✅ Exported to ONNX!" if success else "❌ Export failed.")
        except Exception as e:
            print(f"❌ Export error: {e}")

    def save_final_weights(self) -> None:
        final_path = os.path.join(self.output_dir, "yolov8n_custom", "weights", "final.pt")
        self.model.save(final_path)
        print(f"💾 Final model saved at {final_path}")

    def run(self) -> None:
        if not self.check_files():
            sys.exit(1)

        self.verify_dataset()

        if not self.load_model():
            sys.exit(1)

        self.train_model()
        self.validate_model()
        self.export_model()
        self.save_final_weights()

        print("\n🏁 All done!")

def main():
    trainer = YOLOTrainerPro()
    trainer.run()

if __name__ == "__main__":
    main()
