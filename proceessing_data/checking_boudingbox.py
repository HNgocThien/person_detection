import os
import cv2

# Thư mục ảnh
IMAGES_DIR = r"D:\THIEN_PROJECT\person_detection\datasets\images\train"
# Thư mục nhãn bbox (labels_det)
LABELS_DIR = r"D:\THIEN_PROJECT\person_detection\datasets\labels_det\train"

def draw_bbox(img_path, label_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)

            # Chuyển từ YOLO format về pixel
            x_center *= w
            y_center *= h
            width *= w
            height *= h

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Vẽ bbox màu đỏ
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            # Optional: Vẽ text class id
            cv2.putText(img, f"ID: {int(class_id)}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img

def show_all_images():
    for file_name in os.listdir(IMAGES_DIR):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(IMAGES_DIR, file_name)
            label_name = os.path.splitext(file_name)[0] + '.txt'
            label_path = os.path.join(LABELS_DIR, label_name)

            if not os.path.exists(label_path):
                print(f"Label not found for {file_name}, skipping.")
                continue

            img_with_bbox = draw_bbox(img_path, label_path)
            cv2.imshow("Check Bounding Box", img_with_bbox)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_all_images()
