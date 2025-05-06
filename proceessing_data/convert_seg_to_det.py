import os
import numpy as np

# Đường dẫn
SEG_LABELS_DIR_TRAIN = r"D:\THIEN_PROJECT\person_detection\datasets\labels\train"
SEG_LABELS_DIR_VAL = r"D:\THIEN_PROJECT\person_detection\datasets\labels\val"
DET_LABELS_DIR_TRAIN = r"D:\THIEN_PROJECT\person_detection\datasets\labels_det\train"
DET_LABELS_DIR_VAL = r"D:\THIEN_PROJECT\person_detection\datasets\labels_det\val"

def convert_seg_to_det(label_file, output_file):
    success_lines = 0
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    formatted_lines = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue  # skip dòng trống
        
        parts = list(map(float, line.split()))
        if len(parts) < 7:  # class_id + ít nhất 3 point (x y)
            print(f"Error: Line {line_num} in {label_file} has too few points.")
            continue

        class_id = int(parts[0])
        coords = parts[1:]

        if len(coords) % 2 != 0:
            print(f"Error: Line {line_num} in {label_file} has odd number of coordinates.")
            continue

        points = np.array(coords).reshape(-1, 2)
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        width, height = x_max - x_min, y_max - y_min
        center_x, center_y = x_min + width / 2, y_min + height / 2

        formatted_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        success_lines += 1

    if success_lines > 0:
        with open(output_file, 'w') as f:
            f.write('\n'.join(formatted_lines) + '\n')
        return True
    else:
        print(f"Warning: No valid object in {label_file}")
        return False

def process_labels_dir(seg_dir, det_dir, report_file):
    os.makedirs(det_dir, exist_ok=True)
    total_files = 0
    success_files = 0
    failed_files = []

    with open(report_file, 'w') as report:
        report.write(f"Report for converting {seg_dir} to {det_dir}\n")
        report.write("-" * 50 + "\n")

        for label_file in os.listdir(seg_dir):
            if label_file.endswith('.txt'):
                total_files += 1
                input_path = os.path.join(seg_dir, label_file)
                output_path = os.path.join(det_dir, label_file)
                success = convert_seg_to_det(input_path, output_path)
                if success:
                    success_files += 1
                    report.write(f"[OK] {label_file}\n")
                else:
                    failed_files.append(label_file)
                    report.write(f"[FAIL] {label_file}\n")

        report.write("\nSummary:\n")
        report.write(f"Total files: {total_files}\n")
        report.write(f"Success: {success_files}\n")
        report.write(f"Failed: {len(failed_files)}\n")

    print(f"\nDone. {success_files}/{total_files} files converted successfully.")
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    print("Converting train labels...")
    process_labels_dir(SEG_LABELS_DIR_TRAIN, DET_LABELS_DIR_TRAIN, "train_conversion_report.txt")
    print("Converting val labels...")
    process_labels_dir(SEG_LABELS_DIR_VAL, DET_LABELS_DIR_VAL, "val_conversion_report.txt")
