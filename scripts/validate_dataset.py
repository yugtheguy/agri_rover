import os
import cv2
import random
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
VALID_CLASSES = {0, 1}


def parse_label_file(label_path):
    boxes = []
    invalid_format = False
    invalid_classes = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return boxes, True, invalid_classes
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid_format = True
            continue
        try:
            class_value = float(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            invalid_format = True
            continue
        if int(class_value) != class_value:
            invalid_classes.append((line_number, parts[0]))
            continue
        class_id = int(class_value)
        if class_id not in VALID_CLASSES:
            invalid_classes.append((line_number, class_id))
            continue
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 <= width <= 1.0
            and 0.0 <= height <= 1.0
        ):
            invalid_format = True
            continue
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes, invalid_format, invalid_classes


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_root = os.path.join(project_root, "data", "train")
    images_dir = os.path.join(train_root, "images")
    labels_dir = os.path.join(train_root, "labels")

    image_files = []
    label_files = []

    if os.path.isdir(images_dir):
        image_files = sorted(
            [
                name
                for name in os.listdir(images_dir)
                if os.path.isfile(os.path.join(images_dir, name))
                and name.lower().endswith(IMAGE_EXTS)
            ]
        )
    if os.path.isdir(labels_dir):
        label_files = sorted(
            [
                name
                for name in os.listdir(labels_dir)
                if os.path.isfile(os.path.join(labels_dir, name)) and name.lower().endswith(".txt")
            ]
        )

    image_by_base = {os.path.splitext(name)[0]: name for name in image_files}
    label_by_base = {os.path.splitext(name)[0]: name for name in label_files}

    missing_label_bases = sorted(set(image_by_base) - set(label_by_base))
    missing_image_bases = sorted(set(label_by_base) - set(image_by_base))

    invalid_files = set()
    invalid_class_ids = []
    class_distribution = {0: 0, 1: 0}
    boxes_by_base = {}

    for label_name in label_files:
        base_name = os.path.splitext(label_name)[0]
        label_path = os.path.join(labels_dir, label_name)
        boxes, invalid_format, invalid_classes = parse_label_file(label_path)
        boxes_by_base[base_name] = boxes
        if invalid_format:
            invalid_files.add(label_name)
        if invalid_classes:
            for line_number, class_id in invalid_classes:
                invalid_class_ids.append((label_name, line_number, class_id))
        for class_id, _, _, _, _ in boxes:
            class_distribution[class_id] += 1

    print("dataset summary")
    print(f"total images: {len(image_files)}")
    print(f"total labels: {len(label_files)}")
    print(f"missing labels: {len(missing_label_bases)}")
    print(f"missing images: {len(missing_image_bases)}")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print("errors")
        if not os.path.isdir(images_dir):
            print("missing folder: data/train/images")
        if not os.path.isdir(labels_dir):
            print("missing folder: data/train/labels")

    if missing_label_bases:
        print("images missing labels")
        for base_name in missing_label_bases:
            print(image_by_base[base_name])

    if missing_image_bases:
        print("labels missing images")
        for base_name in missing_image_bases:
            print(label_by_base[base_name])

    if invalid_files:
        print("invalid files")
        for name in sorted(invalid_files):
            print(name)

    if invalid_class_ids:
        print("invalid class IDs")
        for label_name, line_number, class_id in invalid_class_ids:
            print(f"{label_name} line {line_number}: {class_id}")

    print("class distribution")
    print(f"crop: {class_distribution[0]}")
    print(f"weed: {class_distribution[1]}")

    sample_count = min(10, len(image_files))
    if sample_count == 0:
        return

    sampled_images = random.sample(image_files, sample_count)

    for image_name in sampled_images:
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_h, image_w = image.shape[:2]
        base_name = os.path.splitext(image_name)[0]
        boxes = boxes_by_base.get(base_name, [])

        for class_id, x_center, y_center, width, height in boxes:
            x1 = int((x_center - width / 2.0) * image_w)
            y1 = int((y_center - height / 2.0) * image_h)
            x2 = int((x_center + width / 2.0) * image_w)
            y2 = int((y_center + height / 2.0) * image_h)

            clipped = np.clip(
                np.array([x1, y1, x2, y2], dtype=np.int32),
                np.array([0, 0, 0, 0], dtype=np.int32),
                np.array([image_w - 1, image_h - 1, image_w - 1, image_h - 1], dtype=np.int32),
            )
            x1, y1, x2, y2 = clipped.tolist()

            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
            label = "crop" if class_id == 0 else "weed"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        try:
            cv2.imshow("dataset validation", image)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
        except cv2.error as e:
            print(str(e))
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
