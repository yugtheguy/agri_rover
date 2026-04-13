import os
from ultralytics import YOLO


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "scripts", "runs", "detect", "runs", "weed_model", "weights", "best.pt")
    images_dir = os.path.join(root, "data", "test", "images")
    labels_dir = os.path.join(root, "data", "test", "labels")

    model = YOLO(model_path)

    image_files = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f))
            and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        ]
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        actual = 0
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            if int(float(parts[0])) == 1:
                                actual = 1
                                break
                        except ValueError:
                            pass

        predicted = 0
        result = model(image_path, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            for cls_id, conf in zip(classes, confidences):
                if int(cls_id) == 1 and float(conf) > 0.3:
                    predicted = 1
                    break

        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"Total images: {total}")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
