import os
import sys
import cv2
from ultralytics import YOLO


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Error: Image not found")
        sys.exit(1)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "runs", "detect", "train", "weights", "best.pt")

    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        sys.exit(1)

    result = model(image_path, verbose=False)[0]
    weed_count = 0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if cls_id == 1 and conf > 0.3:
                weed_count += 1

            color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)
            label = ("weed" if cls_id == 1 else "crop") + f" {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    weed_present = weed_count > 0

    print(f"Weed Present: {weed_present}")
    print(f"Weed Count: {weed_count}")

    cv2.imshow("Inference", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
