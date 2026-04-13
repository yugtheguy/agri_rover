from ultralytics import YOLO
import os


def main():
    data_yaml = os.path.join(os.path.dirname(__file__), "data.yaml")

    model = YOLO("yolov8n.pt")

    print("Training started")

    model.train(
        data=data_yaml,
        epochs=25,
        imgsz=416,
        batch=8,
        cache=True,
        workers=2,
        patience=10,
        project="runs",
        name="weed_model"
    )


if __name__ == "__main__":
    main()