def main():
    try:
        import torch

        print(torch.cuda.is_available())
    except Exception as e:
        print(str(e))
        return
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        print("Model loaded successfully")
        results = model("https://ultralytics.com/images/bus.jpg")
        print(results[0].verbose() if results else "No results")
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    main()
