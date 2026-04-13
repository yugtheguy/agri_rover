import os
from ultralytics import YOLO
import cv2
import time


def open_stream(stream_url):
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_timeout is not None:
        cap.set(open_timeout, 5000)
    if read_timeout is not None:
        cap.set(read_timeout, 5000)
    return cap


def main():
    model = YOLO("runs/detect/runs/weed_model/weights/best.pt")
    stream_url = "http://192.168.1.211:81/stream"
    save_dir = "weed_detected_frames"
    os.makedirs(save_dir, exist_ok=True)
    cap = open_stream(stream_url)

    while not cap.isOpened():
        print("Stream not accessible")
        time.sleep(2)
        cap.release()
        cap = open_stream(stream_url)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream not accessible")
            cap.release()
            time.sleep(1)
            cap = open_stream(stream_url)
            continue

        result = model(frame, verbose=False)[0]
        weed_count = 0

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if conf < 0.3:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                if cls_id == 1:
                    weed_count += 1
                    color = (0, 0, 255)
                    label = f"weed {conf:.2f}"
                else:
                    color = (0, 255, 0)
                    label = f"crop {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        weed_present = weed_count > 0
        print(f"Weed Present: {weed_present}")
        print(f"Weed Count: {weed_count}")

        if weed_present:
            filename = f"weed_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), frame)

        cv2.imshow("ESP32 Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
