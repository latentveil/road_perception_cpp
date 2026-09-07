from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.pt"


def main():
    model = YOLO(MODEL_PATH)

    output_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        nms=False,
        device=0,
    )

    print("ONNX export finished")
    print("output:", output_path)


if __name__ == "__main__":
    main()
