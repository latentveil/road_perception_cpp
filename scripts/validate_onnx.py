from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PT_MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.pt"
ONNX_MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.onnx"
IMAGE_PATH = (
    PROJECT_ROOT
    / "data/ua_detrac/yolo/images/val/MVI_20012_img00001.jpg"
)

INPUT_SIZE = 640


def letterbox(image, size):
    height, width = image.shape[:2]

    scale = min(size / height, size / width)

    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_width = size - resized_width
    pad_height = size - resized_height

    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded


def preprocess(image):
    image = letterbox(image, INPUT_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = image.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor)


def get_pytorch_output(input_tensor):
    model = YOLO(PT_MODEL_PATH)
    model.model.eval()
    model.model.cuda()

    tensor = torch.from_numpy(input_tensor).cuda()

    with torch.no_grad():
        output = model.model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    return output.float().cpu().numpy()


def get_onnx_output(input_tensor):
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]

    print("ONNX providers:", session.get_providers())

    return output


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    input_tensor = preprocess(image)

    pytorch_output = get_pytorch_output(input_tensor)
    onnx_output = get_onnx_output(input_tensor)

    print("input shape:", input_tensor.shape)
    print("PyTorch output shape:", pytorch_output.shape)
    print("ONNX output shape:", onnx_output.shape)

    if pytorch_output.shape != onnx_output.shape:
        raise RuntimeError(
            "Output shape mismatch: "
            f"{pytorch_output.shape} vs {onnx_output.shape}"
        )

    abs_diff = np.abs(pytorch_output - onnx_output)

    print("max abs diff:", float(abs_diff.max()))
    print("mean abs diff:", float(abs_diff.mean()))
    print("PyTorch output range:",
          float(pytorch_output.min()),
          float(pytorch_output.max()))
    print("ONNX output range:",
          float(onnx_output.min()),
          float(onnx_output.max()))


if __name__ == "__main__":
    main()
