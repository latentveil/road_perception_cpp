from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.onnx"
IMAGE_PATH = (
    PROJECT_ROOT
    / "data/ua_detrac/yolo/images/val/MVI_20012_img00001.jpg"
)
OUTPUT_PATH = PROJECT_ROOT / "results/detection/onnx_single_image.jpg"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "van",
    3: "others",
}


def letterbox(image, size):
    height, width = image.shape[:2]

    scale = min(size / width, size / height)

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

    return padded, scale, left, top


def preprocess(image):
    input_image, scale, pad_left, pad_top = letterbox(
        image,
        INPUT_SIZE,
    )

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_tensor = input_image.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return (
        np.ascontiguousarray(input_tensor),
        scale,
        pad_left,
        pad_top,
    )


def postprocess(
    output,
    original_shape,
    scale,
    pad_left,
    pad_top,
):
    predictions = output[0].T

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:
        class_scores = prediction[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])

        if score < CONF_THRESHOLD:
            continue

        center_x, center_y, width, height = prediction[:4]

        x1 = float(center_x - width / 2)
        y1 = float(center_y - height / 2)
        x2 = float(center_x + width / 2)
        y2 = float(center_y + height / 2)

        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        image_height, image_width = original_shape[:2]

        x1 = np.clip(x1, 0, image_width - 1)
        y1 = np.clip(y1, 0, image_height - 1)
        x2 = np.clip(x2, 0, image_width - 1)
        y2 = np.clip(y2, 0, image_height - 1)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    final_detections = []

    for class_id in sorted(set(class_ids)):
        indices = [
            i
            for i, current_class_id in enumerate(class_ids)
            if current_class_id == class_id
        ]

        class_boxes = []
        class_scores = []

        for i in indices:
            x1, y1, x2, y2 = boxes[i]

            class_boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ]
            )
            class_scores.append(scores[i])

        keep = cv2.dnn.NMSBoxes(
            class_boxes,
            class_scores,
            CONF_THRESHOLD,
            IOU_THRESHOLD,
        )

        for kept_index in keep:
            original_index = indices[int(kept_index)]

            final_detections.append(
                (
                    boxes[original_index],
                    scores[original_index],
                    class_ids[original_index],
                )
            )

    final_detections.sort(
        key=lambda detection: detection[1],
        reverse=True,
    )

    return final_detections


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    input_tensor, scale, pad_left, pad_top = preprocess(image)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    input_name = session.get_inputs()[0].name
    output = session.run(
        None,
        {input_name: input_tensor},
    )[0]

    detections = postprocess(
        output,
        image.shape,
        scale,
        pad_left,
        pad_top,
    )

    print("providers:", session.get_providers())
    print("input shape:", input_tensor.shape)
    print("output shape:", output.shape)
    print("detections:", len(detections))

    for box, score, class_id in detections:
        x1, y1, x2, y2 = box

        print(
            f"class={CLASS_NAMES[class_id]:6s} "
            f"score={score:.4f} "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

        cv2.rectangle(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (0, 255, 0),
            2,
        )

        label = f"{CLASS_NAMES[class_id]} {score:.2f}"

        cv2.putText(
            image,
            label,
            (int(round(x1)), max(int(round(y1)) - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(f"Failed to save image: {OUTPUT_PATH}")

    print("output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
