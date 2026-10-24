#include "road_perception/detector.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace road_perception {
LetterboxResult letterbox(const cv::Mat &image, int input_size) {
    const int original_width = image.cols;
    const int original_height = image.rows;

    const float scale = std::min(static_cast<float>(input_size) / original_width,
                                 static_cast<float>(input_size) / original_height);

    const int resized_width = static_cast<int>(std::round(original_width * scale));

    const int resized_height = static_cast<int>(std::round(original_height * scale));

    cv::Mat resized;

    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

    const int pad_width = input_size - resized_width;

    const int pad_height = input_size - resized_height;

    const int left = static_cast<int>(std::round(pad_width / 2.0 - 0.1));

    const int right = static_cast<int>(std::round(pad_width / 2.0 + 0.1));

    const int top = static_cast<int>(std::round(pad_height / 2.0 - 0.1));

    const int bottom = static_cast<int>(std::round(pad_height / 2.0 + 0.1));

    cv::Mat padded;

    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    return {padded, scale, left, top};
}

PreprocessResult preprocess(const cv::Mat &image, int input_size) {
    const LetterboxResult letterbox_result = letterbox(image, input_size);

    cv::Mat rgb;

    cv::cvtColor(letterbox_result.image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat float_image;

    rgb.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input_tensor(3 * input_size * input_size);

    const int channel_size = input_size * input_size;

    for (int y = 0; y < input_size; ++y) {
        for (int x = 0; x < input_size; ++x) {
            const cv::Vec3f pixel = float_image.at<cv::Vec3f>(y, x);

            const int offset = y * input_size + x;

            input_tensor[0 * channel_size + offset] = pixel[0];

            input_tensor[1 * channel_size + offset] = pixel[1];

            input_tensor[2 * channel_size + offset] = pixel[2];
        }
    }

    return {std::move(input_tensor), letterbox_result.scale, letterbox_result.pad_left,
            letterbox_result.pad_top};
}

float intersection_over_union(const Detection &a, const Detection &b) {
    const float intersection_x1 = std::max(a.x1, b.x1);

    const float intersection_y1 = std::max(a.y1, b.y1);

    const float intersection_x2 = std::min(a.x2, b.x2);

    const float intersection_y2 = std::min(a.y2, b.y2);

    const float intersection_width = std::max(0.0F, intersection_x2 - intersection_x1);

    const float intersection_height = std::max(0.0F, intersection_y2 - intersection_y1);

    const float intersection_area = intersection_width * intersection_height;

    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);

    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);

    const float union_area = area_a + area_b - intersection_area;

    if (union_area <= 0.0F) {
        return 0.0F;
    }

    return intersection_area / union_area;
}

std::vector<Detection> nms(std::vector<Detection> detections, float iou_threshold) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b) { return a.score > b.score; });

    std::vector<Detection> kept;

    std::vector<bool> removed(detections.size(), false);

    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (removed[i]) {
            continue;
        }

        kept.push_back(detections[i]);

        for (std::size_t j = i + 1; j < detections.size(); ++j) {
            if (removed[j]) {
                continue;
            }

            if (detections[i].class_id != detections[j].class_id) {
                continue;
            }

            const float iou = intersection_over_union(detections[i], detections[j]);

            if (iou > iou_threshold) {
                removed[j] = true;
            }
        }
    }

    return kept;
}

std::vector<Detection> postprocess(const float *output_data, int num_classes, int num_predictions,
                                   int image_width, int image_height, float scale, int pad_left,
                                   int pad_top, float confidence_threshold, float iou_threshold) {
    std::vector<Detection> candidates;

    for (int i = 0; i < num_predictions; ++i) {
        int best_class_id = 0;

        float best_score = output_data[4 * num_predictions + i];

        for (int class_id = 1; class_id < num_classes; ++class_id) {
            const float score = output_data[(4 + class_id) * num_predictions + i];

            if (score > best_score) {
                best_score = score;

                best_class_id = class_id;
            }
        }

        if (best_score < confidence_threshold) {
            continue;
        }

        const float center_x = output_data[0 * num_predictions + i];

        const float center_y = output_data[1 * num_predictions + i];

        const float width = output_data[2 * num_predictions + i];

        const float height = output_data[3 * num_predictions + i];

        float x1 = center_x - width / 2.0F;

        float y1 = center_y - height / 2.0F;

        float x2 = center_x + width / 2.0F;

        float y2 = center_y + height / 2.0F;

        x1 = (x1 - pad_left) / scale;

        y1 = (y1 - pad_top) / scale;

        x2 = (x2 - pad_left) / scale;

        y2 = (y2 - pad_top) / scale;

        x1 = std::max(0.0F, std::min(x1, static_cast<float>(image_width - 1)));

        y1 = std::max(0.0F, std::min(y1, static_cast<float>(image_height - 1)));

        x2 = std::max(0.0F, std::min(x2, static_cast<float>(image_width - 1)));

        y2 = std::max(0.0F, std::min(y2, static_cast<float>(image_height - 1)));

        candidates.push_back({x1, y1, x2, y2, best_score, best_class_id});
    }

    return nms(std::move(candidates), iou_threshold);
}

struct Detector::Impl {
    DetectorOptions options;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "road_perception"};
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string input_name, output_name;
    std::vector<int64_t> input_shape;

    explicit Impl(const DetectorOptions &value) : options(value) {
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (!options.cpu) {
#ifdef ROAD_PERCEPTION_CUDA
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = options.device;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
            throw std::runtime_error(
                "Built without CUDA; use --cpu or rebuild with ROAD_PERCEPTION_CUDA=ON");
#endif
        }
        session = Ort::Session(env, std::filesystem::path(options.model).c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session.GetInputNameAllocated(0, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();
        input_shape = {1, 3, options.input_size, options.input_size};
        if (session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != input_shape)
            throw std::runtime_error("Model input shape does not match the configured input_size");
    }
};
Detector::Detector(const DetectorOptions &options) : impl_(std::make_unique<Impl>(options)) {}
Detector::~Detector() = default;
DetectionResult Detector::detect(const cv::Mat &image) {
    auto &impl = *impl_;
    const auto start = Clock::now();
    auto input = preprocess(image, impl.options.input_size);
    const auto preprocessed = Clock::now();
    auto tensor =
        Ort::Value::CreateTensor<float>(impl.memory, input.tensor.data(), input.tensor.size(),
                                        impl.input_shape.data(), impl.input_shape.size());
    const char *inputs[] = {impl.input_name.c_str()};
    const char *outputs[] = {impl.output_name.c_str()};
    const auto inference_start = Clock::now();
    auto result = impl.session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    const auto inferred = Clock::now();
    const auto shape = result[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 8 || shape[2] <= 0)
        throw std::runtime_error("Expected YOLO output [1, 8, N] for the four UA-DETRAC classes");
    auto detections = postprocess(result[0].GetTensorData<float>(), 4, static_cast<int>(shape[2]),
                                  image.cols, image.rows, input.scale, input.pad_left,
                                  input.pad_top, impl.options.confidence, impl.options.iou);
    const auto end = Clock::now();
    return {std::move(detections), elapsed_ms(start, preprocessed),
            elapsed_ms(inference_start, inferred), elapsed_ms(inferred, end)};
}
} // namespace road_perception
