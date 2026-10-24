#pragma once
#include <opencv2/core.hpp>
#include <chrono>
#include <string>
#include <vector>

namespace road_perception {
using Clock = std::chrono::steady_clock;
inline double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
struct LetterboxResult {
    cv::Mat image;
    float scale;
    int pad_left;
    int pad_top;
};
struct PreprocessResult {
    std::vector<float> tensor;
    float scale;
    int pad_left;
    int pad_top;
};
struct Detection {
    float x1, y1, x2, y2, score;
    int class_id;
};
struct DetectionResult {
    std::vector<Detection> detections;
    double preprocess_ms, inference_ms, postprocess_ms;
};
struct DetectorOptions {
    std::string model;
    int input_size = 640;
    int device = 0;
    bool cpu = false;
    float confidence = 0.10F;
    float iou = 0.70F;
};
struct TrackerOptions {
    float high_threshold = 0.25F;
    float new_threshold = 0.25F;
    float match_threshold = 0.80F;
    int buffer = 30;
};
struct PipelineOptions {
    DetectorOptions detector;
    TrackerOptions tracker;
    std::string source, output;
    int warmup_frames = 50;
    std::vector<std::string> names = {"car", "bus", "van", "others"};
};
} // namespace road_perception
