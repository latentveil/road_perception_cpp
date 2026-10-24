#pragma once
#include "road_perception/types.hpp"
#include <memory>
namespace road_perception {
LetterboxResult letterbox(const cv::Mat &image, int input_size);
PreprocessResult preprocess(const cv::Mat &image, int input_size);
std::vector<Detection> postprocess(const float *data, int classes, int predictions, int width,
                                   int height, float scale, int left, int top, float confidence,
                                   float iou);
class Detector {
  public:
    explicit Detector(const DetectorOptions &options);
    ~Detector();
    DetectionResult detect(const cv::Mat &image);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace road_perception
