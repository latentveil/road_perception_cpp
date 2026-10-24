#include "road_perception/pipeline.hpp"
#include "road_perception/detector.hpp"
#include "road_perception/tracker.hpp"
#include "road_perception/visualizer.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace road_perception {
namespace {
double mean(const std::vector<double> &values) {
    return values.empty() ? 0 : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}
double percentile(std::vector<double> values, double p) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(p * values.size())) - 1];
}
} // namespace
int run_pipeline(const PipelineOptions &options) {
    namespace fs = std::filesystem;
    if (fs::weakly_canonical(options.source) == fs::weakly_canonical(options.output))
        throw std::invalid_argument("Output must differ from the input video");
    Detector detector(options.detector);
    cv::VideoCapture capture(options.source);
    if (!capture.isOpened())
        throw std::runtime_error("Cannot open video: " + options.source);
    const double fps = capture.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || !std::isfinite(fps))
        throw std::runtime_error("Input video has no valid FPS");
    const cv::Size size(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    const fs::path output(options.output);
    if (output.has_parent_path())
        fs::create_directories(output.parent_path());
    cv::VideoWriter writer(options.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
    if (!writer.isOpened())
        throw std::runtime_error("Cannot write video: " + options.output);
    Tracker tracker(fps, options.tracker);
    auto tracks_path = output;
    tracks_path.replace_extension(".tracks.csv");
    auto timing_path = output;
    timing_path.replace_extension(".timing.csv");
    std::ofstream tracks_file(tracks_path), timing_file(timing_path);
    if (!tracks_file || !timing_file)
        throw std::runtime_error("Cannot open CSV outputs");
    tracks_file << "frame,id,class_id,score,x1,y1,x2,y2\n" << std::setprecision(9);
    timing_file
        << "frame,preprocess_ms,inference_ms,postprocess_ms,tracking_ms,pipeline_ms,warmup\n";
    std::vector<double> preprocessing, inference, postprocessing, tracking, pipeline;
    const auto video_start = Clock::now();
    cv::Mat frame;
    int frame_index = 0;
    std::cout << "backend requested: " << (options.detector.cpu ? "CPU" : "CUDA") << '\n';
    while (capture.read(frame)) {
        ++frame_index;
        const auto start = Clock::now();
        auto detected = detector.detect(frame);
        const auto tracking_start = Clock::now();
        auto tracks = tracker.update(detected.detections);
        const auto tracking_end = Clock::now();
        const double tracking_ms = elapsed_ms(tracking_start, tracking_end);
        const double pipeline_ms = elapsed_ms(start, tracking_end);
        if (frame_index > options.warmup_frames) {
            preprocessing.push_back(detected.preprocess_ms);
            inference.push_back(detected.inference_ms);
            postprocessing.push_back(detected.postprocess_ms);
            tracking.push_back(tracking_ms);
            pipeline.push_back(pipeline_ms);
        }
        timing_file << frame_index << ',' << detected.preprocess_ms << ',' << detected.inference_ms
                    << ',' << detected.postprocess_ms << ',' << tracking_ms << ',' << pipeline_ms
                    << ',' << (frame_index <= options.warmup_frames ? 1 : 0) << '\n';
        for (const auto &track : tracks) {
            if (track.tlbr.size() != 4)
                continue;
            tracks_file << frame_index << ',' << track.track_id << ',' << track.class_id << ','
                        << track.score;
            for (float value : track.tlbr)
                tracks_file << ',' << value;
            tracks_file << '\n';
        }
        draw_tracks(frame, tracks, options.names);
        writer.write(frame);
        if (frame_index % 100 == 0)
            std::cout << "frame=" << frame_index << ", tracks=" << tracks.size() << '\n';
    }
    capture.release();
    writer.release();
    tracks_file.close();
    timing_file.close();
    const double video_ms = elapsed_ms(video_start, Clock::now());
    std::cout << std::fixed << std::setprecision(3) << "processed frames: " << frame_index << '\n'
              << "benchmark frames: " << pipeline.size() << '\n'
              << "preprocess avg ms: " << mean(preprocessing) << '\n'
              << "inference avg ms: " << mean(inference) << '\n'
              << "postprocess avg ms: " << mean(postprocessing) << '\n'
              << "tracking avg ms: " << mean(tracking) << '\n'
              << "pipeline avg ms: " << mean(pipeline) << '\n'
              << "pipeline p50 ms: " << percentile(pipeline, .5) << '\n'
              << "pipeline p95 ms: " << percentile(pipeline, .95) << '\n'
              << "pipeline fps (excluding video I/O and drawing): "
              << (mean(pipeline) > 0 ? 1000 / mean(pipeline) : 0) << '\n'
              << "video loop fps (including I/O, drawing and CSV): "
              << (video_ms > 0 ? 1000 * frame_index / video_ms : 0) << '\n'
              << "output: " << options.output << '\n';
    return frame_index > 0 ? 0 : 1;
}
} // namespace road_perception
