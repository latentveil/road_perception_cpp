#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "BYTETracker.h"


using Clock = std::chrono::steady_clock;


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
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};


struct BenchmarkRecord {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double tracking_ms;
    double pipeline_ms;
};


double elapsed_ms(
    const Clock::time_point& start,
    const Clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}


double mean(
    const std::vector<double>& values
) {
    if (values.empty()) {
        return 0.0;
    }

    const double sum = std::accumulate(
        values.begin(),
        values.end(),
        0.0
    );

    return sum /
        static_cast<double>(values.size());
}


double percentile(
    std::vector<double> values,
    double percentile_value
) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    const double rank =
        percentile_value *
        static_cast<double>(values.size());

    std::size_t index =
        static_cast<std::size_t>(
            std::ceil(rank)
        );

    if (index == 0) {
        index = 1;
    }

    if (index > values.size()) {
        index = values.size();
    }

    return values[index - 1];
}


LetterboxResult letterbox(
    const cv::Mat& image,
    int input_size
) {
    const int original_width = image.cols;
    const int original_height = image.rows;

    const float scale = std::min(
        static_cast<float>(input_size) /
            original_width,
        static_cast<float>(input_size) /
            original_height
    );

    const int resized_width =
        static_cast<int>(
            std::round(
                original_width * scale
            )
        );

    const int resized_height =
        static_cast<int>(
            std::round(
                original_height * scale
            )
        );

    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(
            resized_width,
            resized_height
        ),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    const int pad_width =
        input_size - resized_width;

    const int pad_height =
        input_size - resized_height;

    const int left =
        static_cast<int>(
            std::round(
                pad_width / 2.0 - 0.1
            )
        );

    const int right =
        static_cast<int>(
            std::round(
                pad_width / 2.0 + 0.1
            )
        );

    const int top =
        static_cast<int>(
            std::round(
                pad_height / 2.0 - 0.1
            )
        );

    const int bottom =
        static_cast<int>(
            std::round(
                pad_height / 2.0 + 0.1
            )
        );

    cv::Mat padded;

    cv::copyMakeBorder(
        resized,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(
            114,
            114,
            114
        )
    );

    return {
        padded,
        scale,
        left,
        top
    };
}


PreprocessResult preprocess(
    const cv::Mat& image,
    int input_size
) {
    const LetterboxResult letterbox_result =
        letterbox(
            image,
            input_size
        );

    cv::Mat rgb;

    cv::cvtColor(
        letterbox_result.image,
        rgb,
        cv::COLOR_BGR2RGB
    );

    cv::Mat float_image;

    rgb.convertTo(
        float_image,
        CV_32FC3,
        1.0 / 255.0
    );

    std::vector<float> input_tensor(
        3 *
        input_size *
        input_size
    );

    const int channel_size =
        input_size *
        input_size;

    for (
        int y = 0;
        y < input_size;
        ++y
    ) {
        for (
            int x = 0;
            x < input_size;
            ++x
        ) {
            const cv::Vec3f pixel =
                float_image.at<cv::Vec3f>(
                    y,
                    x
                );

            const int offset =
                y *
                input_size +
                x;

            input_tensor[
                0 * channel_size +
                offset
            ] = pixel[0];

            input_tensor[
                1 * channel_size +
                offset
            ] = pixel[1];

            input_tensor[
                2 * channel_size +
                offset
            ] = pixel[2];
        }
    }

    return {
        std::move(input_tensor),
        letterbox_result.scale,
        letterbox_result.pad_left,
        letterbox_result.pad_top
    };
}


float intersection_over_union(
    const Detection& a,
    const Detection& b
) {
    const float intersection_x1 =
        std::max(
            a.x1,
            b.x1
        );

    const float intersection_y1 =
        std::max(
            a.y1,
            b.y1
        );

    const float intersection_x2 =
        std::min(
            a.x2,
            b.x2
        );

    const float intersection_y2 =
        std::min(
            a.y2,
            b.y2
        );

    const float intersection_width =
        std::max(
            0.0F,
            intersection_x2 -
                intersection_x1
        );

    const float intersection_height =
        std::max(
            0.0F,
            intersection_y2 -
                intersection_y1
        );

    const float intersection_area =
        intersection_width *
        intersection_height;

    const float area_a =
        std::max(
            0.0F,
            a.x2 - a.x1
        ) *
        std::max(
            0.0F,
            a.y2 - a.y1
        );

    const float area_b =
        std::max(
            0.0F,
            b.x2 - b.x1
        ) *
        std::max(
            0.0F,
            b.y2 - b.y1
        );

    const float union_area =
        area_a +
        area_b -
        intersection_area;

    if (union_area <= 0.0F) {
        return 0.0F;
    }

    return
        intersection_area /
        union_area;
}


std::vector<Detection> nms(
    std::vector<Detection> detections,
    float iou_threshold
) {
    std::sort(
        detections.begin(),
        detections.end(),
        [](
            const Detection& a,
            const Detection& b
        ) {
            return
                a.score >
                b.score;
        }
    );

    std::vector<Detection> kept;

    std::vector<bool> removed(
        detections.size(),
        false
    );

    for (
        std::size_t i = 0;
        i < detections.size();
        ++i
    ) {
        if (removed[i]) {
            continue;
        }

        kept.push_back(
            detections[i]
        );

        for (
            std::size_t j =
                i + 1;
            j < detections.size();
            ++j
        ) {
            if (removed[j]) {
                continue;
            }

            if (
                detections[i]
                    .class_id !=
                detections[j]
                    .class_id
            ) {
                continue;
            }

            const float iou =
                intersection_over_union(
                    detections[i],
                    detections[j]
                );

            if (
                iou >
                iou_threshold
            ) {
                removed[j] = true;
            }
        }
    }

    return kept;
}


std::vector<Detection> postprocess(
    const float* output_data,
    int num_classes,
    int num_predictions,
    int image_width,
    int image_height,
    float scale,
    int pad_left,
    int pad_top,
    float confidence_threshold,
    float iou_threshold
) {
    std::vector<Detection> candidates;

    for (
        int i = 0;
        i < num_predictions;
        ++i
    ) {
        int best_class_id = 0;

        float best_score =
            output_data[
                4 *
                num_predictions +
                i
            ];

        for (
            int class_id = 1;
            class_id <
                num_classes;
            ++class_id
        ) {
            const float score =
                output_data[
                    (4 + class_id) *
                        num_predictions +
                    i
                ];

            if (
                score >
                best_score
            ) {
                best_score =
                    score;

                best_class_id =
                    class_id;
            }
        }

        if (
            best_score <
            confidence_threshold
        ) {
            continue;
        }

        const float center_x =
            output_data[
                0 *
                num_predictions +
                i
            ];

        const float center_y =
            output_data[
                1 *
                num_predictions +
                i
            ];

        const float width =
            output_data[
                2 *
                num_predictions +
                i
            ];

        const float height =
            output_data[
                3 *
                num_predictions +
                i
            ];

        float x1 =
            center_x -
            width / 2.0F;

        float y1 =
            center_y -
            height / 2.0F;

        float x2 =
            center_x +
            width / 2.0F;

        float y2 =
            center_y +
            height / 2.0F;

        x1 =
            (x1 - pad_left) /
            scale;

        y1 =
            (y1 - pad_top) /
            scale;

        x2 =
            (x2 - pad_left) /
            scale;

        y2 =
            (y2 - pad_top) /
            scale;

        x1 = std::max(
            0.0F,
            std::min(
                x1,
                static_cast<float>(
                    image_width - 1
                )
            )
        );

        y1 = std::max(
            0.0F,
            std::min(
                y1,
                static_cast<float>(
                    image_height - 1
                )
            )
        );

        x2 = std::max(
            0.0F,
            std::min(
                x2,
                static_cast<float>(
                    image_width - 1
                )
            )
        );

        y2 = std::max(
            0.0F,
            std::min(
                y2,
                static_cast<float>(
                    image_height - 1
                )
            )
        );

        candidates.push_back({
            x1,
            y1,
            x2,
            y2,
            best_score,
            best_class_id
        });
    }

    return nms(
        std::move(candidates),
        iou_threshold
    );
}


std::vector<Object> make_tracker_objects(
    const std::vector<Detection>& detections
) {
    std::vector<Object> objects;

    objects.reserve(
        detections.size()
    );

    for (
        const Detection& detection :
        detections
    ) {
        const float width =
            detection.x2 -
            detection.x1;

        const float height =
            detection.y2 -
            detection.y1;

        if (
            width <= 0.0F ||
            height <= 0.0F
        ) {
            continue;
        }

        Object object;

        object.rect =
            cv::Rect_<float>(
                detection.x1,
                detection.y1,
                width,
                height
            );

        object.label =
            detection.class_id;

        object.prob =
            detection.score;

        objects.push_back(
            object
        );
    }

    return objects;
}


void draw_tracks(
    cv::Mat& frame,
    const std::vector<STrack>& tracks,
    const std::array<std::string, 4>& class_names
) {
    for (
        const STrack& track :
        tracks
    ) {
        if (
            track.tlbr.size() != 4
        ) {
            continue;
        }

        const float x1 =
            std::max(
                0.0F,
                std::min(
                    track.tlbr[0],
                    static_cast<float>(
                        frame.cols - 1
                    )
                )
            );

        const float y1 =
            std::max(
                0.0F,
                std::min(
                    track.tlbr[1],
                    static_cast<float>(
                        frame.rows - 1
                    )
                )
            );

        const float x2 =
            std::max(
                0.0F,
                std::min(
                    track.tlbr[2],
                    static_cast<float>(
                        frame.cols - 1
                    )
                )
            );

        const float y2 =
            std::max(
                0.0F,
                std::min(
                    track.tlbr[3],
                    static_cast<float>(
                        frame.rows - 1
                    )
                )
            );

        const cv::Point top_left(
            static_cast<int>(
                std::round(x1)
            ),
            static_cast<int>(
                std::round(y1)
            )
        );

        const cv::Point bottom_right(
            static_cast<int>(
                std::round(x2)
            ),
            static_cast<int>(
                std::round(y2)
            )
        );

        cv::rectangle(
            frame,
            top_left,
            bottom_right,
            cv::Scalar(
                0,
                255,
                0
            ),
            2
        );

        std::ostringstream label_stream;

        label_stream
            << "ID "
            << track.track_id
            << " ";

        if (
            track.class_id >= 0 &&
            track.class_id <
                static_cast<int>(
                    class_names.size()
                )
        ) {
            label_stream
                << class_names[
                    track.class_id
                ];
        } else {
            label_stream
                << track.class_id;
        }

        label_stream
            << " "
            << std::fixed
            << std::setprecision(2)
            << track.score;

        cv::putText(
            frame,
            label_stream.str(),
            cv::Point(
                top_left.x,
                std::max(
                    top_left.y - 8,
                    20
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(
                0,
                255,
                0
            ),
            2,
            cv::LINE_AA
        );
    }
}


int main(
    int argc,
    char* argv[]
) {
    const std::string model_path =
        argc > 1
            ? argv[1]
            : "models/"
              "yolov8s_ua_detrac_best.onnx";

    const std::string video_path =
        argc > 2
            ? argv[2]
            : "videos/"
              "MVI_20012.mp4";

    const std::string output_path =
        "results/tracking/"
        "MVI_20012_cpp_bytetrack.mp4";

    constexpr int input_size = 640;
    constexpr int num_classes = 4;
    constexpr int num_predictions = 8400;
    constexpr int warmup_frames = 50;

    constexpr float confidence_threshold =
        0.10F;

    constexpr float iou_threshold =
        0.70F;

    const std::array<
        std::string,
        num_classes
    > class_names = {
        "car",
        "bus",
        "van",
        "others"
    };

    try {
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "road_perception"
        );

        Ort::SessionOptions
            session_options;

        session_options
            .SetGraphOptimizationLevel(
                GraphOptimizationLevel::
                    ORT_ENABLE_ALL
            );

        OrtCUDAProviderOptions
            cuda_options;

        cuda_options.device_id = 0;

        session_options
            .AppendExecutionProvider_CUDA(
                cuda_options
            );

        Ort::Session session(
            env,
            model_path.c_str(),
            session_options
        );

        Ort::AllocatorWithDefaultOptions
            allocator;

        auto input_name =
            session.GetInputNameAllocated(
                0,
                allocator
            );

        auto output_name =
            session.GetOutputNameAllocated(
                0,
                allocator
            );

        const char* input_names[] = {
            input_name.get()
        };

        const char* output_names[] = {
            output_name.get()
        };

        const std::vector<int64_t>
            input_shape = {
                1,
                3,
                input_size,
                input_size
            };

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        cv::VideoCapture capture(
            video_path
        );

        if (!capture.isOpened()) {
            std::cerr
                << "Failed to open video: "
                << video_path
                << '\n';

            return 1;
        }

        const double fps =
            capture.get(
                cv::CAP_PROP_FPS
            );

        const int width =
            static_cast<int>(
                capture.get(
                    cv::CAP_PROP_FRAME_WIDTH
                )
            );

        const int height =
            static_cast<int>(
                capture.get(
                    cv::CAP_PROP_FRAME_HEIGHT
                )
            );

        const int frame_count =
            static_cast<int>(
                capture.get(
                    cv::CAP_PROP_FRAME_COUNT
                )
            );

        const std::filesystem::path output_file(output_path);

        if (output_file.has_parent_path()) {
            std::filesystem::create_directories(
                output_file.parent_path()
            );
        }

        cv::VideoWriter writer(
            output_path,
            cv::VideoWriter::fourcc(
                'm',
                'p',
                '4',
                'v'
            ),
            fps,
            cv::Size(
                width,
                height
            )
        );

        if (!writer.isOpened()) {
            std::cerr
                << "Failed to open video writer: "
                << output_path
                << '\n';

            return 1;
        }

        BYTETracker tracker(
            30,
            30
        );

        std::vector<BenchmarkRecord>
            benchmark_records;

        benchmark_records.reserve(
            std::max(
                0,
                frame_count -
                    warmup_frames
            )
        );

        std::cout
            << "execution provider: CUDA"
            << '\n';

        std::cout
            << "tracker: ByteTrack"
            << '\n';

        std::cout
            << "model input: "
            << input_size
            << "x"
            << input_size
            << '\n';

        std::cout
            << "warmup frames: "
            << warmup_frames
            << '\n';

        std::cout
            << "video: "
            << video_path
            << '\n';

        std::cout
            << "frames: "
            << frame_count
            << '\n';

        std::cout
            << "resolution: "
            << width
            << "x"
            << height
            << '\n';

        cv::Mat frame;

        int frame_index = 0;

        while (
            capture.read(frame)
        ) {
            ++frame_index;

            const auto pipeline_start =
                Clock::now();

            const auto preprocess_start =
                pipeline_start;

            PreprocessResult
                preprocess_result =
                    preprocess(
                        frame,
                        input_size
                    );

            const auto preprocess_end =
                Clock::now();

            Ort::Value input_tensor =
                Ort::Value::
                    CreateTensor<float>(
                        memory_info,
                        preprocess_result
                            .tensor
                            .data(),
                        preprocess_result
                            .tensor
                            .size(),
                        input_shape.data(),
                        input_shape.size()
                    );

            const auto inference_start =
                Clock::now();

            std::vector<Ort::Value>
                outputs =
                    session.Run(
                        Ort::RunOptions{
                            nullptr
                        },
                        input_names,
                        &input_tensor,
                        1,
                        output_names,
                        1
                    );

            const auto inference_end =
                Clock::now();

            const float* output_data =
                outputs[0]
                    .GetTensorData<float>();

            const auto postprocess_start =
                Clock::now();

            std::vector<Detection>
                detections =
                    postprocess(
                        output_data,
                        num_classes,
                        num_predictions,
                        frame.cols,
                        frame.rows,
                        preprocess_result
                            .scale,
                        preprocess_result
                            .pad_left,
                        preprocess_result
                            .pad_top,
                        confidence_threshold,
                        iou_threshold
                    );

            std::vector<Object>
                tracker_objects =
                    make_tracker_objects(
                        detections
                    );

            const auto postprocess_end =
                Clock::now();

            const auto tracking_start =
                Clock::now();

            std::vector<STrack>
                tracks =
                    tracker.update(
                        tracker_objects
                    );

            const auto tracking_end =
                Clock::now();

            const auto pipeline_end =
                tracking_end;

            if (
                frame_index >
                warmup_frames
            ) {
                benchmark_records.push_back({
                    elapsed_ms(
                        preprocess_start,
                        preprocess_end
                    ),
                    elapsed_ms(
                        inference_start,
                        inference_end
                    ),
                    elapsed_ms(
                        postprocess_start,
                        postprocess_end
                    ),
                    elapsed_ms(
                        tracking_start,
                        tracking_end
                    ),
                    elapsed_ms(
                        pipeline_start,
                        pipeline_end
                    )
                });
            }

            draw_tracks(
                frame,
                tracks,
                class_names
            );

            writer.write(frame);

            if (
                frame_index % 100 == 0 ||
                frame_index == frame_count
            ) {
                std::cout
                    << "frame "
                    << frame_index
                    << "/"
                    << frame_count
                    << ", detections="
                    << detections.size()
                    << ", tracks="
                    << tracks.size()
                    << '\n';
            }
        }

        capture.release();
        writer.release();

        std::vector<double>
            preprocess_times;

        std::vector<double>
            inference_times;

        std::vector<double>
            postprocess_times;

        std::vector<double>
            tracking_times;

        std::vector<double>
            pipeline_times;

        for (
            const BenchmarkRecord& record :
            benchmark_records
        ) {
            preprocess_times.push_back(
                record.preprocess_ms
            );

            inference_times.push_back(
                record.inference_ms
            );

            postprocess_times.push_back(
                record.postprocess_ms
            );

            tracking_times.push_back(
                record.tracking_ms
            );

            pipeline_times.push_back(
                record.pipeline_ms
            );
        }

        const double pipeline_avg =
            mean(
                pipeline_times
            );

        const double pipeline_fps =
            pipeline_avg > 0.0
                ? 1000.0 /
                    pipeline_avg
                : 0.0;

        std::cout << '\n';

        std::cout
            << "finished"
            << '\n';

        std::cout
            << "processed frames: "
            << frame_index
            << '\n';

        std::cout
            << "output: "
            << output_path
            << '\n';

        std::cout << '\n';

        std::cout
            << "===== Benchmark ====="
            << '\n';

        std::cout
            << "benchmark frames: "
            << benchmark_records.size()
            << '\n';

        std::cout
            << "input resolution: "
            << input_size
            << "x"
            << input_size
            << '\n';

        std::cout
            << std::fixed
            << std::setprecision(3);

        std::cout
            << "preprocess avg ms: "
            << mean(
                preprocess_times
            )
            << '\n';

        std::cout
            << "inference avg ms: "
            << mean(
                inference_times
            )
            << '\n';

        std::cout
            << "postprocess avg ms: "
            << mean(
                postprocess_times
            )
            << '\n';

        std::cout
            << "tracking avg ms: "
            << mean(
                tracking_times
            )
            << '\n';

        std::cout
            << "pipeline avg ms: "
            << pipeline_avg
            << '\n';

        std::cout
            << "pipeline p50 ms: "
            << percentile(
                pipeline_times,
                0.50
            )
            << '\n';

        std::cout
            << "pipeline p95 ms: "
            << percentile(
                pipeline_times,
                0.95
            )
            << '\n';

        std::cout
            << "pipeline fps: "
            << pipeline_fps
            << '\n';
    } catch (
        const Ort::Exception& error
    ) {
        std::cerr
            << "ONNX Runtime error: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}
