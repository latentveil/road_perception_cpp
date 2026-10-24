#include "road_perception/visualizer.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
namespace road_perception {
void draw_tracks(cv::Mat &frame, const std::vector<STrack> &tracks,
                 const std::vector<std::string> &class_names) {
    for (const STrack &track : tracks) {
        if (track.tlbr.size() != 4) {
            continue;
        }

        const float x1 =
            std::max(0.0F, std::min(track.tlbr[0], static_cast<float>(frame.cols - 1)));

        const float y1 =
            std::max(0.0F, std::min(track.tlbr[1], static_cast<float>(frame.rows - 1)));

        const float x2 =
            std::max(0.0F, std::min(track.tlbr[2], static_cast<float>(frame.cols - 1)));

        const float y2 =
            std::max(0.0F, std::min(track.tlbr[3], static_cast<float>(frame.rows - 1)));

        const cv::Point top_left(static_cast<int>(std::round(x1)),
                                 static_cast<int>(std::round(y1)));

        const cv::Point bottom_right(static_cast<int>(std::round(x2)),
                                     static_cast<int>(std::round(y2)));

        cv::rectangle(frame, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);

        std::ostringstream label_stream;

        label_stream << "ID " << track.track_id << " ";

        if (track.class_id >= 0 && track.class_id < static_cast<int>(class_names.size())) {
            label_stream << class_names[track.class_id];
        } else {
            label_stream << track.class_id;
        }

        label_stream << " " << std::fixed << std::setprecision(2) << track.score;

        cv::putText(frame, label_stream.str(), cv::Point(top_left.x, std::max(top_left.y - 8, 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}

} // namespace road_perception
