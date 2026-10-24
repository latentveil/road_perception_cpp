#include "road_perception/tracker.hpp"
#include <cmath>
namespace road_perception {
std::vector<Object> make_tracker_objects(const std::vector<Detection> &detections) {
    std::vector<Object> objects;

    objects.reserve(detections.size());

    for (const Detection &detection : detections) {
        const float width = detection.x2 - detection.x1;

        const float height = detection.y2 - detection.y1;

        if (width <= 0.0F || height <= 0.0F) {
            continue;
        }

        Object object;

        object.rect = cv::Rect_<float>(detection.x1, detection.y1, width, height);

        object.label = detection.class_id;

        object.prob = detection.score;

        objects.push_back(object);
    }

    return objects;
}

Tracker::Tracker(double fps, const TrackerOptions &options)
    : tracker_(static_cast<int>(std::round(fps)), options.buffer, options.high_threshold,
               options.new_threshold, options.match_threshold) {}
std::vector<STrack> Tracker::update(const std::vector<Detection> &detections) {
    return tracker_.update(make_tracker_objects(detections));
}
} // namespace road_perception
