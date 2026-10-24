#pragma once
#include "road_perception/types.hpp"
#include "BYTETracker.h"
namespace road_perception {
std::vector<Object> make_tracker_objects(const std::vector<Detection> &detections);
class Tracker {
  public:
    Tracker(double fps, const TrackerOptions &options);
    std::vector<STrack> update(const std::vector<Detection> &detections);

  private:
    BYTETracker tracker_;
};
} // namespace road_perception
