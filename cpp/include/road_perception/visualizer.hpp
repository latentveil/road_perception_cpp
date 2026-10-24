#pragma once
#include "road_perception/types.hpp"
#include "STrack.h"
namespace road_perception {
void draw_tracks(cv::Mat &frame, const std::vector<STrack> &tracks,
                 const std::vector<std::string> &names);
}
