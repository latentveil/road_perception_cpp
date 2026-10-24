#include "road_perception/pipeline.hpp"
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

int main(int argc, char *argv[]) {
    try {
        road_perception::PipelineOptions options;
        const std::set<std::string> allowed = {
            "--model", "--source", "--output",     "--input-size", "--device", "--conf",
            "--iou",   "--warmup", "--track-high", "--new-track",  "--match",  "--track-buffer"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "road_perception --model MODEL --source VIDEO --output VIDEO [--cpu]\n"
                          << "Use python infer.py to process all configured sequences.\n";
                return 0;
            }
            if (key == "--cpu") {
                options.detector.cpu = true;
                continue;
            }
            if (!allowed.count(key) || i + 1 >= argc)
                throw std::invalid_argument("Unknown or missing argument: " + key);
            args[key] = argv[++i];
        }
        for (const auto &key : {"--model", "--source", "--output"})
            if (!args.count(key))
                throw std::invalid_argument(std::string("Required: ") + key);
        options.detector.model = args.at("--model");
        options.source = args.at("--source");
        options.output = args.at("--output");
        const auto integer = [&](const char *key, int fallback) {
            return args.count(key) ? std::stoi(args.at(key)) : fallback;
        };
        const auto number = [&](const char *key, float fallback) {
            return args.count(key) ? std::stof(args.at(key)) : fallback;
        };
        options.detector.input_size = integer("--input-size", 640);
        options.detector.device = integer("--device", 0);
        options.detector.confidence = number("--conf", .10F);
        options.detector.iou = number("--iou", .70F);
        options.warmup_frames = integer("--warmup", 50);
        options.tracker.high_threshold = number("--track-high", .25F);
        options.tracker.new_threshold = number("--new-track", .25F);
        options.tracker.match_threshold = number("--match", .80F);
        options.tracker.buffer = integer("--track-buffer", 30);
        const auto probability = [](float x) { return std::isfinite(x) && x >= 0 && x <= 1; };
        if (options.detector.input_size <= 0 || options.detector.device < 0 ||
            options.warmup_frames < 0 || options.tracker.buffer <= 0 ||
            !probability(options.detector.confidence) || !probability(options.detector.iou) ||
            !probability(options.tracker.high_threshold) ||
            !probability(options.tracker.new_threshold) ||
            !probability(options.tracker.match_threshold) ||
            options.detector.confidence > options.tracker.high_threshold ||
            options.tracker.new_threshold < options.tracker.high_threshold)
            throw std::invalid_argument(
                "Invalid input size, timing settings, or detection/tracking thresholds");
        return road_perception::run_pipeline(options);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
