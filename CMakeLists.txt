cmake_minimum_required(VERSION 3.16)

project(road_perception_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(
    ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-gpu-1.19.2"
)

add_executable(
    road_perception

    src/main.cpp

    src/bytetrack/BYTETracker.cpp
    src/bytetrack/STrack.cpp
    src/bytetrack/kalmanFilter.cpp
    src/bytetrack/lapjv.cpp
    src/bytetrack/utils.cpp
)

target_include_directories(
    road_perception
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${ONNXRUNTIME_ROOT}/include"
    "${CMAKE_SOURCE_DIR}/include/bytetrack"
)

target_link_directories(
    road_perception
    PRIVATE
    "${ONNXRUNTIME_ROOT}/lib"
)

target_link_libraries(
    road_perception
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    onnxruntime
)

if(
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9
)
    target_link_libraries(
        road_perception
        PRIVATE
        stdc++fs
    )
endif()

set_target_properties(
    road_perception
    PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)