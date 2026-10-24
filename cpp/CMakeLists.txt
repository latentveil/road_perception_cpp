find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
option(ROAD_PERCEPTION_CUDA "Enable ONNX Runtime CUDA provider" ON)
get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(ONNXRUNTIME_ROOT "${REPOSITORY_ROOT}/third_party/onnxruntime-linux-x64-gpu-1.19.2" CACHE PATH "ONNX Runtime distribution root")
# Always resolve from the selected root, including after a cached path is corrected.
unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
unset(ONNXRUNTIME_LIBRARY CACHE)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime was not found under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}. "
        "The real distribution must contain include/onnxruntime_cxx_api.h and lib/onnxruntime. "
        "Reconfigure with -DONNXRUNTIME_ROOT pointing to that directory; /path/to/... is only a placeholder.")
endif()
set(BYTETRACK_ROOT "${REPOSITORY_ROOT}/third_party/bytetrack")
add_library(bytetrack STATIC
    "${BYTETRACK_ROOT}/src/BYTETracker.cpp"
    "${BYTETRACK_ROOT}/src/STrack.cpp"
    "${BYTETRACK_ROOT}/src/kalmanFilter.cpp"
    "${BYTETRACK_ROOT}/src/lapjv.cpp"
    "${BYTETRACK_ROOT}/src/utils.cpp")
target_include_directories(bytetrack PUBLIC "${BYTETRACK_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bytetrack PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
add_library(road_perception_core STATIC src/detector.cpp src/tracker.cpp src/visualizer.cpp src/pipeline.cpp)
target_include_directories(road_perception_core PUBLIC include PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(road_perception_core PUBLIC bytetrack PRIVATE "${ONNXRUNTIME_LIBRARY}")
if(ROAD_PERCEPTION_CUDA)
    target_compile_definitions(road_perception_core PRIVATE ROAD_PERCEPTION_CUDA)
endif()
add_executable(road_perception src/main.cpp)
target_link_libraries(road_perception PRIVATE road_perception_core)
set_target_properties(road_perception PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib" INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(road_perception_core PUBLIC stdc++fs)
endif()
