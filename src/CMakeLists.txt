project(ncnn-webassembly-yolov5)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE release)

set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=15")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=15")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=15")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -sEXPORTED_FUNCTIONS=['_yolo_ncnn'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sEXPORTED_FUNCTIONS=['_yolo_ncnn'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@.")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -sEXPORTED_FUNCTIONS=['_yolo_ncnn'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@.")


add_executable(yolo yolo.cpp)
target_link_libraries(yolo ncnn pthread)

#add_executable(yolov5 yolov5.cpp)
#target_link_libraries(yolov5 ncnn pthread)

# add_executable(yolov5up yolov5.cpp)
# target_link_libraries(yolov5up ncnn)
