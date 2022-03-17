# @file CMakeLists.txt
# @author zhangmeng (zhangmeng@shenhaoinfo.com)
# @brief subway project cmakelists
# @version 1.0
# @date 2022-03-10
# @copyright Copyright (c) 2022

cmake_minimum_required(VERSION "3.0")
project(matchalgorithem LANGUAGES CXX CUDA)
add_definitions(-D GLOG_NO_ABBREVIATED_SEVERITIES)

set(CMAKE_CUDA_STANDARD 14)

string(REGEX REPLACE "(.+)/match_algorithem.*" "\\1" PROJECT_INIT_PATH ${PROJECT_SOURCE_DIR})
message("-- PROJECT_SOURCE_DIR path in ${PROJECT_SOURCE_DIR}")
set(THIRDPARTY_PATH  ${PROJECT_SOURCE_DIR}/thirdparty)
message("-- Thirdparty path in ${THIRDPARTY_PATH}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")

find_package(CUDA REQUIRED) 
find_package(OpenCV REQUIRED)

# dll project
include_directories( matchalgorithem
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${THIRDPARTY_PATH}/glog/include
  ${THIRDPARTY_PATH}/gflags/include
  ${THIRDPARTY_PATH}/json/include
  ${THIRDPARTY_PATH}/robot_exceptions/include
  ${THIRDPARTY_PATH}/MIPP
  ${THIRDPARTY_PATH}/MIPP/math
)

link_directories( matchalgorithem
  ${THIRDPARTY_PATH}/glog/lib
)

aux_source_directory( ${PROJECT_SOURCE_DIR}/src SRC_FILES)
add_library( matchalgorithem SHARED
${SRC_FILES}
 )

target_link_libraries( matchalgorithem
  ${OpenCV_LIBS}
  ${CUDA_LIBS}
  glog.lib
)

# test project
include_directories( matchalgorithemtest
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
)

add_executable( matchalgorithemtest 
  ${PROJECT_SOURCE_DIR}/test/test.cc
)

target_link_libraries( matchalgorithemtest
  matchalgorithem
)

file(COPY
  ${THIRDPARTY_PATH}/opencv/x64/vc14/bin/opencv_world341.dll
  ${THIRDPARTY_PATH}/glog/bin/glog.dll
  ${THIRDPARTY_PATH}/gflags/bin/gflags.dll
  DESTINATION
  ${PROJECT_SOURCE_DIR}/lib
  FOLLOW_SYMLINK_CHAIN
)