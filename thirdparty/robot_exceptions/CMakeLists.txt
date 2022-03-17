cmake_minimum_required(VERSION 2.8.3)

project(robot_exceptions)

add_compile_options(-std=c++11)
include_directories(include)

#同时编译静态库和动态库
add_library(robot_exceptions SHARED 
            src/robot_exceptions.cpp)
add_library(robot_exceptions_static STATIC 
            src/robot_exceptions.cpp)
