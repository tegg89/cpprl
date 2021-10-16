cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)

add_executable(example example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 14)