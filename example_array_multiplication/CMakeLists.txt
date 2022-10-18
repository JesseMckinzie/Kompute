cmake_minimum_required(VERSION 3.11.0)
project(kompute_array_mult VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

# Include main Kompute directory
add_subdirectory(${PROJECT_SOURCE_DIR}/../kompute/ ${CMAKE_CURRENT_BINARY_DIR}/kompute)

add_executable(kompute_array_mult
    src/Main.cpp)

target_link_libraries(kompute_array_mult
    kompute::kompute
)