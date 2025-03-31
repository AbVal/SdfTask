cmake_minimum_required(VERSION 3.10)

# Set the project name
project(SdfTask CXX)

# Find the SDL2 library
find_package(SDL2 REQUIRED)
# Find OpenMP
find_package(OpenMP REQUIRED)

# Include SDL2 headers
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/LiteMath/external)
include_directories(${SDL2_INCLUDE_DIRS})

add_compile_definitions(USE_STB_IMAGE)

# Add the executable
add_executable(render
    main.cpp
    mesh.cpp
    LiteMath/Image2d.cpp)

# Link the SDL2 library to the executable
target_link_libraries(render ${SDL2_LIBRARIES} OpenMP::OpenMP_CXX)

# Set path to executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
