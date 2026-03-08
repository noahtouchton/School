cmake_minimum_required(VERSION 3.10)
project(Clue)

# 1. Tell CMake where to find your .h files
include_directories(include)

# 2. Automatically grab every .cpp file inside the src folder
file(GLOB SOURCES "src/*.cpp")

# 3. Build the game using all of those files together
add_executable(Clue ${SOURCES})