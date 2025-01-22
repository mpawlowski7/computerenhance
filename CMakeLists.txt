cmake_minimum_required(VERSION 3.30)
project(ComputerEnhance)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 COMPONENTS
        Core
        Gui
        Quick
        REQUIRED)

qt_standard_project_setup(REQUIRES 6.6.2)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)

add_subdirectory(prebuilts/llama-cpp)
add_subdirectory(src)