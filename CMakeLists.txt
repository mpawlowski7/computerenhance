cmake_minimum_required(VERSION 3.30)

project(ComputerEnhance)

set(CMAKE_CXX_STANDARD 17)

find_package(Qt6 COMPONENTS
        Core
        Gui
        Quick
        Qml
        Multimedia
        REQUIRED)

qt_standard_project_setup(REQUIRES 6.6.2)


# Specify the directory containing .jpg files
set(IMG_LIST)
set(IMAGE_DIR "images")
file(GLOB IMG_FILES "${IMAGE_DIR}/*")

# Convert to relative paths to be used with Qt resource file
if(IMG_FILES)
    foreach(PATH ${IMG_FILES})
        get_filename_component(NAME ${PATH} NAME)
        set(IMAGE_REL "../${IMAGE_DIR}/${NAME}")
        message(STATUS "${IMAGE_REL}")
        list(APPEND IMG_LIST "${IMAGE_REL}")
    endforeach()
endif()

find_package(OpenMP REQUIRED)
#find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)

add_subdirectory(prebuilts/llama-cpp)
add_subdirectory(prebuilts/sherpa-onnx)
add_subdirectory(src)