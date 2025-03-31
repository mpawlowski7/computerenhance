set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(BINARY_NAME "ComputerEnhance")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building debug version")
    set(BINARY_NAME "ComputerEnhance_d")
endif ()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

qt_add_executable(${BINARY_NAME}
        llama_llava_phi.cpp
        tts_sherpa_onnx.cpp
        qt_imageworker.cpp
        qt_mainwindow.cpp
        main.cpp
)

qt_add_qml_module(${BINARY_NAME}
        URI MainWindow
        VERSION 1.0
        QML_FILES
        MainWindow.qml
        RESOURCES
        ${IMG_LIST}
)

target_link_libraries(${BINARY_NAME}
        PRIVATE Qt6::Core Qt6::Gui Qt6::Quick Qt6::Qml Qt6::Multimedia
        PRIVATE OpenMP::OpenMP_CXX
        PRIVATE fmt
        PRIVATE ${LLAMA_LIBS}
        PRIVATE ${SHERPA_LIBS}
#        PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static
#        PRIVATE CUDA::cuda_driver
)