set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

qt_add_executable(ComputerEnhance
        main.cpp
        qt_mainwindow.cpp
        llama_llava_phi.cpp
        tts_sherpa_onnx.cpp
)

qt_add_qml_module(ComputerEnhance
        URI MainModule
        SOURCES
        main.cpp
        QML_FILES
        Main.qml
)

target_link_libraries(ComputerEnhance
        PRIVATE Qt6::Core Qt6::Gui Qt6::Quick
        PRIVATE OpenMP::OpenMP_CXX
        PRIVATE llava_static llama ggml ggml-base ggml-cpu ggml-cuda
        PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static
        PRIVATE CUDA::cuda_driver
        PRIVATE fmt
)