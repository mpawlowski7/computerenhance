# Correct order is important
set(SHERPA_LIBS sherpa-onnx-cxx-api sherpa-onnx-c-api sherpa-onnx-core kaldi-decoder-core sherpa-onnx-kaldifst-core sherpa-onnx-fstfar
                sherpa-onnx-fst kaldi-native-fbank-core piper_phonemize espeak-ng ucd onnxruntime ssentencepiece_core)

foreach (LIBNAME IN LISTS SHERPA_LIBS)
    add_library(${LIBNAME} STATIC IMPORTED GLOBAL)
    set_target_properties(${LIBNAME} PROPERTIES
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/lib${LIBNAME}.a"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endforeach ()

set(SHERPA_LIBS ${SHERPA_LIBS} PARENT_SCOPE)