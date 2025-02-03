# Correct order is important
set(LLAMA_LIBS llava_static common ggml-base llama ggml ggml-cpu ggml-cuda)

foreach (LIBNAME IN LISTS LLAMA_LIBS)
    add_library("${LIBNAME}" STATIC IMPORTED GLOBAL)
    set_target_properties("${LIBNAME}" PROPERTIES
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/lib${LIBNAME}.a"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endforeach ()

set(LLAMA_LIBS ${LLAMA_LIBS} PARENT_SCOPE)