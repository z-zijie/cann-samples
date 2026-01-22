set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

add_executable(vector_add main.cpp)

target_include_directories(vector_add PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(vector_add PRIVATE ${ASCEND_DIR}/lib64)

target_compile_options(vector_add PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O3"
)

target_link_libraries(vector_add
    PRIVATE
    m
    dl
    platform
    tiling_api
)

install(TARGETS vector_add
    RUNTIME DESTINATION 0_Introduction/vector_add
)
