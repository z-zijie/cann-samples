set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

# quantize
add_executable(hif8_mm_demo hif8_matmul.cpp)
target_include_directories(hif8_mm_demo PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(hif8_mm_demo PRIVATE ${ASCEND_DIR}/lib64)
target_compile_options(hif8_mm_demo PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O3"
)
target_link_libraries(hif8_mm_demo
    PRIVATE
    m
    dl
    platform
    tiling_api
)
install(TARGETS hif8_mm_demo
    RUNTIME DESTINATION 1_Features/hif8
)
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
    DESTINATION 1_Features/hif8
    FILES_MATCHING
    PATTERN "*.py"
)