set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

# simt
add_executable(simt main.cpp)
target_include_directories(simt PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(simt PRIVATE ${ASCEND_DIR}/lib64)
target_compile_options(simt PRIVATE
    "-xcce"
    "--npu-arch=dav-3101"
    "-w"
    "-O1"
)
target_link_libraries(simt
    PRIVATE
    m
    dl
    platform
    tiling_api
    ascendcl
    runtime
    stdc++
)
install(TARGETS simt
    RUNTIME DESTINATION 1_Features/simt
)