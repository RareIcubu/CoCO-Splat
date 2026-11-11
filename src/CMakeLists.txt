cmake_minimum_required(VERSION 3.22)

project(CoCO-Splat LANGUAGES CXX C CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

option(USE_IMGUI "Enable ImGui" ON)

find_package(OpenMP REQUIRED)

find_package(glfw3 3.3 REQUIRED)

find_package(OpenGL REQUIRED)

find_package(glm REQUIRED)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.93.0
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_gl STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)


target_include_directories(imgui_gl PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(imgui_gl PUBLIC
    glfw
    OpenGL::GL
)

set(TARGET_NAME CoCO-Splat)

add_executable(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME} PRIVATE USE_IMGUI)

target_sources(${TARGET_NAME} PRIVATE
    main.cpp
    glad.c
    shader.cpp
    shader.h
    camera.h
    camera.cpp
    parser.h
    parser.cpp
    sorter.cu
    EngineContext.h
    EngineContext.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE
    glfw
    OpenGL::GL
    glm::glm
    OpenMP::OpenMP_CXX
    CUDA::cudart_static
    imgui_gl
)
