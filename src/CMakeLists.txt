cmake_minimum_required(VERSION 3.22)

project(CoCO-Splat LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

find_package(glfw3 3.3 REQUIRED)

find_package(OpenGL REQUIRED)

find_package(glm REQUIRED)

set(TARGET_NAME CoCO-Splat)

add_executable(${TARGET_NAME})

target_sources(${TARGET_NAME} PRIVATE
    main.cpp
    glad.c
    shader.cpp
    shader.h
    camera.h
    camera.cpp
    parser.h
    parser.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE
    glfw
    OpenGL::GL
    glm::glm
    OpenMP::OpenMP_CXX
)
