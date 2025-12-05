cmake_minimum_required(VERSION 3.22)

project(3DGS-OpenGL LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 3.3 REQUIRED)

find_package(OpenGL REQUIRED)

find_package(glm REQUIRED)

set(TARGET_NAME 3DGS-OpenGL)

add_executable(${TARGET_NAME})

target_sources(${TARGET_NAME} PRIVATE
    main.cpp
    glad.c
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE
	glfw
	OpenGL::GL
    glm::glm
)
