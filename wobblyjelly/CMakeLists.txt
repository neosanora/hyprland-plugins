cmake_minimum_required(VERSION 3.25)
project(wobblyjelly
        DESCRIPTION "window wobblyjelly drag animation"
        VERSION 0.1
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pastikan HYPRLAND_SRC di-pass dari hyprpm build
if(NOT DEFINED HYPRLAND_SRC)
    message(FATAL_ERROR "HYPRLAND_SRC not set. Run cmake with -DHYPRLAND_SRC=/path/to/hyprland")
endif()

include_directories(${HYPRLAND_SRC}/)
include_directories(${HYPRLAND_SRC}/src/)

find_package(PkgConfig REQUIRED)
pkg_check_modules(DEPS REQUIRED wayland-server egl)

add_library(wobblyjelly SHARED
    main.cpp
)

target_include_directories(wobblyjelly PRIVATE
    ${DEPS_INCLUDE_DIRS}
)

target_link_libraries(wobblyjelly
    ${DEPS_LIBRARIES}
    GL
)

# Export ke folder build (hyprpm expect .so file di plugins/)
set_target_properties(wobblyjelly PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
)
