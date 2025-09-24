cmake_minimum_required(VERSION 3.25)
project(wobblyjelly
        DESCRIPTION "window wobblyjelly drag animation"
        VERSION 0.1
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
# link to Hyprland plugin interface (adjust if package name differs)
find_path(HYPRLAND_PLUGIN_INCLUDE_DIR NAMES PluginAPI.hpp PATHS ${HYPRLAND_SRC} /usr/include/hyprland)

include_directories(${HYPRLAND_PLUGIN_INCLUDE_DIR})
find_package(OpenGL REQUIRED)

add_library(wobblyjelly SHARED main.cpp)
target_include_directories(wobblyjelly PRIVATE ${OPENGL_INCLUDE_DIR})
target_link_libraries(wobblyjelly PRIVATE GL)
# if hyprland exports a library to link, add it here:
# target_link_libraries(wobblyjelly PRIVATE hyprland)