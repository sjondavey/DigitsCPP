cmake_minimum_required(VERSION 3.16)

include (${CMAKE_CURRENT_LIST_DIR}/project-meta-info.in)

#================================#
#  Global Configuration          #
#================================#
set(PROJECT_NAME_STR neural_network_first_principles)
project(${PROJECT_NAME_STR}
        VERSION ${project_version}    # from project-meta-info.in
        DESCRIPTION ${project_description}
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")

#https://www.youtube.com/watch?v=y9kSr5enrSk

# Find other CMakeList.txt
add_subdirectory(src)
add_subdirectory(app)

# Testing : see https://cliutils.gitlab.io/modern-cmake/chapters/testing.html
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    MESSAGE("CTest included")
endif()

enable_testing()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(test)
endif()


# include_directories (
#     E:/SD/Programming/C++/vcpkg/installed/x64-windows/include
# )


# Notes:
# Good Overall refernces for CMake projects:
# https://stackoverflow.com/questions/40227333/cmake-setup-multiple-projects-and-dependiencies-between-them/40242257#40242257

# include_directories vs. target_include_directories (https://stackoverflow.com/questions/31969547/what-is-the-difference-between-include-directories-and-target-include-directorie)
# include_directories(x/y): accessible for all the files in the source-tree
# target_include_directories(t x/y): accessible for a specific target when compile
