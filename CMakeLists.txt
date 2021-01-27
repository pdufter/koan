cmake_minimum_required(VERSION 3.10)

project(koan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(koan koan.cpp)

include_directories("${PROJECT_SOURCE_DIR}/")
include_directories("${PROJECT_SOURCE_DIR}/eigen/")

add_compile_options(-Wall -Wextra -Werror)
target_compile_options(koan PUBLIC -Ofast -march=native -mtune=native)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(koan PRIVATE Threads::Threads)

install(TARGETS koan DESTINATION bin)
