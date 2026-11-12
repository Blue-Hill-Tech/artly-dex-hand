cmake_minimum_required(VERSION 3.22)
project(dexrt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DEXRT_BUILD_TESTS "Build unit tests" ON)
option(DEXRT_BUILD_TOOLS "Build command-line tools" ON)

include(cmake/Dependencies.cmake)

add_library(dexrt SHARED
  src/kinematics.cpp
  src/retargeter.cpp
  src/impedance.cpp
  src/compliance.cpp
  src/d21_canfd.cpp
  src/c_api.cpp)
target_include_directories(dexrt PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(dexrt PUBLIC Eigen3::Eigen PRIVATE osqpstatic tinyxml2::tinyxml2)
target_compile_options(dexrt PRIVATE -Wall -Wextra)

if(DEXRT_BUILD_TOOLS)
  add_executable(dexrt_impedance_loop tools/impedance_loop.cpp)
  target_link_libraries(dexrt_impedance_loop PRIVATE dexrt)
  target_compile_options(dexrt_impedance_loop PRIVATE -Wall -Wextra)
endif()

if(DEXRT_BUILD_TESTS)
  enable_testing()
  foreach(name kinematics retargeter impedance compliance modbus motor_map)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE dexrt)
    target_compile_definitions(test_${name} PRIVATE
      DEXRT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      DEXRT_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()
endif()
