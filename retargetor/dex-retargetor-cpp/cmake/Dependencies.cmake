# Third-party dependencies. Everything is fetched at configure time and built
# into the tree, so the module does not depend on a conda environment or the
# Python venv. Set FETCHCONTENT_SOURCE_DIR_<NAME> to point at a local checkout
# for offline builds.
include(FetchContent)

# Eigen (header-only). Prefer a system package when present.
find_package(Eigen3 3.3 QUIET CONFIG)
if(NOT TARGET Eigen3::Eigen)
  FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

# OSQP 1.0 (static, built-in algebra, long indices, double precision).
FetchContent_Declare(osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG v1.0.0
  GIT_SHALLOW TRUE)
set(OSQP_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
set(OSQP_ENABLE_PRINTING OFF CACHE BOOL "" FORCE)  # polish notices ignore verbose=0 and would flood the loop
set(OSQP_ENABLE_PROFILING ON CACHE BOOL "" FORCE)
set(OSQP_ENABLE_INTERRUPT OFF CACHE BOOL "" FORCE)
set(OSQP_CODEGEN OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(osqp)

# tinyxml2 for URDF parsing.
FetchContent_Declare(tinyxml2
  GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
  GIT_TAG 10.0.0
  GIT_SHALLOW TRUE)
set(tinyxml2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(tinyxml2_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinyxml2)
