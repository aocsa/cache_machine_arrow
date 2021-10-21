cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cache-machine-arrow LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
set(ARROW_HOME $ENV{CONDA_PREFIX})
set(CMAKE_SYSTEM_PREFIX_PATH
  "${ARROW_HOME};${CMAKE_SYSTEM_PREFIX_PATH}"
)
find_package(Arrow REQUIRED)
find_package(Threads)

include_directories(
  ${PROJECT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${ARROW_INCLUDE_DIR}
  $ENV{CONDA_PREFIX}/include
)
link_directories(${ARROW_LIB_DIR} $ENV{CONDA_PREFIX}/lib)
set(SRC_FILES
  ${PROJECT_SOURCE_DIR}/src/main.cpp
)
add_executable(cache-machine-arrow ${SRC_FILES})
target_link_libraries(cache-machine-arrow
  libarrow.so
  libarrow_dataset.so
  libarrow_testing.so
  libparquet.so
  libgtest.so
  libgmock.so
  libpthread.so
)
feature_summary(
  WHAT ALL
  INCLUDE_QUIET_PACKAGES
  FATAL_ON_MISSING_REQUIRED_PACKAGES
)
