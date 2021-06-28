cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mcas-gmra)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# find_package(Python REQUIRED)


get_filename_component(BASE_ROOT ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
get_filename_component(3RD_PARTY_ROOT ${BASE_ROOT}/3rd_party ABSOLUTE)
get_filename_component(BIN_ROOT ${BASE_ROOT}/bin ABSOLUTE)
get_filename_component(TESTS_ROOT ${BIN_ROOT}/tests ABSOLUTE)

# c/c++ srcs
get_filename_component(CSRC_ROOT ${BASE_ROOT}/csrc ABSOLUTE)
get_filename_component(SRC_ROOT ${CSRC_ROOT}/src ABSOLUTE)
get_filename_component(INC_ROOT ${CSRC_ROOT}/include ABSOLUTE)
get_filename_component(PYBIND_ROOT ${CSRC_ROOT}/pybind ABSOLUTE)

# python srcs
get_filename_component(PYSRC_ROOT ${BASE_ROOT}/pysrc ABSOLUTE)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# set(BUILD_PYTESTS   OFF)

# everything needs pybind11
add_subdirectory(${3RD_PARTY_ROOT}/pybind11)
# add_subdirectory(${3RD_PARTY_ROOT}/pytorch_sparse)

include_directories(${INC_ROOT})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${3RD_PARTY_ROOT})

# message(STATUS ${TORCH_INCLUDES})

# build modules first
set(HDRS
    ${INC_ROOT}/trees/covertree.h
)
set(SRCS
    ${SRC_ROOT}/covertree.cc
)
set(PYBIND_SRCS
    ${PYBIND_ROOT}/pybind_trees.cc
)
add_library(gmra_trees_cpp SHARED
            ${HDRS} ${SRCS})
target_link_libraries(gmra_trees_cpp PUBLIC
                      "${TORCH_LIBRARIES}"
                      ${TORCH_PYTHON_LIBRARY})

pybind11_add_module(gmra_trees MODULE
                    ${PYBIND_SRCS})
target_link_libraries(gmra_trees PRIVATE
                      gmra_trees_cpp
                       "${TORCH_LIBRARIES}"
                      ${TORCH_PYTHON_LIBRARY})

