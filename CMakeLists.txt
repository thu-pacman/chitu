cmake_minimum_required(VERSION 3.12.2)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0048 NEW)
project(cinfer-cpp LANGUAGES CXX)


if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
  cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
  message("ARCH_FLAGS = ${ARCH_FLAGS}")
  set(CMAKE_CUDA_ARCHITECTURES ${ARCH_FLAGS})
else()
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  SET(CUDA_ARCH_LIST 86)
  message("ARCH_FLAGS = ${CUDA_ARCH_LIST}")
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  # set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0) # necessary if using googletest and torch

# set(CMAKE_CXX_STANDARD 14)

set(Python_FIND_STRATEGY "LOCATION") # Find the first Python in the path (which is the Python we load),
                                     # instead of finding Python of the most latest version
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# find torch path
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE TorchVersion
)
message(STATUS "Torch Version: ${TorchVersion}")


set(CInfer_Compute_CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/cpp)
set(CMAKE_MODULE_PATH "${CInfer_Compute_CPP_SOURCE}/cmake" ${CMAKE_MODULE_PATH})
set(project_INCLUDE_DIR ${CInfer_Compute_CPP_SOURCE}/include)
set(project_SOURCE_DIR ${CInfer_Compute_CPP_SOURCE}/src)
# set(project_TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/test)
set(project_DOCS_DIR ${CMAKE_CURRENT_LIST_DIR}/docs)
set(project_BINDINGS_DIR ${CInfer_Compute_CPP_SOURCE}/python_bindings)
set(project_THIRD_PARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/src/third_party)

set(project_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "working dir: ${project_WORKING_DIR}")

add_definitions(-DCInfer_DL_BASE_DIRECTORY="${CMAKE_CURRENT_LIST_DIR}")
add_definitions(-DCInfer_DL_TEST_DIRECTORY="${project_TEST_DIR}")

# set(CMAKE_CXX_FLAGS "-fopenmp -Wall ")
# set(CMAKE_CUDA_FLAGS " -fPIC -fopenmp -g -O3  --use_fast_math -Wall ")
set(CMAKE_CUDA_FLAGS " -fPIC -g -O3  --use_fast_math ")

if (EXISTS ${project_INCLUDE_DIR})
    file(GLOB_RECURSE project_HEADERS ${project_HEADERS} ${project_INCLUDE_DIR}/*.h)
endif ()
if (EXISTS ${project_SOURCE_DIR})
    file(GLOB_RECURSE project_SOURCES ${project_SOURCES} ${project_SOURCE_DIR}/*.cu ${project_SOURCE_DIR}/*.cpp)
endif ()
if (EXISTS ${project_BINDINGS_DIR})
    file(GLOB_RECURSE project_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/*.cpp)
endif ()

add_subdirectory(${project_THIRD_PARTY_DIR})
include_directories(${project_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${project_THIRD_PARTY_DIR}/inih ${project_THIRD_PARTY_DIR}/pybind11/include ${project_THIRD_PARTY_DIR}/parallel_hashmap/parallel_hashmap)
include_directories(${project_THIRD_PARTY_DIR}/yaml)
link_directories(${project_THIRD_PARTY_DIR}/yaml)
# include_directories(${project_THIRD_PARTY_DIR}/yaml-cpp/include)


add_library(${PROJECT_NAME} SHARED ${project_SOURCES} ${project_HEADERS})

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

message(STATUS "torch library: ${TORCH_LIBRARY}")
target_link_libraries(${PROJECT_NAME} spdlog::spdlog)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} cxxopts)
target_link_libraries(${PROJECT_NAME} Yaml)


set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${project_HEADERS}")

if (EXISTS ${project_INCLUDE_DIR})
    target_include_directories(${PROJECT_NAME} PUBLIC ${project_INCLUDE_DIR})
endif ()
if (EXISTS ${project_SOURCE_DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE ${project_SOURCE_DIR})
endif ()

if (EXISTS ${project_TEST_DIR})
    enable_testing()
    add_subdirectory(${project_TEST_DIR})
endif ()

pybind11_add_module(cinfer_backend ${project_BINDINGS})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
target_link_libraries(cinfer_backend PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(cinfer_backend PRIVATE ${PYTHON_LIBRARIES})
target_include_directories(cinfer_backend PRIVATE ${project_INCLUDE_DIR})
