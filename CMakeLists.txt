cmake_minimum_required(VERSION 3.19.0 FATAL_ERROR)
project(GEMM LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

#check if cuda is available
include(CheckLanguage)
check_language(CUDA)

#set versions
set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 86)

add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

# Configure headr file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

#cllects names of the sources files in dir and store the list in SRC
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(gemm gemm.cu ${SRC})
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(gemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# add_executable(cuBLAS_gemm cuBLAS_gemm.cu)
# set_target_properties(cuBLAS_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
# target_link_libraries(cuBLAS_gemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# add_executable(simplest_kernel simplest_kernel.cu)
# set_target_properties(simplest_kernel PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
# target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})