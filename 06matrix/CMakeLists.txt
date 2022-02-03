

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(main)
# cuda库
FIND_PACKAGE(CUDA REQUIRED)
# 编译链接
CUDA_ADD_EXECUTABLE(matrix matrix.cu)
TARGET_LINK_LIBRARIES(matrix)

# ==================================================
# cmake_minimum_required(VERSION 2.8)
# project(SOR_openMP)
# set(CMAKE_CXX_STANDARD 11)
 

# # PCL
# find_package(PCL 1.2 REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

# # OpenCV
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )

# # openMP settings
# FIND_PACKAGE(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     message("OPENMP FOUND")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif ()

# # ncnn
# # find_package(ncnn)


# # 编译链接
# add_executable (01sum 01sum.cpp)
# # 链接pcl和opencv会报错
# # target_link_libraries (01sum ${OpenMP_LIBRARIES} ${PCL_LIBRARIES}  ${OpenCV_LIBRARIES})
# target_link_libraries (01sum ${OpenMP_LIBRARIES} )
