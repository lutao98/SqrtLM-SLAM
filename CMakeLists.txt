cmake_minimum_required(VERSION 3.0.2)
project(lt_slam)
set( CMAKE_BUILD_TYPE "Release" )
add_compile_options(-std=c++14 -O3)
#add_compile_options(-std=c++11)
set (CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
  roscpp
  rosbag
  rospy
  sensor_msgs
  std_msgs
  nav_msgs
  tf
  tf_conversions
  geometry_msgs
  nav_msgs
  dynamic_reconfigure
  message_generation
  message_runtime
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES raildetector
 CATKIN_DEPENDS pcl_ros roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

add_definitions(-w)  #忽略warning

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()



#find_package(Gflags)
#include_directories(${GFLAGS_INCLUDE_DIRS})

#find_package(Glog)
#include_directories( ${GLOG_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED COMPONENTS system)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIR})

find_package(Sophus REQUIRED)
#set(Sophus_INCLUDE_DIRS "/usr/local/include/sophus")
#set(Sophus_LIBS "/usr/local/lib/libSophus.so")
include_directories(${Sophus_INCLUDE_DIRS})

include_directories(
include
${catkin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/algorithm
${PROJECT_SOURCE_DIR}/include/data_structure
${PROJECT_SOURCE_DIR}/include/utils
${PROJECT_SOURCE_DIR}/include/frontend
${PROJECT_SOURCE_DIR}/include/backend
${PROJECT_SOURCE_DIR}/include/backend/mybackend
${EIGEN3_INCLUDE_DIR}
#${Boost_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(lom_EXT_LIBS
#  ${catkin_LIBRARIES}
#  ${PCL_LIBRARIES}
#  ${OpenCV_LIBRARIES}
#  ${GFLAGS_LIBRARIES}
#  ${GLOG_LIBRARIES}
#  ${CERES_LIBRARIES}
#)

add_library(${PROJECT_NAME} SHARED
src/utils/Converter.cc
src/utils/KalmanFilter.cc
src/utils/lidarconfig.cc
src/data_structure/MapPoint.cc
src/data_structure/KeyFrame.cc
src/data_structure/Map.cc
src/data_structure/Frame.cc
src/frontend/ORBextractor.cc
src/frontend/ORBmatcher.cc
src/frontend/Initializer.cc
src/frontend/Tracking.cc
src/backend/LocalMapping.cc
src/backend/LoopClosing.cc
src/backend/Optimizer.cc
src/backend/g2oOptimizer.cc
src/backend/CeresOptimizer.cc
src/backend/myOptimizer.cc
src/backend/mybackend/edge_reprojection.cc
src/backend/mybackend/edge.cc
src/backend/mybackend/loss_function.cc
src/backend/mybackend/problem.cc
src/backend/mybackend/vertex_pose.cc
src/backend/mybackend/vertex.cc
src/algorithm/KeyFrameDatabase.cc
src/algorithm/Sim3Solver.cc
src/algorithm/PnPsolver.cc
src/algorithm/lidarOdom.cc
src/System.cc
src/fusionsystem.cc
)


link_directories(${PCL_LIBRARY_DIRS})

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${catkin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PCL_LIBRARIES}
${CERES_LIBRARIES}
${Sophus_LIBRARIES}
)

add_executable(lidar_slam Node/lidar_slam.cpp)
add_dependencies(lidar_slam ${PROJECT_NAME})
target_link_libraries(lidar_slam
${PROJECT_NAME}
)

add_executable(kittiHelper Node/kittiHelper.cpp)
target_link_libraries(kittiHelper
${catkin_LIBRARIES}
${PCL_LIBRARIES}
${OpenCV_LIBS}
)

add_executable(fusion_slam Node/fusion_slam.cpp)
add_dependencies(fusion_slam ${PROJECT_NAME})

target_link_libraries(fusion_slam
${PROJECT_NAME}
#${catkin_LIBRARIES}
)
