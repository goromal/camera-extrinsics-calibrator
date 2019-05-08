cmake_minimum_required(VERSION 2.8.3)
project(vi_ekf)

set(CMAKE_CXX_FLAGS "-std=c++17 -mavx -mfma")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#find_package(catkin REQUIRED COMPONENTS
#  roscpp
#  rospy
#  sensor_msgs
#  image_transport
#  cv_bridge
#  geometry_msgs
#  nav_msgs
#  rosbag
#)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

#catkin_package(
#    INCLUDE_DIRS include
#    CATKIN_DEPENDS roscpp sensor_msgs image_transport cv_bridge geometry_msgs nav_msgs
#)

if (NOT TARGET geometry)
    add_subdirectory(lib/geometry)
    include_directories(lib/geometry/include)
endif()


###########
## Build ##
###########

include_directories(include)
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
#  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

add_library(math_helper
        src/math_helper.cpp
        include/math_helper.h
)

add_library(vi_ekf STATIC
  src/vi_ekf/vi_ekf.cpp
  src/vi_ekf/vi_ekf_helper.cpp
  src/vi_ekf/vi_ekf_feat.cpp
  src/vi_ekf/vi_ekf_log.cpp
  src/vi_ekf/vi_ekf_meas.cpp
  src/vi_ekf/vi_ekf_error.cpp
  src/vi_ekf/vi_ekf_kfr.cpp
  src/vi_ekf/vi_ekf_dyn.cpp
  include/vi_ekf.h
)
target_link_libraries(vi_ekf math_helper ${YAML_CPP_LIBRARIES} geometry)

add_library(klt_tracker
  src/klt_tracker.cpp
  include/klt_tracker.h
)

add_executable(jac_test src/test/jac_test.cpp)
target_link_libraries(jac_test ${GTEST_LIBRARIES} pthread vi_ekf geometry)

#add_library(vi_ekf_ros
#  src/vi_ekf_ros.cpp
#  include/vi_ekf_ros.h
#  include/eigen_helpers.h
#)
#target_link_libraries(vi_ekf_ros
##  ${catkin_LIBRARIES}
#  ${OpenCV_LIBS}
#  vi_ekf
#  klt_tracker
#)

#add_executable(vi_ekf_node
#  src/vi_ekf_node.cpp
#  )
#target_link_libraries(vi_ekf_node
##  ${catkin_LIBRARIES}
#  vi_ekf_ros
#  geometry
#)

#add_executable(vi_ekf_rosbag
#  src/rosbag_parser.cpp
#  )
#target_link_libraries(vi_ekf_rosbag
#  ${catkin_LIBRARIES}
#  vi_ekf_ros
#  geometry
#  vi_ekf
#)
