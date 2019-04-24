cmake_minimum_required(VERSION 2.8.3)
project(camera-extrinsics-calibrator)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -fopenmp -mfma")

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(GTest)
if (${GTEST_NOTFOUND})
    add_subdirectory(lib/googletest)
endif()

add_definitions("-DMC_SIM")
add_definitions("-DNUM_FEATURES=12")

include_directories(include
    ${GTEST_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    lib/geometry/include
)

include_directories(include
    lib
    lib/multirotor_sim/include
    lib/geometry/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    lib/VI-EKF/include
)

add_subdirectory(lib/geometry)
add_subdirectory(lib/multirotor_sim)
add_subdirectory(lib/VI-EKF)

# CALIBRATION LIBRARY
add_library(camera_extrinsics_calibration
    src/cal.cpp
)
target_link_libraries(camera_extrinsics_calibration
    geometry
    ${CERES_LIBRARIES}
    )

# SIMULATION LIBRARY
add_library(calibration_simulator src/sim.cpp)
target_link_libraries(calibration_simulator
    geometry
    multirotor_sim
    vi_ekf
    camera_extrinsics_calibration)

# UNIT TESTS EXECUTABLE
add_executable(calibration_test
    src/test/test.cpp
)
target_link_libraries(calibration_test
    calibration_simulator
    camera_extrinsics_calibration
    gtest
    gtest_main
    ${GTEST_LIBRARIES}
)

