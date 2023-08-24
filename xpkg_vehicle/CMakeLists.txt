cmake_minimum_required(VERSION 3.0.2)
project(xpkg_vehicle)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  xpkg_comm
  geometry_msgs
  nav_msgs
  tf
  std_msgs
)

catkin_package(

)

add_executable(xnode_vehicle
src/xnode_vehicle.cpp
src/vehicle_calc_odom.cpp
src/vehicle_data_convert.cpp
)

include_directories(
include
  ${catkin_INCLUDE_DIRS}
)

target_link_libraries(xnode_vehicle
   ${catkin_LIBRARIES}
 )

add_dependencies(xnode_vehicle xpkg_comm_gencpp)
