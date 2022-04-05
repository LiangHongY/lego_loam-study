cmake_minimum_required(VERSION 3.0.2)
project(cloud_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    message_generation
    geometry_msgs
    std_msgs
    nav_msgs
)

# Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

generate_messages(
    DEPENDENCIES
    geometry_msgs
    std_msgs
    nav_msgs
)

catkin_package(
    CATKIN_DEPENDS
    message_runtime
    message_generation
    geometry_msgs
    std_msgs
    nav_msgs
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)
