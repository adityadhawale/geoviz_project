cmake_minimum_required(VERSION 2.8.3)
project(geoviz_project)

add_compile_options(-std=c++17)

#set(Ceres_DIR "/home/aditya/libraries/ceres-solver/release/lib/cmake/Ceres")
set(EIGEN3_DIR "/home/aditya/libraries/eigen-335/release/share/eigen3/cmake")

find_package(Eigen3 REQUIRED)
#find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "{CMAKE_CXX_STANDARD_REQUIRED} -O3 -ffast_math -g")

find_package(catkin REQUIRED COMPONENTS
	std_msgs
	sensor_msgs
	message_filters
	image_transport
	cv_bridge
	rosbag
	nav_msgs
	cpp_utils
	roscpp
	gmm_msgs
	pcl_ros
	aditya_utils
	tf
)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
catkin_package(CATKIN_DEPENDS roscpp
	DEPENDS OpenCV Eigen3 PCL)

#catkin_python_setup()

include_directories(
	include 
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIRS}
	#${CERES_INCLUDE_DIRS}
)

link_directories(
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIRS}
	#${CERES_INCLUDE_DIRS}
	include
)

add_library(multi_stereo_mapping src/stereo_mapping/multi_stereo_mapping.cpp)
target_link_libraries(multi_stereo_mapping
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
	${OpenCV_LIBRARIES}
	yaml-cpp
)

add_executable(multi_stereo_mapping_node src/stereo_mapping/multi_stereo_mapping_node.cpp)
target_link_libraries(multi_stereo_mapping_node
	multi_stereo_mapping
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	yaml-cpp
)

