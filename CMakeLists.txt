cmake_minimum_required(VERSION 2.8.11)
project(HW1_Andrea_Tonon)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} include)

add_library(calibration
	src/calibration.cpp
)

target_link_libraries(calibration
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
)

add_library(pointCloud
	src/pointCloud.cpp
)

target_link_libraries(pointCloud
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
)

add_library(reconstruction3D
        src/reconstruction3D.cpp
)

target_link_libraries(reconstruction3D
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
)

add_executable(hw1
	src/main.cpp
)

target_link_libraries(hw1
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	calibration
        pointCloud
        reconstruction3D
)
