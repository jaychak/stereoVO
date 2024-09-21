cmake_minimum_required(VERSION 2.8.3)
project(stereovo)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -march=native")
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("OPENMP FOUND")
endif()

#set("Eigen3_DIR" "/home/jay/Downloads/eigen-3.4.0")


find_package(OpenCV 4.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Pangolin REQUIRED)




include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(shared_libs
	${EIGEN3_LIBS}
	${OPENGL_LIBRARIES}
	${Pangolin_LIBRARIES}
	${OpenCV_LIBS}
)


add_executable(stereovo src/main.cpp)

target_link_libraries(stereovo ${OpenCV_LIBS} ${EIGEN3_LIBS} ${OPENGL_LIBRARIES} ${Pangolin_LIBRARIES})

#target_link_libraries(eigen-test-cmake ${SDL2_LIBRARIES} ${OPENGL_LIBRARIES})

