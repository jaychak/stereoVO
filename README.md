# Stereo Visual Odometry (stereoVO)

This repository provides a C++ implementation of stereo visual odometry (VO), estimating camera pose from sequential stereo/RGBD image pairs. It's designed for robotics, SLAM, and computer vision applications, focusing on real-time performance.

## Features
- **C++ based stereo VO**: Efficient pose estimation using stereo (or left + depth) image pairs.
- **6-DOF trajectory estimation**: Accurate camera movement tracking.

## Getting Started

### Prerequisites
- CMake 3.10+
- OpenCV 4.0+
- Eigen3

### Building the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/jaychak/stereoVO.git

2. Create a build directory and compile:
   mkdir build && cd build
   cmake ..
   make

3. Run the code (paths to left and right images are hard-coded) 
   ./stereoVO

