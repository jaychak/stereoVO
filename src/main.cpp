//************************************************************************
// Stereo VO with BA.
// No loop closure.
//
// Jay Chakravarty

// Does VO on a sequence of stereo image.
// Trying to add fixed-lag smoothing with BA.

//************************************************************************


// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/geometry/Cal3_S2Stereo.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/nonlinear/NonlinearEquality.h>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
// #include <gtsam/inference/Symbol.h>
// #include <gtsam/slam/StereoFactor.h>
// #include <gtsam/slam/dataset.h>


#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cmath> // Include this at the top of your file for std::isfinite

#include <atomic>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/calib3d.hpp>

#include <pangolin/pangolin.h>
#include <vector>
#include <map>

#include <chrono>
#include <thread>


#include <opencv2/aruco.hpp>



#include "Eigen/Geometry"

using namespace std;
using namespace cv;

const float SCALE_FACT_ZED = 0.3125/1000;

#define IMAGE_READ_RESIZE
//#define GTSAM_INC

//#ifdef GTSAM

#define ADD_NEW_PTS



//using namespace gtsam;

//#endif


//*****************Pangolin Viewer Class***************************************

class pangolinViewer
{
public:
    pangolinViewer();
    ~pangolinViewer();

    void init();
    void run();
    void stop();
    void draw_path(float point_colour[], float point_size);
    void add_path_point(cv::Point3f new_pt);
    void add_path_pose(cv::Mat new_pose);
    //void add_path_pose_ba(cv::Mat new_pose);
    void add_closest_aruco_pose(cv::Mat closest_aruco_pose);
    void add_pts_3d(vector<cv::Point3f> pts_3d, int frame_idx);
    void draw_path_full_pose();
    
    void draw_aruco_markers(float point_size);

    void draw_landmark_pts(float point_size);
    void print_path();

    void clear_old_landmarks(int current_frame_idx);

private:
    pangolin::View* d_cam;
    pangolin::OpenGlRenderState* s_cam;
    pangolin::Var<bool>* test_button;
    pangolin::Var<bool>* gt_path_check;

    std::vector<cv::Point3f> pathPoints;
    std::vector<cv::Point3f> gtPath;
    std::vector<cv::Point3f> nodes;

    

    vector<cv::Point3f> camera_traj;
    vector<cv::Mat> camera_traj_full_pose;
    //vector<cv::Mat> camera_traj_full_pose_ba;
    vector<cv::Mat> aruco_poses; // loose container for holding multiple Aruco poses. TODO: Modify.
    vector<cv::Point3f> landmark_pts_3d;
    std::vector<std::pair<int, cv::Point3f>> landmark_pts_3d_with_frame;

    cv::Mat cam_se3;

    cv::Mat aruco_draw_matrix; // matrix for drawing Aruco pose.

    float pathPointColor[3]     = {.0f,0.0f,1.0f};
    float landmarkPointColor[3] = {1.0f,0.0f,0.0f};
    float arucoPointColor[3] = {0.0f,1.0f,0.0f};

    int max_pose_history = 50;//0;  // Set a limit on how many poses to keep in visualizer
    // Note: this does not affect poses kept in the main program, just here.
    // Poses from origin to latest time are kept in the main program in the poses_gtsam vector atm.


    std::atomic<bool> stop_viewer;  // Use atomic for thread safety

};

// pangolinViewer::pangolinViewer()
// {

// }
pangolinViewer::pangolinViewer() : stop_viewer(false)  // Initialize stop_viewer to false
{
}


pangolinViewer::~pangolinViewer()
{
    if (s_cam) {
        delete s_cam;
        s_cam = nullptr;
    }

    if (test_button) {
        delete test_button;
        test_button = nullptr;
    }

    if (gt_path_check) {
        delete gt_path_check;
        gt_path_check = nullptr;
    }

    pangolin::DestroyWindow("Main");  // Destroy Pangolin window

}

void pangolinViewer::stop()  // Implementation of the stop() method
{
    stop_viewer = true;  // Set stop_viewer to true to stop the loop
}


void pangolinViewer::add_path_point(cv::Point3f new_pt)
{
    camera_traj.push_back(new_pt);
    
}

void pangolinViewer::add_path_pose(cv::Mat new_pose)
{
    camera_traj_full_pose.push_back(new_pose);

    // Periodically clear out old data if the history exceeds the limit
    if (camera_traj_full_pose.size() > max_pose_history) {
        camera_traj_full_pose.erase(camera_traj_full_pose.begin(), camera_traj_full_pose.begin() + max_pose_history / 2);
    }
    
}

void pangolinViewer::add_pts_3d(vector<cv::Point3f> pts_3d, int frame_idx) {
    for (auto& pt : pts_3d) {
        landmark_pts_3d_with_frame.push_back(std::make_pair(frame_idx, pt));
    }
}


// Pangolin aruco display
void pangolinViewer::add_closest_aruco_pose(cv::Mat closest_aruco_pose)
{
    aruco_poses.clear();
    aruco_poses.push_back(closest_aruco_pose);

}

void pangolinViewer::init()
{



    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Main",640,480);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Define Camera Render Object (for view / scene browsing)
    s_cam = new pangolin::OpenGlRenderState  (
    pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
    //pangolin::ModelViewLookAt(3,-4.5,0, 0,0,+0.00, pangolin::AxisNegY)
    pangolin::ModelViewLookAt(3,-10.5,0, 0,0,+0.00, pangolin::AxisNegY)//-10.5 initializes the camera further up (zooms out) compared to -4.5
    );


    const int UI_WIDTH = 100;

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam_t = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
    .SetHandler(new pangolin::Handler3D(*s_cam));
    d_cam = &d_cam_t;

    // d_cam = &d_cam_t;

    // Add named Panel and bind to variables beginning 'ui'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));


    test_button = new pangolin::Var<bool>("ui.Test",false,false);
    gt_path_check = new pangolin::Var<bool>("ui.Draw_Gt",false,true);

    camera_traj.push_back(cv::Point3f(0,0,0));

    cv::Mat orig_pose = cv::Mat::zeros(cv::Size(4,4), CV_64FC1);
    orig_pose.at<float>(3,3) = 1.0f;

    camera_traj_full_pose.push_back(orig_pose);

    cv::Mat kk(4, 5, CV_32FC1);
    float scale = 0.5;

    kk.at<float>(0,0) = scale;
    kk.at<float>(0,1) = -scale;
    kk.at<float>(0,2) = -scale;
    kk.at<float>(0,3) = scale;
    kk.at<float>(0,4) = 0;

    kk.at<float>(1,0) = scale;
    kk.at<float>(1,1) = scale;
    kk.at<float>(1,2) = -scale;
    kk.at<float>(1,3) = -scale;
    kk.at<float>(1,4) = 0;

    kk.at<float>(2,0) = 0;
    kk.at<float>(2,1) = 0;
    kk.at<float>(2,2) = 0;
    kk.at<float>(2,3) = 0;
    kk.at<float>(2,4) = -scale;

    kk.at<float>(3,0) = 1;
    kk.at<float>(3,1) = 1;
    kk.at<float>(3,2) = 1;
    kk.at<float>(3,3) = 1;
    kk.at<float>(3,4) = 1;

    aruco_draw_matrix = kk;

}

void pangolinViewer::run()
{
    init();
    while( !pangolin::ShouldQuit() && !stop_viewer) 
    {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        if(pangolin::Pushed(*test_button)) {
            cout << gtPath.size() << " " << pathPoints.size() << endl;
        }
        // Activate efficiently by object
        d_cam->Activate(*s_cam);

        // Render the path, poses, and other elements
        draw_path(pathPointColor, 5.0f);

        draw_path_full_pose();

        draw_aruco_markers(8.0f);

        draw_landmark_pts(5.0f);

        // Swap frames and Process Events
        pangolin::FinishFrame();
        //usleep(30e3);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }   

    // Explicit cleanup
    if (pangolin::ShouldQuit()) {
        pangolin::DestroyWindow("Main");
    }
    std::cout << "Pangolin viewer exiting..." << std::endl;

}

void pangolinViewer::draw_path(float point_colour[], float point_size)
{

    glPointSize(point_size);
    glColor3f(point_colour[0], point_colour[1], point_colour[2]);

    glBegin(GL_POINTS);
    for(auto& pt: camera_traj)
    {
        glVertex3f(pt.x, pt.y, pt.z);
    }
    glEnd();
}

void pangolinViewer::draw_path_full_pose(void)
{
    const float w = 0.5;
    const float h = w * 0.75;
    const float z = w * 0.6;

    float lineWidth = 1.5f;

    for (auto& pose : camera_traj_full_pose)
    {
        glPushMatrix();
        float cam_array[16] = {0}; // Initialize to avoid undefined behavior

        if (!pose.empty()) {
            cv::Mat cam_se3_flat = cv::Mat(pose.t()).reshape(1, 1);

            // Ensure cam_se3_flat has exactly 16 elements
            if (cam_se3_flat.total() == 16) {
                for (int i = 0; i < 16; i++) {
                    cam_array[i] = cam_se3_flat.at<float>(i);
                }
            } else {
                std::cerr << "Warning: cam_se3_flat does not have 16 elements, skipping this pose." << std::endl;
                glPopMatrix();
                continue;
            }
        } else {
            std::cerr << "Warning: pose matrix is empty, skipping this pose." << std::endl;
            glPopMatrix();
            continue;
        }

        glMultMatrixf(cam_array);

        glLineWidth(lineWidth);
        glColor3f(0.0f, 0.0f, 1.0f);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(w, h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);
        glVertex3f(w, -h, z);
        glVertex3f(-w, h, z);
        glVertex3f(-w, -h, z);
        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);
        glVertex3f(-w, -h, z);
        glVertex3f(w, -h, z);
        glEnd();

        glPopMatrix();
    }

}

// void pangolinViewer::draw_landmark_pts(float point_size)
// {

//     glPointSize(point_size);
//     glColor3f(landmarkPointColor[0], landmarkPointColor[1], landmarkPointColor[2]);

//     glBegin(GL_POINTS);
//     for(auto& pt: landmark_pts_3d)
//     {
//         glVertex3f(pt.x, pt.y, pt.z);
//     }
//     glEnd();

// }

void pangolinViewer::draw_landmark_pts(float point_size) {
    glPointSize(point_size);
    glColor3f(landmarkPointColor[0], landmarkPointColor[1], landmarkPointColor[2]);

    glBegin(GL_POINTS);
    for (auto& pt_pair : landmark_pts_3d_with_frame) {
        const cv::Point3f& pt = pt_pair.second;
        glVertex3f(pt.x, pt.y, pt.z);
    }
    glEnd();
}

void pangolinViewer::clear_old_landmarks(int current_frame_idx) {
    landmark_pts_3d_with_frame.erase(
        std::remove_if(
            landmark_pts_3d_with_frame.begin(), landmark_pts_3d_with_frame.end(),
            [current_frame_idx, this](const std::pair<int, cv::Point3f>& pt_pair) {
                return (current_frame_idx - pt_pair.first) > this->max_pose_history;
            }
        ),
        landmark_pts_3d_with_frame.end()
    );
}


void pangolinViewer::draw_aruco_markers(float point_size)
{
    float lineWidth = 3.5f;
    glColor3f(0.0f,1.0f,0.0f);

    for(auto& aruco_pose: aruco_poses)
    {
        cv::Mat cMat; //corner matrix
        cMat = aruco_pose*aruco_draw_matrix;

        std::vector<cv::Vec3d> cVecs; //corner vector
        for(int i = 0; i < 4; i++)
        {
            cv::Vec3d corner_vec_this(cMat.at<float>(0,i), cMat.at<float>(1,i), cMat.at<float>(2,i));
            cVecs.push_back(corner_vec_this);
        }
        pangolin::glDrawLine(cVecs[0][0],cVecs[0][1],cVecs[0][2],  cVecs[1][0],cVecs[1][1],cVecs[1][2] );
        pangolin::glDrawLine(cVecs[1][0],cVecs[1][1],cVecs[1][2],  cVecs[2][0],cVecs[2][1],cVecs[2][2] );
        pangolin::glDrawLine(cVecs[2][0],cVecs[2][1],cVecs[2][2],  cVecs[3][0],cVecs[3][1],cVecs[3][2] );
        pangolin::glDrawLine(cVecs[3][0],cVecs[3][1],cVecs[3][2], cVecs[0][0],cVecs[0][1],cVecs[0][2] );
    }



}

void pangolinViewer::print_path()
{
    for(auto& pt:camera_traj)
    {
        cout << pt.x << " " << pt.y << " " << pt.z << endl;
    }
}

//*****************End Pangolin Viewer Class***********************************


void load_images(string image_path, int image_idx, cv::Mat &image_left, cv::Mat &image_right, cv::Mat &image_depth)
{
    string image_name_left  = image_path  + "left/";
    string image_name_right = image_path  + "right/";
    string image_name_depth = image_path  + "depth/";

    stringstream ss;
    ss << image_idx;

    image_name_left  = image_name_left  + ss.str() + ".png";
    image_name_right = image_name_right + ss.str() + ".png";
    image_name_depth = image_name_depth + ss.str() + ".png";

    // cout << image_name_left << endl;
    // cout << image_name_right << endl;
    // cout << image_name_depth << endl;

    cv::Mat image_left_, image_right_, image_depth_;
    image_left_  = imread(image_name_left);
    image_right_ = imread(image_name_right);
    //image_depth_ = imread(image_name_depth, CV_LOAD_IMAGE_ANYDEPTH);
    image_depth_ = imread(image_name_depth, cv::IMREAD_ANYDEPTH );

#ifdef IMAGE_READ_RESIZE
    // cv::resize(image_left_, image_left, cv::Size(), 0.5, 0.5, CV_INTER_CUBIC);
    // cv::resize(image_right_, image_right, cv::Size(), 0.5, 0.5, CV_INTER_CUBIC);
    // cv::resize(image_depth_, image_depth, cv::Size(), 0.5, 0.5, CV_INTER_CUBIC);
    cv::resize(image_left_, image_left, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);
    cv::resize(image_right_, image_right, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);
    cv::resize(image_depth_, image_depth, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);
#else
    image_left  = image_left_;
    image_right = image_right_;
    image_depth = image_depth_;
#endif

    // cv::imshow("left image", image_left);
    // cv::imshow("right image", image_right);
    // cv::imshow("depth image", image_depth);
    // cv::waitKey(500);

}

void draw_pts(cv::Mat image_left, vector<cv::Point2f> image_pts, string image_disp_name)
{
    cv::Mat image_left_ = cv::Mat(image_left.size(), image_left.type());
    image_left.copyTo(image_left_);
    for(unsigned int pt_idx = 0; pt_idx < image_pts.size(); pt_idx++)
    {
        cv::circle(image_left_, image_pts[pt_idx], 3, cv::Scalar(255,255,255), -1, 8);
    }
    cv::imshow(image_disp_name, image_left_);
    cv::waitKey(1);
}

cv::Mat convert_depth(cv::Mat image_depth, float scale_stereo_zed=SCALE_FACT_ZED)
{
    cv::Mat image_depth_float;
    image_depth.convertTo(image_depth_float, CV_32FC1);

    image_depth_float*= scale_stereo_zed;

    return image_depth_float;
}
void get_3d_pts(cv::Mat image_depth_float, vector<cv::Point2f> image_pts, vector<cv::Point2f> &image_pts_trunc, vector<cv::Point3f> &image_pts_3d)
{
    // Unproject 2D pts
    // Multiply with depth
    float fx = 348.925, fy = 351.135, cx = 339.075, cy = 177.45; // 348.925 351.135 339.075 177.45

    //vector<cv::Point3f> unprojected_pts;

    for(auto& kp: image_pts) 
    {
        float x = (kp.x - cx) / fx;
        float y = (kp.y - cy) / fy;
        cv::Point3f unprojected_pt(x, y, 1);

        unprojected_pt *= image_depth_float.at<float>(round(kp.y), round(kp.x));

        if(unprojected_pt.z != 0)
        {
            //unprojected_pts.push_back(unprojected_pt);
            image_pts_3d.push_back(unprojected_pt);
            image_pts_trunc.push_back(cv::Point2f(kp.x, kp.y));
            //cout << unprojected_pt.x << " " << unprojected_pt.y << " " << unprojected_pt.z << endl;
        }

    }
    //cout << "sizes: " << endl;
    //cout << image_pts_3d.size() << " " << image_pts_trunc.size() << endl;

}

//This gets (new) 3D points in the new camera coordinate frame, and converts them into the first (original) camera coordinate frame
void get_3d_pts_new(cv::Mat cam_pose, cv::Mat image_depth_float, vector<cv::Point2f> image_pts, vector<cv::Point2f> &image_pts_trunc, vector<cv::Point3f> &image_pts_3d)
{
    // Unproject 2D pts
    // Multiply with depth
    float fx = 348.925, fy = 351.135, cx = 339.075, cy = 177.45; // 348.925 351.135 339.075 177.45

    //vector<cv::Point3f> unprojected_pts;


    // Convert pose matrix from 16x1 to 4x4.
    unsigned int pose_mat_idx = 1; // Starts at 1 and not 0 because the 0 index contains some other info.
    cv::Mat cam_pose_4x4 = cv::Mat::zeros(4,4,CV_32F);
    cam_pose_4x4.at<float>(3,3) = 1.0f;
    for(unsigned int row_idx = 0; row_idx < 4; row_idx++)
    {
        for(unsigned int col_idx = 0; col_idx < 4; col_idx++)
        {
            cam_pose_4x4.at<float>(row_idx, col_idx) = cam_pose.at<float>(pose_mat_idx, 0);
            pose_mat_idx++;

        }
    }


    // cout << "Current pose matrix: " << endl;
    // cout << cam_pose_4x4 << endl;

    cv::Mat image_depth_float_;
    image_depth_float.copyTo(image_depth_float_);

    



    for(auto& kp: image_pts) 
    {
        float x = (kp.x - cx) / fx;
        float y = (kp.y - cy) / fy;
        cv::Point3f unprojected_pt(x, y, 1);

        float depth_val = image_depth_float.at<float>(round(kp.y), round(kp.x));
        //cout << "x,y,depth: " << kp.x << " " << kp.y << " " << depth_val << endl;

        cv::circle(image_depth_float_, kp, 3, CV_RGB(255,0,0), 1);

        unprojected_pt *= depth_val;

        if(depth_val > 0.0f)
        {

            cv::Mat pt_3d_curr_camera_frame = cv::Mat::zeros(4,1,CV_32FC1);
            pt_3d_curr_camera_frame.at<float>(0,0) = unprojected_pt.x;
            pt_3d_curr_camera_frame.at<float>(1,0) = unprojected_pt.y;
            pt_3d_curr_camera_frame.at<float>(2,0) = unprojected_pt.z;
            pt_3d_curr_camera_frame.at<float>(3,0) = 1.0f;
            //cout << "3d pt in homog coords:" << pt_3d_curr_camera_frame << endl;

            // // cout << "cam pose size: " << cam_pose_4x4.size() << endl;
            // // cout << "3d pt size: "    << pt_3d_curr_camera_frame.size() << endl;

            cv::Mat pt_3d_orig_camera_frame = cam_pose_4x4*pt_3d_curr_camera_frame;
            //cout << "3d pt after multi: " << pt_3d_orig_camera_frame << endl;
            // // //cv::Mat pt_3d_orig_camera_frame = pt_3d_curr_camera_frame.t()*cam_pose_4x4;

            // // cout << "pt_3d_orig_camera_frame size:" << pt_3d_orig_camera_frame.size() << endl;

            // // cv::Point3f unprojected_pt_orig_camera_frame( 
            // //     pt_3d_orig_camera_frame.at<float>(0,0)/pt_3d_orig_camera_frame.at<float>(3,0), 
            // //     pt_3d_orig_camera_frame.at<float>(1,0)/pt_3d_orig_camera_frame.at<float>(3,0), 
            // //     pt_3d_orig_camera_frame.at<float>(2,0)/pt_3d_orig_camera_frame.at<float>(3,0) );

            cv::Point3f unprojected_pt_orig_camera_frame( 
                pt_3d_orig_camera_frame.at<float>(0,0),
                pt_3d_orig_camera_frame.at<float>(1,0),
                pt_3d_orig_camera_frame.at<float>(2,0) );

            //cout << "3d pt after multi1: " << unprojected_pt_orig_camera_frame << endl;

            //cv::Point3f unprojected_pt_orig_camera_frame = unprojected_pt;



        
            //unprojected_pts.push_back(unprojected_pt);
            image_pts_3d.push_back(unprojected_pt_orig_camera_frame);
            image_pts_trunc.push_back(cv::Point2f(kp.x, kp.y));
            //cout << cv::Point2f(kp.x, kp.y) << endl;
            //cout << unprojected_pt.x << " " << unprojected_pt.y << " " << unprojected_pt.z << endl;
            cv::circle(image_depth_float_, kp, 3, CV_RGB(0,255,255), 1);
        }

    }
    // cv::imshow("image_depth_float with 3d pts", image_depth_float_);
    //cout << "sizes: " << endl;
    //cout << image_pts_3d.size() << " " << image_pts_trunc.size() << endl;

}

void get_pts_right(vector<cv::Point2f> image_pts_left, vector<cv::Point2f> &image_pts_right, cv::Mat image_depth_float, float fx, float baseline)
{
    for(auto& kp: image_pts_left) 
    {
        float depth = image_depth_float.at<float>(round(kp.y), round(kp.x));
        float   disparity =  baseline*fx/(depth);
        //cout << disparity << endl;
        cv::Point2f image_pt_right(kp.x-disparity, kp.y);
        image_pts_right.push_back(image_pt_right);

    }

}

cv::Mat get_pose_from_origin(cv::Mat rvec, cv::Mat tvec)
{
    cv::Mat rotmat;
    cv::Rodrigues(rvec, rotmat);

    cv::Mat tmat = Mat::eye(4,4, CV_32F);
    tvec.copyTo(tmat(Rect(3,0,1,3)));
    rotmat.copyTo(tmat(Rect(0,0,3,3)));

    //return tmat;


//Pose Pose::inv() const {
    cv::Mat transMat = tmat;
    cv::Mat invTrans = Mat::eye(4,4, CV_32F);

    if(!transMat(Rect(0,0,3,3)).empty())
    {
        cv::Mat r = transMat(Rect(0,0,3,3)).t();
        cv::Mat t = -r * transMat(Rect(3,0,1,3));// * (-1);

        
        r.copyTo(invTrans(Rect(0,0,3,3)));
        t.copyTo(invTrans(Rect(3,0,1,3)));
    }


    return invTrans;
//}


}

void write_pts_file(ofstream &output_file_stream, int frame_idx, vector<cv::Point2f> image_pts_left, vector<cv::Point2f> image_pts_right,
    vector<cv::Point3f> image_pts_3d)
{
    // Frame idx, 3D pt idx, image_left_u, image_right_u, image_v, X, Y, Z.
    

    // Iterate thru 3D pts and make sure that they correspond correctly with 2D pts
    for(unsigned int pt_idx = 0; pt_idx < image_pts_left.size(); pt_idx++)
    {
        cv::Point2f image_pt_left  = image_pts_left[pt_idx];
        cv::Point2f image_pt_right = image_pts_right[pt_idx];
        cv::Point3f image_pt_3d    = image_pts_3d[pt_idx];
        if(image_pt_left.x != -1.0f && image_pt_left.y != -1.0f &&
            !isinf(image_pt_left.x)  && !isinf(image_pt_left.y) &&
            !isinf(image_pt_right.x)  && !isinf(image_pt_right.y))
        {
            output_file_stream << frame_idx << " " << pt_idx+1 << " " << image_pt_left.x << " "  << image_pt_right.x << " " 
             << image_pt_left.y << " "  << image_pt_3d.x << " "  << image_pt_3d.y << " " << image_pt_3d.z << endl;

        }


        // cout << "left: " << image_pts_left << endl;
    }
    
    // cout << "size right: " << image_pts_right.size() << endl;
    // cout << "size 3d: " << image_pts_3d.size() << endl;

}

void write_pts_vector(vector<cv::Mat> &pts_vector, int frame_idx, vector<cv::Point2f> image_pts_left, vector<cv::Point2f> image_pts_right,
    vector<cv::Point3f> image_pts_3d)
{
    // Frame idx, 3D pt idx, image_left_u, image_right_u, image_v, X, Y, Z.

    vector<cv::Mat> pts_this_image_vector;
    

    // Iterate thru 3D pts and make sure that they correspond correctly with 2D pts
    for(unsigned int pt_idx = 0; pt_idx < image_pts_left.size(); pt_idx++)
    {
        cv::Point2f image_pt_left  = image_pts_left[pt_idx];
        cv::Point2f image_pt_right = image_pts_right[pt_idx];
        cv::Point3f image_pt_3d    = image_pts_3d[pt_idx];
        
        if(image_pt_left.x != -1.0f && image_pt_left.y != -1.0f &&
            !isinf(image_pt_left.x)  && !isinf(image_pt_left.y) &&
            !isinf(image_pt_right.x)  && !isinf(image_pt_right.y))
        {
            // output_file_stream << frame_idx << " " << pt_idx+1 << " " << image_pt_left.x << " "  << image_pt_right.x << " " 
            //  << image_pt_left.y << " "  << image_pt_3d.x << " "  << image_pt_3d.y << " " << image_pt_3d.z << endl;

            cv::Mat pt_this = cv::Mat::zeros(8, 1, CV_32F);
            pt_this.at<float>(0,0) = frame_idx;
            pt_this.at<float>(1,0) = pt_idx+1;
            pt_this.at<float>(2,0) = image_pt_left.x;
            pt_this.at<float>(3,0) = image_pt_right.x;
            pt_this.at<float>(4,0) = image_pt_left.y;
            pt_this.at<float>(5,0) = image_pt_3d.x;
            pt_this.at<float>(6,0) = image_pt_3d.y;
            pt_this.at<float>(7,0) = image_pt_3d.z;

            pts_this_image_vector.push_back(pt_this);
        }



        // cout << "left: " << image_pts_left << endl;
    }
    for(unsigned int i = 0; i < pts_this_image_vector.size(); i++)
    {
        pts_vector.push_back(pts_this_image_vector[i]);

    }
    // cout << "size right: " << image_pts_right.size() << endl;
    // cout << "size 3d: " << image_pts_3d.size() << endl;

}




void write_poses_file(int image_idx, ofstream &output_file_stream, cv::Mat pose_mat)
{
    output_file_stream << image_idx << " ";
    for(unsigned int row_idx = 0; row_idx < pose_mat.rows; row_idx++)
    {
        for(unsigned int col_idx = 0; col_idx < pose_mat.cols; col_idx++)
        {
            output_file_stream << pose_mat.at<float>(row_idx, col_idx) << " ";
        }
    }
    output_file_stream << endl;
}

void write_poses_vector(int image_idx, vector<cv::Mat> &poses_vector, cv::Mat pose_mat)
{
    cv::Mat this_pose_gtsam = cv::Mat::zeros(16, 1, CV_32F);
    this_pose_gtsam.at<float>(0, 0) = image_idx;

    unsigned int this_pose_idx = 1;
    for(unsigned int row_idx = 0; row_idx < pose_mat.rows; row_idx++)
    {
        for(unsigned int col_idx = 0; col_idx < pose_mat.cols; col_idx++)
        {
            this_pose_gtsam.at<float>(this_pose_idx, 0) = pose_mat.at<float>(row_idx, col_idx);
            this_pose_idx++;
        }
    }
    poses_vector.push_back(this_pose_gtsam);
}



// Remove pts that have gone outside the scope of the image
void flag_bad_pts(vector<cv::Point2f> &image_pts_left, vector<cv::Point2f> &image_pts_right,
    vector<cv::Point3f> &image_pts_3d, int image_width, int image_height)
{
    int margin = 10;//margin around edges of image from where pts are removed
    vector<cv::Point2f> image_pts_left_; // = image_pts_left;
    vector<cv::Point2f> image_pts_right_;// = image_pts_right;
    vector<cv::Point3f> image_pts_3d_;   // = image_pts_3d;

    vector<int> image_pt_indices; // indices for pts within allowed margins of the image

    

    // Get allowed pt indices
    for(unsigned int pt_idx = 0; pt_idx < image_pts_left.size(); pt_idx++)
    {
        cv::Point2f image_pt_left = image_pts_left[pt_idx];
        //cout << image_pt_left << endl;
        // if(image_pt_left.x >=margin && image_pt_left.x < image_width-margin && 
        //     image_pt_left.y >=margin && image_pt_left.y < image_height-margin)
        // {
        //     image_pt_indices.push_back(pt_idx);

        // }

        if(image_pt_left.x <=margin || image_pt_left.x > image_width-margin || 
            image_pt_left.y <=margin || image_pt_left.y > image_height-margin)
        {
            image_pts_left[pt_idx]  = cv::Point2f(-1.0f,-1.0f);
            image_pts_right[pt_idx] = cv::Point2f(-1.0f,-1.0f);
            image_pts_3d[pt_idx]    = cv::Point3f(-1.0f,-1.0f,-1.0f);

            

        }
        
    }

    // // Copy these over to the new list of left, right, 3D pts.
    // for(unsigned int pt_idx = 0; pt_idx < image_pt_indices.size(); pt_idx++)
    // {
    //     int pt_idx_orig = image_pt_indices[pt_idx];
    //     cv::Point2f image_pt_left  = image_pts_left[pt_idx_orig];
    //     cv::Point2f image_pt_right = image_pts_right[pt_idx_orig];
    //     cv::Point3f image_pt_3d    = image_pts_3d[pt_idx_orig];
    //     image_pts_left_.push_back(image_pt_left);
    //     image_pts_right_.push_back(image_pt_right);
    //     image_pts_3d_.push_back(image_pt_3d);

    // }

    // image_pts_left.clear();
    // image_pts_right.clear();
    // image_pts_3d.clear();

    // image_pts_left  = image_pts_left_;
    // image_pts_right = image_pts_right_;
    // image_pts_3d    = image_pts_3d_;

}

// void remove_bad_pts(vector<cv::Point2f> &image_pts_left,
//     vector<cv::Point3f> &image_pts_3d)
// {
//     //int margin = 10;//margin around edges of image from where pts are removed
//     vector<cv::Point2f> image_pts_left_; // = image_pts_left;
//     // vector<cv::Point2f> image_pts_right_;// = image_pts_right;
//     vector<cv::Point3f> image_pts_3d_;   // = image_pts_3d;

//     vector<int> image_pt_indices; // indices for pts within allowed margins of the image


//     // Get allowed pt indices
//     for(unsigned int pt_idx = 0; pt_idx < image_pts_left.size(); pt_idx++)
//     {
//         cv::Point2f image_pt_left = image_pts_left[pt_idx];
//         //cout << image_pt_left << endl;
//         if(image_pt_left.x != -1 && image_pt_left.y != -1)
//         {
//             image_pt_indices.push_back(pt_idx);

//         }

//         // if(image_pt_left.x <=margin || image_pt_left.x > image_width-margin || 
//         //     image_pt_left.y <=margin || image_pt_left.y > image_height-margin)
//         // {
//         //     image_pts_left[pt_idx]  = cv::Point2f(-1.0f,-1.0f);
//         //     image_pts_right[pt_idx] = cv::Point2f(-1.0f,-1.0f);
//         //     image_pts_3d[pt_idx]    = cv::Point3f(-1.0f,-1.0f,-1.0f);

//         // }
//     }

//     // Copy these over to the new list of left, right, 3D pts.
//     for(unsigned int pt_idx = 0; pt_idx < image_pt_indices.size(); pt_idx++)
//     {
//         int pt_idx_orig = image_pt_indices[pt_idx];
//         cv::Point2f image_pt_left  = image_pts_left[pt_idx_orig];
//         // cv::Point2f image_pt_right = image_pts_right[pt_idx_orig];
//         cv::Point3f image_pt_3d    = image_pts_3d[pt_idx_orig];
//         image_pts_left_.push_back(image_pt_left);
//         // image_pts_right_.push_back(image_pt_right);
//         image_pts_3d_.push_back(image_pt_3d);

//     }

//     image_pts_left.clear();
//     // image_pts_right.clear();
//     image_pts_3d.clear();

//     image_pts_left  = image_pts_left_;
//     // image_pts_right = image_pts_right_;
//     image_pts_3d    = image_pts_3d_;

// }

// // modification of remove_bad_pts: checking for points within FOV of camera are checked in the
// // same function as they are removed.
// void remove_bad_pts1(cv::Mat image_left, vector<cv::Point2f> &image_pts_left,
//     vector<cv::Point3f> &image_pts_3d, int ncols, int nrows)
// {
//     int margin = 10;//margin around edges of image from where pts are removed
//     vector<cv::Point2f> image_pts_left_; // = image_pts_left;
//     // vector<cv::Point2f> image_pts_right_;// = image_pts_right;
//     vector<cv::Point3f> image_pts_3d_;   // = image_pts_3d;

//     vector<int> image_pt_indices; // indices for pts within allowed margins of the image

//     int image_width  = ncols;
//     int image_height = nrows;

//     cv::Mat image_left_;
//     image_left.copyTo(image_left_);


//     int num_pts_removed = 0;

//     // Get allowed pt indices
//     for(unsigned int pt_idx = 0; pt_idx < image_pts_left.size(); pt_idx++)
//     {
//         cv::Point2f image_pt_left = image_pts_left[pt_idx];
//         //cout << image_pt_left << endl;
//         if(!(image_pt_left.x <=margin || image_pt_left.x > image_width-margin || 
//              image_pt_left.y <=margin || image_pt_left.y > image_height-margin))
//         {
//             image_pt_indices.push_back(pt_idx);
//             cv::circle(image_left_, image_pt_left, 3, CV_RGB(255,0,0), -1);
            

//         }
//         else
//         {
//             //cout << image_pt_left << endl;
//             num_pts_removed++;
//             cv::circle(image_left_, image_pt_left, 3, CV_RGB(255,255,255), -1);
//         }

//         // if(image_pt_left.x <=margin || image_pt_left.x > image_width-margin || 
//         //     image_pt_left.y <=margin || image_pt_left.y > image_height-margin)
//         // {
//         //     image_pts_left[pt_idx]  = cv::Point2f(-1.0f,-1.0f);
//         //     image_pts_right[pt_idx] = cv::Point2f(-1.0f,-1.0f);
//         //     image_pts_3d[pt_idx]    = cv::Point3f(-1.0f,-1.0f,-1.0f);

//         // }
//     }
//     // cout << "#" << num_pts_removed << " pts removed" << endl;
//     // cv::imshow("new_pts1", image_left_);
//     // cv::waitKey(20);

//     // Copy these over to the new list of left, right, 3D pts.
//     for(unsigned int pt_idx = 0; pt_idx < image_pt_indices.size(); pt_idx++)
//     {
//         int pt_idx_orig = image_pt_indices[pt_idx];
//         cv::Point2f image_pt_left  = image_pts_left[pt_idx_orig];
//         // cv::Point2f image_pt_right = image_pts_right[pt_idx_orig];
//         cv::Point3f image_pt_3d    = image_pts_3d[pt_idx_orig];
//         image_pts_left_.push_back(image_pt_left);
//         // image_pts_right_.push_back(image_pt_right);
//         image_pts_3d_.push_back(image_pt_3d);

//     }

//     image_pts_left.clear();
//     // image_pts_right.clear();
//     image_pts_3d.clear();

//     image_pts_left  = image_pts_left_;
//     // image_pts_right = image_pts_right_;
//     image_pts_3d    = image_pts_3d_;

// }

// Modification of remove_bad_pts1 to include checking for invalid 3D points
void remove_bad_pts1(cv::Mat image_left, vector<cv::Point2f> &image_pts_left,
    vector<cv::Point3f> &image_pts_3d, int ncols, int nrows)
{
    int margin = 10; // Margin around edges of image from where pts are removed
    vector<cv::Point2f> image_pts_left_;
    vector<cv::Point3f> image_pts_3d_;

    int image_width  = ncols;
    int image_height = nrows;

    cv::Mat image_left_;
    image_left.copyTo(image_left_);

    int num_pts_removed = 0;

    // Iterate through all points
    for (unsigned int pt_idx = 0; pt_idx < image_pts_left.size(); pt_idx++)
    {
        cv::Point2f image_pt_left = image_pts_left[pt_idx];
        cv::Point3f image_pt_3d = image_pts_3d[pt_idx];

        // Check if the 2D point is within the image margins
        if (!(image_pt_left.x <= margin || image_pt_left.x > image_width - margin ||
              image_pt_left.y <= margin || image_pt_left.y > image_height - margin))
        {
            // Now also check if the corresponding 3D point is valid
            if (std::isfinite(image_pt_3d.x) && std::isfinite(image_pt_3d.y) && std::isfinite(image_pt_3d.z)
                && cv::norm(image_pt_3d) > 1e-6)
            {
                // Point is valid; keep it
                image_pts_left_.push_back(image_pt_left);
                image_pts_3d_.push_back(image_pt_3d);
                cv::circle(image_left_, image_pt_left, 3, CV_RGB(255, 0, 0), -1);
            }
            else
            {
                // 3D point is invalid; remove it
                num_pts_removed++;
                cv::circle(image_left_, image_pt_left, 3, CV_RGB(255, 255, 255), -1);
            }
        }
        else
        {
            // 2D point is outside the image margins; remove it
            num_pts_removed++;
            cv::circle(image_left_, image_pt_left, 3, CV_RGB(255, 255, 255), -1);
        }
    }

    // Update the original vectors with the filtered ones
    image_pts_left = image_pts_left_;
    image_pts_3d = image_pts_3d_;
}

// Gets mask for initializing new points (they should be away from old points)
void get_pts_mask(cv::Mat& mask_image, vector<cv::Point2f> image_pts_left, int mask_circle_radius)
{
    mask_image*= 255;
    for(unsigned int i = 0; i < image_pts_left.size(); i++)
    {
        cv::Point2d pt_this = image_pts_left[i];
        cv::circle(mask_image, pt_this, mask_circle_radius, CV_RGB(0,0,0), -1);//5
    }

    // cv::imshow("mask_image", mask_image);

}

// *************** Aruco marker functionality *********************************

void draw_aruco_markers(cv::Mat image_left, std::vector<std::vector<cv::Point2f>> aruco_corners, std::vector<int> aruco_ids)
{
    cv::Mat image_left_ = cv::Mat(image_left.size(), image_left.type());
    image_left.copyTo(image_left_);

    cv::aruco::drawDetectedMarkers(image_left_, aruco_corners, aruco_ids);
    cv::imshow("aruco detections", image_left_);
    cv::waitKey(1);


}

void draw_aruco_markers_pose(cv::Mat image_left, std::vector<int> aruco_ids, cv::Mat cam_calib_mat,
                cv::Mat cam_distortion_mat, std::vector<cv::Vec3d> aruco_rvecs, std::vector<cv::Vec3d> aruco_tvecs,
                float aruco_size)
{
    cv::Mat image_left_ = cv::Mat(image_left.size(), image_left.type());
    image_left.copyTo(image_left_);

    for(int i = 0; i < aruco_ids.size(); i++)
    {
        cv::drawFrameAxes(image_left_, cam_calib_mat, cam_distortion_mat, aruco_rvecs[i], aruco_tvecs[i], aruco_size);
    }
    cv::imshow("aruco poses", image_left_);
    cv::waitKey(1);

}


void aruco_convert_rtvecs_pose_matrix(cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat& pose_mat)
{

    cout << "closest aruco rvec: " << rvec << endl;
    cout << "closest aruco tvec: " << tvec << endl;
    cv::Mat rot_mat(3, 3, CV_32FC1);

    cv::Rodrigues(rvec, rot_mat);
    cout << "closest aruco rot_mat: " << rot_mat << endl;

    // 4x4 pose tranformation matrix.
    pose_mat = cv::Mat::eye(4, 4, CV_32FC1);

    // Assign rotation to pose tranformation matrix
    // cv::Mat rot_roi(pose_mat(cv::Rect(0,0,3,3)));
    // rot_mat.copyTo(rot_roi);
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            //cout << "(" << i << ", " << j << "): " << (float)rot_mat.at<double>(i,j) << endl;
            pose_mat.at<float>(i,j) = (float)rot_mat.at<double>(i,j);

        }
    }

    // Copy the translation vector to the matrix
    pose_mat.at<float>(0,3) = tvec[0];
    pose_mat.at<float>(1,3) = tvec[1];
    pose_mat.at<float>(2,3) = tvec[2];

    cout << "closest aruco pose_mat: " << pose_mat << endl;

}

int aruco_get_closest_aruco_id(std::vector<int> ids, std::vector<Vec3d> tvecs)
{

    float dist, distance = INT_MAX;
    int order = 0;
    for(int i = 0; i < ids.size(); i++)
    {
        dist = sqrt(tvecs[i][0]*tvecs[i][0]  +
                    tvecs[i][1]*tvecs[i][1] +
                    tvecs[i][2]*tvecs[i][2]); // TODO: replace with ^2.
        if(dist < distance)
        {
            order = i;
            distance = dist;
        }
    }
    return order;

}

void aruco_get_global_pose(cv::Mat closest_aruco_pose_tmat, cv::Mat tmat_origin2camera,
            cv::Mat& closest_aruco_global_pose_tmat)
{
    cout << "closest_aruco_pose_tmat :" << closest_aruco_pose_tmat << endl;
    cv::Mat camera_to_closest_aruco_pose_tmat = closest_aruco_pose_tmat;//closest_aruco_pose_tmat.inv(); // TODO: replace inverse with transpose trick for easier compute.
    cout << "closest_aruco_pose_tmat inv: " << camera_to_closest_aruco_pose_tmat << endl;

    // Convert pose matrix from 16x1 to 4x4 if using GTSAM version of pose
    // unsigned int tmat_origin2camera_idx = 1; // Starts at 1 and not 0 because the 0 index contains some other info.
    // cv::Mat tmat_origin2camera_4x4 = cv::Mat::zeros(4,4,CV_64FC1);
    // tmat_origin2camera_4x4.at<float>(3,3) = 1.0f;
    // for(unsigned int row_idx = 0; row_idx < 4; row_idx++)
    // {
    //     for(unsigned int col_idx = 0; col_idx < 4; col_idx++)
    //     {
    //         tmat_origin2camera_4x4.at<float>(row_idx, col_idx) = tmat_origin2camera.at<float>(tmat_origin2camera_idx, 0);
    //         tmat_origin2camera_idx++;

    //     }
    // }

    // cv::Mat tmat_origin2camera_4x4 = cv::Mat::zeros(4,4,CV_32FC1);
    
    // for(unsigned int row_idx = 0; row_idx < 4; row_idx++)
    // {
    //     for(unsigned int col_idx = 0; col_idx < 4; col_idx++)
    //     {
    //         tmat_origin2camera_4x4.at<double>(row_idx, col_idx) = tmat_origin2camera.at<float>(row_idx, col_idx);
    //     }
    // }

    cout << "tmat_origin2camera: " << tmat_origin2camera << endl;
    cout << "camera_to_closest_aruco_pose_tmat: " << camera_to_closest_aruco_pose_tmat << endl;
    closest_aruco_global_pose_tmat = tmat_origin2camera*camera_to_closest_aruco_pose_tmat;//camera_to_closest_aruco_pose_tmat*tmat_origin2camera; 
    cout << "closest aruco pose wrt origin: " << closest_aruco_global_pose_tmat << endl;


}

void initialize_tracking(cv::Mat& cam_pose_curr, cv::Mat& image_left_grey, cv::Mat& image_depth_float,
                         vector<cv::Point2f>& image_pts_curr, vector<cv::Point3f>& image_pts_3d_curr,
                         cv::Size subPixWinSize, cv::TermCriteria termcrit, int max_count)
{
    // Create a mask to ignore the bottom 20% of the image
    cv::Mat mask = cv::Mat::ones(image_left_grey.size(), CV_8U);
    int ignore_height = image_left_grey.rows * 0.2;
    mask(cv::Rect(0, image_left_grey.rows - ignore_height, image_left_grey.cols, ignore_height)) = 0;

    // Detect features using the mask
    cv::goodFeaturesToTrack(image_left_grey, image_pts_curr, max_count, 0.01, 10, mask, 3, false, 0.04);
    cv::cornerSubPix(image_left_grey, image_pts_curr, subPixWinSize, Size(-1, -1), termcrit);

    // Get 3D points
    vector<cv::Point2f> image_pts_trunc;
    image_pts_3d_curr.clear();
    //get_3d_pts(image_depth_float, image_pts_curr, image_pts_trunc, image_pts_3d_curr);

    get_3d_pts_new(cam_pose_curr, image_depth_float, image_pts_curr, image_pts_trunc, image_pts_3d_curr);

    // Update the current points with valid ones
    image_pts_curr = image_pts_trunc;
}



// *************** Aruco marker functionality *********************************

int main(int argc, char ** argv){

    // Images and points (2D&3D) 
    string images_path = "/home/jay/Documents/data/corridorData/04/";//"/home/jay/Documents/data/b234/06/";// "/home/jay/Documents/data/bcorridors/04/";
    // string output_file1_name = "stereo_factors_large.txt";
    // string output_file2_name = "camera_poses_large.txt";
    // ofstream output_file1_stream, output_file2_stream;
    // output_file1_stream.open(output_file1_name);
    // output_file2_stream.open(output_file2_name);

    cv::Mat image_left, image_depth, image_left_grey, image_left_prev, image_depth_float;
    cv::Mat image_right, image_right_grey;
    vector<cv::Point2f> image_pts[2];
    vector<cv::Point2f> image_pts_new;
    vector<cv::Point3f> image_pts_3d;
    vector<cv::Point3f> image_pts_3d_new;

    vector<cv::Point2f> image_pts_right; //optional right point for GTSAM.

    vector<cv::Point2f> image_pts_trunc;
    vector<cv::Point2f> image_pts_trunc_new;

    const int start_image_idx = 42;//246;//228;
    const int end_image_idx   = 1371;//867;//300;//258;//245;

    // A selection of start and end image indices in the Corridor sequence. 
    //660;//620;//570;//530;//500;//450;//380;//320;//290;//210;//100;//228
    //680;//640;//590;//550;//520;//470;//400;//340;//310;//235;//120;//245

    // LKT parameters
    const int MAX_COUNT = 500; // max num of points added from beginning.
    const int MAX_COUNT_NEW = 50; // max num of new points added in each frame.
    const int MIN_PTS_PNP = 10; // min num of points for PnP.
    cv::TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    cv::Size subPixWinSize(10,10);
    cv::Size winSize(31,31);

    // Camera calibration
    const double fx = 348.925;
    const double fy = 351.135;
    const double s  = 0.0;
    const double u0 = 339.075;
    const double v0 = 177.45;
    const float baseline = 0.120;//metres

    cv::Mat cam_calib_mat(3, 3, CV_32F);
    cam_calib_mat.at<float>(0,0) = fx;
    cam_calib_mat.at<float>(1,1) = fy;
    cam_calib_mat.at<float>(0,2) = u0;
    cam_calib_mat.at<float>(1,2) = v0;

    cv::Mat cam_distortion_mat = cv::Mat::zeros(8, 1, CV_64F);
    

    //Pangolin
    pangolinViewer pgViewer;
    std::thread viewer_thread(&pangolinViewer::run, &pgViewer);

// #ifdef GTSAM_INC
    // GTSAM
    // gtsam::Values initial_estimate;
    // gtsam::NonlinearFactorGraph graph;
    // const gtsam::noiseModel::Isotropic::shared_ptr noise_model = gtsam::noiseModel::Isotropic::Sigma(3,1);
    // const gtsam::Cal3_S2Stereo::shared_ptr K(new gtsam::Cal3_S2Stereo(fx,fy,s,u0,v0,baseline));
    vector<cv::Mat> poses_gtsam; // seems to be used for storing historic pose information even outside of GTSAM.
    // vector<cv::Mat> pts_gtsam;
// #endif

    cv::Mat tmat_origin2camera;

    // Aruco marker detection
    // Note use of opencv smart pointer
    cv::Ptr<cv::aruco::Dictionary> aruco_dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000); 
    float aruco_size = 0.18; // size of side of Aruco
    map<int, cv::Mat> aruco_map; // contains detected Aruco markers.

    // Aruco-map saved file
    string aruco_file_name = "aruco-map.json";
    cv::FileStorage fs(aruco_file_name, cv::FileStorage::WRITE);



    
    
    for(int image_idx = start_image_idx; image_idx < end_image_idx; image_idx++)
    {
        cout << "Image# " << image_idx << endl;
        //load_images(images_path, image_idx, image_left, image_right, image_depth);

        try {
            // Attempt to load the images
            load_images(images_path, image_idx, image_left, image_right, image_depth);
        } catch (const std::exception& e) {
            // Catch any exceptions thrown by load_images and handle them
            std::cerr << "Error: Failed to load images at index " << image_idx 
                    << ". Exception: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);  // Exit the program with failure status
        }

        cv::cvtColor(image_left, image_left_grey, COLOR_BGR2GRAY);
        cv::cvtColor(image_right, image_right_grey, COLOR_BGR2GRAY);
        
        // //cout << "NUM OF PTS IN IMAGE_PTS 0 b4 LKT " << image_pts[0].size() << endl;

        image_depth_float  = convert_depth(image_depth);
        // cv::imshow("image_depth", image_depth);
        // cv::imshow("image_depth_float_", image_depth_float);
        // Initialize 2D image points -> get depth -> get 3D world points.
        if(image_idx == start_image_idx)//image_pts[1].empty())
        {
            // cv::goodFeaturesToTrack(image_left_grey, image_pts[1], MAX_COUNT, 0.01, 10, Mat(),
            //  3, 3, 0, 0.04);

            cv::Mat mask = cv::Mat::ones(image_left_grey.size(), CV_8U); // Create a mask of ones (use the whole image)
            int ignore_height = image_left_grey.rows * 0.2; // 20% of the image height
            cv::Rect roi(0, 0, image_left_grey.cols, image_left_grey.rows - ignore_height); // Define the ROI as top 80%

            // Set the bottom 20% of the mask to zero (ignore region)
            mask(cv::Rect(0, image_left_grey.rows - ignore_height, image_left_grey.cols, ignore_height)) = 0;


            // cv::goodFeaturesToTrack(image_left_grey, image_pts[1], MAX_COUNT, 0.01, 10, Mat(),
            //  3, false, 0.04);

            // Detect features only in the top 80% of the image using the mask
            cv::goodFeaturesToTrack(image_left_grey, image_pts[1], MAX_COUNT, 0.01, 10, mask, 3, false, 0.04);

            cv::cornerSubPix(image_left_grey, image_pts[1], subPixWinSize, Size(-1, -1), termcrit);
        
            // Get depth values for 2D image points and convert to 3D
            //image_depth_float  = convert_depth(image_depth);
            get_3d_pts(image_depth_float, image_pts[1], image_pts_trunc, image_pts_3d);
            //image_pts[1].swap(image_pts_trunc);

            cout << image_pts_3d << endl;

            pgViewer.add_pts_3d(image_pts_3d, image_idx-start_image_idx);

            image_pts[1].clear();
            for(unsigned int i = 0; i < image_pts_trunc.size(); i++)
            {
                 cv::Point2f this_pt_2d = image_pts_trunc[i];
                 image_pts[1].push_back(this_pt_2d);
            }


            draw_pts(image_left_grey, image_pts[1], "image_pts_left");

        }
        // Do LKT (track from pts0 to pts1) and PnP to get pose of current image wrt prev image
        if(!image_pts[0].empty())//image_idx > start_image_idx)
        {
            // cv::imshow("left image curr", image_left_grey);
            // cv::imshow("left image prev", image_left_prev);
            //cv::waitKey(5000);
#ifdef ADD_NEW_PTS
            //cout << "Adding new pts for tracking..." << endl;
            // Add new points for tracking
            // 1. Get mask image. 2. Get pts within mask. 3. Get 3D pts (unprojection).
            // cv::Mat mask_image = cv::Mat::ones(image_left_grey.size(), CV_8UC1);
            // int mask_circle_radius = 25;
            // get_pts_mask(mask_image, image_pts[0], mask_circle_radius);
            //cv::goodFeaturesToTrack(image_left_grey, image_pts_new, MAX_COUNT_NEW, 0.01, 10, mask_image, 3, 3, 0, 0.04);//image_pts_new

            // Step 1: Create a mask with the bottom 20% of the image set to 0
            cv::Mat base_mask = cv::Mat::ones(image_left_grey.size(), CV_8UC1); // Start with a mask of ones (use the whole image)
            int ignore_height = image_left_grey.rows * 0.2; // Calculate 20% of the image height
            base_mask(cv::Rect(0, image_left_grey.rows - ignore_height, image_left_grey.cols, ignore_height)) = 0; // Set the bottom 20% to 0

            // Step 2: Apply existing mask logic around tracked points
            int mask_circle_radius = 5;//25
            cv::Mat mask_image = base_mask.clone(); // Clone the base mask to add circular masks
            get_pts_mask(mask_image, image_pts[0], mask_circle_radius); // Modify mask_image with circles around the tracked points


            cv::goodFeaturesToTrack(image_left_grey, image_pts_new, MAX_COUNT_NEW, 0.01, 10, mask_image, 3, false, 0.04);//image_pts_new

            if (image_pts_new.size() > 0 && image_idx - start_image_idx > 2) // wait at least 2 frames (with good pose) before adding new points
            {
                // cout << "new pts #########################################################" << endl;
                // cout << image_pts_new << endl;
                // cout << "#########################################################" << endl;
                cv::cornerSubPix(image_left_grey, image_pts_new, subPixWinSize, Size(-1, -1), termcrit);
                // cout << "new pts after sub pixel #########################################################" << endl;
                // cout << image_pts_new << endl;
                // cout << "#########################################################" << endl;
                draw_pts(image_left_grey, image_pts_new, "new pts");

                //get_3d_pts(image_depth_float, image_pts_new, image_pts_trunc_new, image_pts_3d_new);
                // Get the current pose of the camera.
                //cv::Mat cam_pose_curr = cv::Mat::zeros(4,4,CV_32FC1);
                cv::Mat cam_pose_curr = cv::Mat::zeros(16,1,CV_32FC1);

                if(!poses_gtsam.empty())
                {
                    cam_pose_curr = poses_gtsam.back();
                    // cout << "pose_gtsam: " << poses_gtsam.back() << endl;
                }

                image_pts_3d_new.clear();
                image_pts_trunc_new.clear();

                get_3d_pts_new(cam_pose_curr, image_depth_float, image_pts_new, image_pts_trunc_new, image_pts_3d_new);
                // pgViewer.add_pts_3d(image_pts_3d_new);

                //cout << image_pts_3d_new << endl;
                // cout << "new pts after 3d estimation#########################################################" << endl;
                // cout << image_pts_new << endl;
                // cout << "#########################################################" << endl;
                

                image_pts_new.clear();
                for(unsigned int i = 0; i < image_pts_trunc_new.size(); i++)
                {
                     cv::Point2f this_pt_2d = image_pts_trunc_new[i];
                     image_pts_new.push_back(this_pt_2d);
                }

                // cout << "new pts after 3d estimation and cleaning using truncated pts#########################################################" << endl;
                // cout << image_pts_new << endl;
                // cout << "#########################################################" << endl;

                // Add new points (2D and 3D) to old set.
                for(unsigned int i = 0; i < image_pts_new.size(); i++)
                {
                     cv::Point2f this_pt_2d = image_pts_new[i];
                     image_pts[0].push_back(this_pt_2d);
                }

                // cout << "image_pts[0]#########################################################" << endl;
                // cout << image_pts[0] << endl;
                // cout << "#########################################################" << endl;

                for(unsigned int i = 0; i < image_pts_3d_new.size(); i++)
                {
                     cv::Point3f this_pt_3d = image_pts_3d_new[i];
                     image_pts_3d.push_back(this_pt_3d);
                }
                
                pgViewer.add_pts_3d(image_pts_3d, image_idx-start_image_idx);
                cv::waitKey(1);
            }
#endif


            vector<uchar> status;
            vector<float> err;

            cv::calcOpticalFlowPyrLK(image_left_prev, image_left_grey, image_pts[0], image_pts[1], 
                status, err, winSize, 3, termcrit, 0, 0.001);

            // ----------  Seg-fault correction -------

            // Add this code to filter points based on status
            std::vector<cv::Point2f> image_pts_0_good;
            std::vector<cv::Point2f> image_pts_1_good;
            std::vector<cv::Point3f> image_pts_3d_good;

            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                {
                    image_pts_0_good.push_back(image_pts[0][i]);
                    image_pts_1_good.push_back(image_pts[1][i]);
                    image_pts_3d_good.push_back(image_pts_3d[i]);
                }
            }

            // Replace the old vectors with the filtered ones
            image_pts[0] = image_pts_0_good;
            image_pts[1] = image_pts_1_good;
            image_pts_3d = image_pts_3d_good;
            // ----------------------------------------

            draw_pts(image_left_grey, image_pts[1], "lkt");


        }
        if(!image_pts[0].empty() && !image_pts[1].empty() && !image_pts_3d.empty())
        {
            // ************************
            // *** PnP happens here ***
            // ************************

            //cout << "Calculating Pose..." << endl;
            //cout << "3d pts size: " << image_pts_3d.size() << endl;
            //cout << "2d pts size: " << image_pts[1].size() << endl;
            //vector<double> dist_coeff; //empty
            cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);
            cv::Mat rvec, tvec;

            vector<cv::Point2f> image_pts_left_trunc = image_pts[1];
            vector<cv::Point3f> image_pts_3d_trunc   = image_pts_3d;

            //remove_bad_pts(image_pts_left_trunc, image_pts_3d_trunc);

            // cout << "number of total 2D pts b4 removing bad pts: " << image_pts_left_trunc.size() << endl;
            // cout << "number of total 3D pts b4 removing bad pts: " << image_pts_3d_trunc.size() << endl;

            remove_bad_pts1(image_left_grey, image_pts_left_trunc,image_pts_3d_trunc, image_left.cols, image_left.rows);

            // cout << "number of total 2D pts after removing bad pts: " << image_pts_left_trunc.size() << endl;
            // cout << "number of total 3D pts after removing bad pts: " << image_pts_3d_trunc.size() << endl;

            //cv::solvePnPRansac(image_pts_3d, image_pts[1], cam_calib_mat, dist_coeffs, rvec, tvec);//, false, 100, 8.0);
            //cv::solvePnPRansac(image_pts_3d_trunc, image_pts_left_trunc, cam_calib_mat, dist_coeffs, rvec, tvec);//, false, 100, 8.0);
            // Add this check before calling cv::solvePnPRansac
            cout << "Num of tracked 2D points: " << image_pts_left_trunc.size() << " Num of 3D pts: " << image_pts_3d_trunc.size() << endl;
            if (image_pts_left_trunc.size() >= MIN_PTS_PNP && image_pts_3d_trunc.size() >= MIN_PTS_PNP && !cam_calib_mat.empty())
            {   
                try
                {
                    //cv::solvePnPRansac(image_pts_3d_trunc, image_pts_left_trunc, cam_calib_mat, dist_coeffs, rvec, tvec);
                    cv::solvePnPRansac(image_pts_3d_trunc, image_pts_left_trunc, cam_calib_mat, dist_coeffs, rvec, tvec);

                }
                catch(const std::exception& e)
                {
                    std::cerr << "Exception during solvePnPRansac at image idx: " << image_idx << std::endl;
                    std::cerr << e.what() << std::endl;
                    // Handle exception, e.g., re-initialize tracking
                    image_pts[0].clear();
                    image_pts[1].clear();
                    image_pts_3d.clear();

                    cv::Mat cam_pose_curr = cv::Mat::zeros(16,1,CV_32FC1);

                    if(!poses_gtsam.empty())
                    {
                        cam_pose_curr = poses_gtsam.back();
                        cout << "pose_gtsam: " << poses_gtsam.back() << endl;
                    }

                    initialize_tracking(cam_pose_curr, image_left_grey, image_depth_float, image_pts[1], image_pts_3d, subPixWinSize, termcrit, MAX_COUNT);
                    swap(image_pts[1], image_pts[0]);
                    cv::swap(image_left_prev, image_left_grey);

                    continue;
                }
                
            }
            else
            {
                // Handle the case when there are not enough valid points
                std::cerr << "Not enough points for solvePnPRansac at image idx: " << image_idx << std::endl;
                // Optionally, you can re-initialize tracking or skip this frame
                // Re-initialize tracking
                image_pts[0].clear();
                image_pts[1].clear();
                image_pts_3d.clear();

                cv::Mat cam_pose_curr = cv::Mat::zeros(16,1,CV_32FC1);

                if(!poses_gtsam.empty())
                {
                    cam_pose_curr = poses_gtsam.back();
                    // cout << "pose_gtsam: " << poses_gtsam.back() << endl;
                }

                initialize_tracking(cam_pose_curr, image_left_grey, image_depth_float, image_pts[1], image_pts_3d, subPixWinSize, termcrit, MAX_COUNT);
                swap(image_pts[1], image_pts[0]);
                cv::swap(image_left_prev, image_left_grey);

                continue;

                // You might also consider adding new points here
                // ...
            }

            tmat_origin2camera = get_pose_from_origin(rvec, tvec);
            cv::Mat translation = tmat_origin2camera(Rect(3,0,1,3));


            //pgViewer.add_path_point(cv::Point3f(tvec));
            pgViewer.add_path_point(cv::Point3f(translation));

            pgViewer.add_path_pose(tmat_origin2camera);


            pgViewer.clear_old_landmarks(image_idx-start_image_idx);


            //cout << "rvec" << rvec << endl;
            // cout << "tvec: " << tvec << endl;
            // cout << "translation: " << translation << endl;

            //output_file2_stream << image_idx << " " << tmat_origin2camera << endl;
            //write_poses_file(image_idx-start_image_idx, output_file2_stream, tmat_origin2camera);

// #ifdef GTSAM
            write_poses_vector(image_idx-start_image_idx, poses_gtsam, tmat_origin2camera);
            //insert_poses_gtsam_graph(initial_estimate, image_idx-start_image_idx+1, tmat_origin2camera);
// #endif

        }



        cout << "image idx: " << image_idx << endl;
        // Swap images and points between current and prev images
        swap(image_pts[1], image_pts[0]); // moving image points from 0 to 1 and 1 to 0
        //image_left_prev = image_left_grey;
        cv::swap(image_left_prev, image_left_grey);


    }
    // Signal the viewer thread to exit (you may need to implement this)
    pgViewer.stop();  // Assuming you have a function to stop the viewer

    // Join the viewer thread to ensure proper cleanup
    if (viewer_thread.joinable()) {
        viewer_thread.join();
    }

    cout << "Reached end of program, exiting..." << endl;


    return 0;
}
