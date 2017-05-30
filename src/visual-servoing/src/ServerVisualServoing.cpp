#include "ServerVisualServoing.h"

#include <cmath>
#include <iostream>

#include <iCub/ctrl/minJerkCtrl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <yarp/math/Math.h>
#include <yarp/math/SVD.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/Time.h>

using namespace yarp::dev;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;
using namespace iCub::ctrl;


bool ServerVisualServoing::configure(ResourceFinder &rf)
{
    robot_name_ = rf.find("robot").asString();
    if (robot_name_.empty())
    {
        yError() << "Robot name not provided! Closing.";
        return false;
    }


    if (!port_pose_left_in_.open("/visual-servoing/pose/left:i"))
    {
        yError() << "Could not open /visual-servoing/pose/left:i port! Closing.";
        return false;
    }
    if (!port_pose_right_in_.open("/visual-servoing/pose/right:i"))
    {
        yError() << "Could not open /visual-servoing/pose/right:i port! Closing.";
        return false;
    }


    if (!port_image_left_in_.open("/visual-servoing/cam_left/img:i"))
    {
        yError() << "Could not open /visual-servoing/cam_left/img:i port! Closing.";
        return false;
    }
    if (!port_image_left_out_.open("/visual-servoing/cam_left/img:o"))
    {
        yError() << "Could not open /visual-servoing/cam_left/img:o port! Closing.";
        return false;
    }
    if (!port_click_left_.open("/visual-servoing/cam_left/click:i"))
    {
        yError() << "Could not open /visual-servoing/cam_left/click:in port! Closing.";
        return false;
    }


    if (!port_image_right_in_.open("/visual-servoing/cam_right/img:i"))
    {
        yError() << "Could not open /visual-servoing/cam_right/img:i port! Closing.";
        return false;
    }
    if (!port_image_right_out_.open("/visual-servoing/cam_right/img:o"))
    {
        yError() << "Could not open /visual-servoing/cam_right/img:o port! Closing.";
        return false;
    }
    if (!port_click_right_.open("/visual-servoing/cam_right/click:i"))
    {
        yError() << "Could not open /visual-servoing/cam_right/click:i port! Closing.";
        return false;
    }


    if (!setGazeController()) return false;

    if (!setRightArmCartesianController()) return false;


    Bottle btl_cam_info;
    itf_gaze_->getInfo(btl_cam_info);
    yInfo() << "[CAM INFO]" << btl_cam_info.toString();
    Bottle* cam_left_info = btl_cam_info.findGroup("camera_intrinsics_left").get(1).asList();
    Bottle* cam_right_info = btl_cam_info.findGroup("camera_intrinsics_right").get(1).asList();

    float left_fx = static_cast<float>(cam_left_info->get(0).asDouble());
    float left_cx = static_cast<float>(cam_left_info->get(2).asDouble());
    float left_fy = static_cast<float>(cam_left_info->get(5).asDouble());
    float left_cy = static_cast<float>(cam_left_info->get(6).asDouble());

    yInfo() << "[CAM]" << "Left camera:";
    yInfo() << "[CAM]" << " - fx:"     << left_fx;
    yInfo() << "[CAM]" << " - fy:"     << left_fy;
    yInfo() << "[CAM]" << " - cx:"     << left_cx;
    yInfo() << "[CAM]" << " - cy:"     << left_cy;

    l_proj_       = zeros(3, 4);
    l_proj_(0, 0) = left_fx;
    l_proj_(0, 2) = left_cx;
    l_proj_(1, 1) = left_fy;
    l_proj_(1, 2) = left_cy;
    l_proj_(2, 2) = 1.0;

    yInfo() << "l_proj_ =\n" << l_proj_.toString();

    float right_fx = static_cast<float>(cam_right_info->get(0).asDouble());
    float right_cx = static_cast<float>(cam_right_info->get(2).asDouble());
    float right_fy = static_cast<float>(cam_right_info->get(5).asDouble());
    float right_cy = static_cast<float>(cam_right_info->get(6).asDouble());

    yInfo() << "[CAM]" << "Right camera:";
    yInfo() << "[CAM]" << " - fx:"     << right_fx;
    yInfo() << "[CAM]" << " - fy:"     << right_fy;
    yInfo() << "[CAM]" << " - cx:"     << right_cx;
    yInfo() << "[CAM]" << " - cy:"     << right_cy;

    r_proj_       = zeros(3, 4);
    r_proj_(0, 0) = right_fx;
    r_proj_(0, 2) = right_cx;
    r_proj_(1, 1) = right_fy;
    r_proj_(1, 2) = right_cy;
    r_proj_(2, 2) = 1.0;

    yInfo() << "r_proj_ =\n" << r_proj_.toString();


    Vector left_eye_x;
    Vector left_eye_o;
    itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

    Vector right_eye_x;
    Vector right_eye_o;
    itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

    yInfo() << "left_eye_o =" << left_eye_o.toString();
    yInfo() << "left_eye_x =" << left_eye_x.toString();
    yInfo() << "right_eye_o =" << right_eye_o.toString();
    yInfo() << "right_eye_x =" << right_eye_x.toString();


    l_H_eye_to_r_ = axis2dcm(left_eye_o);
    left_eye_x.push_back(1.0);
    l_H_eye_to_r_.setCol(3, left_eye_x);
    l_H_r_to_eye_ = SE3inv(l_H_eye_to_r_);

    r_H_eye_to_r_ = axis2dcm(right_eye_o);
    right_eye_x.push_back(1.0);
    r_H_eye_to_r_.setCol(3, right_eye_x);
    r_H_r_to_eye_ = SE3inv(r_H_eye_to_r_);

    yInfo() << "l_H_r_to_eye_ =\n" << l_H_r_to_eye_.toString();
    yInfo() << "r_H_r_to_eye_ =\n" << r_H_r_to_eye_.toString();

    l_H_r_to_cam_ = l_proj_ * l_H_r_to_eye_;
    r_H_r_to_cam_ = r_proj_ * r_H_r_to_eye_;

    yInfo() << "l_H_r_to_cam_ =\n" << l_H_r_to_cam_.toString();
    yInfo() << "r_H_r_to_cam_ =\n" << r_H_r_to_cam_.toString();


    if (!setCommandPort())
    {
        yError() << "Could not open /visual-servoing/cmd:i port! Closing.";
        return false;
    }

    return true;
}


bool ServerVisualServoing::updateModule()
{
    while (!shall_go_);

    if (should_stop_) return false;

    yInfo() << "RUNNING!\n";

    Vector  est_copy_left(6);
    Vector  est_copy_right(6);
    Vector* estimates;

    /* Restoring cartesian and gaze context */
    itf_rightarm_cart_->restoreContext(ctx_cart_);
    itf_gaze_->restoreContext(ctx_gaze_);

    Vector l_px0_position    = zeros(2);
    Vector l_px1_position    = zeros(2);
    Vector l_px2_position    = zeros(2);
    Vector l_px3_position    = zeros(2);
    Vector l_px0_orientation = zeros(2);
    Vector l_px1_orientation = zeros(2);
    Vector l_px2_orientation = zeros(2);
    Vector l_px3_orientation = zeros(2);
    Vector r_px0_position    = zeros(2);
    Vector r_px1_position    = zeros(2);
    Vector r_px2_position    = zeros(2);
    Vector r_px3_position    = zeros(2);
    Vector r_px0_orientation = zeros(2);
    Vector r_px1_orientation = zeros(2);
    Vector r_px2_orientation = zeros(2);
    Vector r_px3_orientation = zeros(2);

    Vector px_ee_cur_position    = zeros(12);
    Matrix jacobian_position     = zeros(12, 6);
    Vector px_ee_cur_orientation = zeros(12);
    Matrix jacobian_orientation  = zeros(12, 6);

    bool is_vs_done = false;
    while (!should_stop_ && !is_vs_done)
    {
        /* Get the initial end-effector pose from left eye view */
        estimates = port_pose_left_in_.read(true);
        yInfo() << "Got [" << estimates->toString() << "] from left eye particle filter.";
        if (estimates->length() == 7)
        {
            est_copy_left = estimates->subVector(0, 5);
            float ang     = (*estimates)[6];

            est_copy_left[3] *= ang;
            est_copy_left[4] *= ang;
            est_copy_left[5] *= ang;
        }
        else
            est_copy_left = *estimates;

        /* Get the initial end-effector pose from right eye view */
        estimates = port_pose_right_in_.read(true);
        yInfo() << "Got [" << estimates->toString() << "] from right eye particle filter.";
        if (estimates->length() == 7)
        {
            est_copy_right = estimates->subVector(0, 5);
            float ang      = (*estimates)[6];

            est_copy_right[3] *= ang;
            est_copy_right[4] *= ang;
            est_copy_right[5] *= ang;
        }
        else
            est_copy_right = *estimates;

        yInfo() << "EE estimates left  = ["  << est_copy_left.toString()  << "]";
        yInfo() << "EE estimates right = [" << est_copy_right.toString() << "]";

        /* SIM */
//        /* Simulate reaching starting from the initial position */
//        /* Comment any previous write on variable 'estimates' */
//
//        /* Evaluate the new orientation vector from axis-angle representation */
//        /* The following code is a copy of the setTaskVelocities() code */
//        Vector l_o = getAxisAngle(est_copy_left.subVector(3, 5));
//        Matrix l_R = axis2dcm(l_o);
//        Vector r_o = getAxisAngle(est_copy_right.subVector(3, 5));
//        Matrix r_R = axis2dcm(r_o);
//
//        vel_o[3] *= Ts_;
//        l_R = axis2dcm(vel_o) * l_R;
//        r_R = axis2dcm(vel_o) * r_R;
//
//        Vector l_new_o = dcm2axis(l_R);
//        double l_ang = l_new_o(3);
//        l_new_o.pop_back();
//        l_new_o *= l_ang;
//
//        Vector r_new_o = dcm2axis(r_R);
//        double r_ang = r_new_o(3);
//        r_new_o.pop_back();
//        r_new_o *= r_ang;
//
//        est_copy_left.setSubvector(0, est_copy_left.subVector(0, 2)  + vel_x * Ts_);
//        est_copy_left.setSubvector(3, l_new_o);
//        est_copy_right.setSubvector(0, est_copy_right.subVector(0, 2)  + vel_x * Ts_);
//        est_copy_right.setSubvector(3, r_new_o);
        /* **************************************************** */

        /* EVALUATING CONTROL POINTS */
        getControlPixelsFromPose(est_copy_left, CamSel::left, ControlPixelMode::origin_x, l_px0_position, l_px1_position, l_px2_position, l_px3_position);

        yInfo() << "Left (original position) control px0 = [" << l_px0_position.toString() << "]";
        yInfo() << "Left (original position) control px1 = [" << l_px1_position.toString() << "]";
        yInfo() << "Left (original position) control px2 = [" << l_px2_position.toString() << "]";
        yInfo() << "Left (original position) control px3 = [" << l_px3_position.toString() << "]";


        getControlPixelsFromPose(est_copy_left, CamSel::left, ControlPixelMode::origin_o, l_px0_orientation, l_px1_orientation, l_px2_orientation, l_px3_orientation);

        yInfo() << "Left (original orientation) control px0 = [" << l_px0_orientation.toString() << "]";
        yInfo() << "Left (original orientation) control px1 = [" << l_px1_orientation.toString() << "]";
        yInfo() << "Left (original orientation) control px2 = [" << l_px2_orientation.toString() << "]";
        yInfo() << "Left (original orientation) control px3 = [" << l_px3_orientation.toString() << "]";


        getControlPixelsFromPose(est_copy_right, CamSel::right, ControlPixelMode::origin_x, r_px0_position, r_px1_position, r_px2_position, r_px3_position);

        yInfo() << "Right (original position) control px0 = [" << r_px0_position.toString() << "]";
        yInfo() << "Right (original position) control px1 = [" << r_px1_position.toString() << "]";
        yInfo() << "Right (original position) control px2 = [" << r_px2_position.toString() << "]";
        yInfo() << "Right (original position) control px3 = [" << r_px3_position.toString() << "]";


        getControlPixelsFromPose(est_copy_right, CamSel::right, ControlPixelMode::origin_o, r_px0_orientation, r_px1_orientation, r_px2_orientation, r_px3_orientation);

        yInfo() << "Right (original orientation) control px0 = [" << r_px0_orientation.toString() << "]";
        yInfo() << "Right (original orientation) control px1 = [" << r_px1_orientation.toString() << "]";
        yInfo() << "Right (original orientation) control px2 = [" << r_px2_orientation.toString() << "]";
        yInfo() << "Right (original orientation) control px3 = [" << r_px3_orientation.toString() << "]";


        /* FEATURES AND JACOBIAN (original position) */
        getCurrentStereoFeaturesAndJacobian(l_px0_position, l_px1_position, l_px2_position, l_px3_position,
                                            r_px0_position, r_px1_position, r_px2_position, r_px3_position,
                                            px_ee_cur_position, jacobian_position);

        yInfo() << "px_ee_cur_position = [" << px_ee_cur_position.toString() << "]";
        yInfo() << "jacobian_position  = [" << jacobian_position.toString() << "]";


        /* FEATURES AND JACOBIAN (original orientation) */
        getCurrentStereoFeaturesAndJacobian(l_px0_orientation, l_px1_orientation, l_px2_orientation, l_px3_orientation,
                                            r_px0_orientation, r_px1_orientation, r_px2_orientation, r_px3_orientation,
                                            px_ee_cur_orientation, jacobian_orientation);
        
        yInfo() << "px_ee_cur_orientation = [" << px_ee_cur_orientation.toString() << "]";
        yInfo() << "jacobian_orientation  = [" << jacobian_orientation.toString() << "]";


        Vector e_position               = px_des_ - px_ee_cur_position;
        Matrix inv_jacobian_position    = pinv(jacobian_position);

        Vector e_orientation            = px_des_ - px_ee_cur_orientation;
        Matrix inv_jacobian_orientation = pinv(jacobian_orientation);


        Vector vel_x = zeros(3);
        Vector vel_o = zeros(3);
        for (int i = 0; i < inv_jacobian_position.cols(); ++i)
        {
            Vector delta_vel_position    = inv_jacobian_position.getCol(i)    * e_position(i);
            Vector delta_vel_orientation = inv_jacobian_orientation.getCol(i) * e_orientation(i);

            if (i == 1 || i == 4 || i == 7 || i == 10)
            {
                vel_x += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_position.subVector(0, 2);
                vel_o += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_orientation.subVector(3, 5);
            }
            else
            {
                vel_x += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_position.subVector(0, 2);
                vel_o += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_orientation.subVector(3, 5);
            }
        }


        yInfo() << "px_des_               = [" << px_des_.toString()               << "]";
        yInfo() << "px_ee_cur_position    = [" << px_ee_cur_position.toString()    << "]";
        yInfo() << "px_ee_cur_orientation = [" << px_ee_cur_orientation.toString() << "]";
        yInfo() << "e_position            = [" << e_position.toString()            << "]";
        yInfo() << "e_orientation         = [" << e_orientation.toString()         << "]";
        yInfo() << "vel_x                 = [" << vel_x.toString()                 << "]";
        yInfo() << "vel_o                 = [" << vel_o.toString()                 << "]";

        double ang = norm(vel_o);
        vel_o /= ang;
        vel_o.push_back(ang);
        yInfo() << "axis-angle vel_o      = [" << vel_o.toString()                 << "]";


        /* Enforce translational velocity bounds */
        for (size_t i = 0; i < vel_x.length(); ++i)
            vel_x[i] = sign(vel_x[i]) * std::min(vx_max_, std::fabs(vel_x[i]));
        yInfo() << "bounded vel_x = [" << vel_x.toString() << "]";


        /* Enforce rotational velocity bounds */
        vel_o[3] = sign(vel_o[3]) * std::min(vo_max_, std::fabs(vel_o[3]));
        yInfo() << "bounded vel_o = [" << vel_o.toString() << "]";


        /* Visual control law */
        vel_x    *= K_x_;
        vel_o(3) *= K_o_;

        if (op_mode_ == OperatingMode::position)
            itf_rightarm_cart_->setTaskVelocities(vel_x, Vector(4, 0.0));
        else if (op_mode_ == OperatingMode::orientation)
            itf_rightarm_cart_->setTaskVelocities(Vector(3, 0.0), vel_o);
        else if (op_mode_ == OperatingMode::pose)
            itf_rightarm_cart_->setTaskVelocities(vel_x, vel_o);


        /* Wait for some motion */
        Time::delay(Ts_);


        /* Check for goal */
        bool is_pos_done = ((std::abs(px_des_(0) - px_ee_cur_position(0)) < px_tol_) && (std::abs(px_des_(1)  - px_ee_cur_position(1))  < px_tol_) && (std::abs(px_des_(2)  - px_ee_cur_position(2))  < px_tol_) &&
                            (std::abs(px_des_(3) - px_ee_cur_position(3)) < px_tol_) && (std::abs(px_des_(4)  - px_ee_cur_position(4))  < px_tol_) && (std::abs(px_des_(5)  - px_ee_cur_position(5))  < px_tol_) &&
                            (std::abs(px_des_(6) - px_ee_cur_position(6)) < px_tol_) && (std::abs(px_des_(7)  - px_ee_cur_position(7))  < px_tol_) && (std::abs(px_des_(8)  - px_ee_cur_position(8))  < px_tol_) &&
                            (std::abs(px_des_(9) - px_ee_cur_position(9)) < px_tol_) && (std::abs(px_des_(10) - px_ee_cur_position(10)) < px_tol_) && (std::abs(px_des_(11) - px_ee_cur_position(11)) < px_tol_));

        bool is_orient_done = ((std::abs(px_des_(0) - px_ee_cur_orientation(0)) < px_tol_) && (std::abs(px_des_(1)  - px_ee_cur_orientation(1))  < px_tol_) && (std::abs(px_des_(2)  - px_ee_cur_orientation(2))  < px_tol_) &&
                               (std::abs(px_des_(3) - px_ee_cur_orientation(3)) < px_tol_) && (std::abs(px_des_(4)  - px_ee_cur_orientation(4))  < px_tol_) && (std::abs(px_des_(5)  - px_ee_cur_orientation(5))  < px_tol_) &&
                               (std::abs(px_des_(6) - px_ee_cur_orientation(6)) < px_tol_) && (std::abs(px_des_(7)  - px_ee_cur_orientation(7))  < px_tol_) && (std::abs(px_des_(8)  - px_ee_cur_orientation(8))  < px_tol_) &&
                               (std::abs(px_des_(9) - px_ee_cur_orientation(9)) < px_tol_) && (std::abs(px_des_(10) - px_ee_cur_orientation(10)) < px_tol_) && (std::abs(px_des_(11) - px_ee_cur_orientation(11)) < px_tol_));

        if (op_mode_ == OperatingMode::position)
            is_vs_done = is_pos_done;
        else if (op_mode_ == OperatingMode::orientation)
            is_vs_done = is_orient_done;
        else if (op_mode_ == OperatingMode::pose)
            is_vs_done = is_pos_done && is_orient_done;


        if (is_vs_done)
        {
            yInfo() << "\npx_des ="              << px_des_.toString();
            yInfo() << "px_ee_cur_position ="    << px_ee_cur_position.toString();
            yInfo() << "px_ee_cur_orientation =" << px_ee_cur_orientation.toString();
            yInfo() << "\nTERMINATING!\n";
        }


        /* *** *** *** DEBUG OUTPUT *** *** *** */
        cv::Scalar red   (255,   0,   0);
        cv::Scalar green (  0, 255,   0);
        cv::Scalar blue  (  0,   0, 255);
        cv::Scalar yellow(255, 255,   0);

        /* Left eye end-effector superimposition */
        ImageOf<PixelRgb>* l_imgin  = port_image_left_in_.read(true);
        ImageOf<PixelRgb>& l_imgout = port_image_left_out_.prepare();
        l_imgout = *l_imgin;
        cv::Mat l_img = cv::cvarrToMat(l_imgout.getIplImage());

        cv::circle(l_img, cv::Point(l_px0_position[0],    l_px0_position[1]),    4, red   , 4);
        cv::circle(l_img, cv::Point(l_px1_position[0],    l_px1_position[1]),    4, green , 4);
        cv::circle(l_img, cv::Point(l_px2_position[0],    l_px2_position[1]),    4, blue  , 4);
        cv::circle(l_img, cv::Point(l_px3_position[0],    l_px3_position[1]),    4, yellow, 4);
        cv::circle(l_img, cv::Point(l_px0_orientation[0], l_px0_orientation[1]), 4, red   , 4);
        cv::circle(l_img, cv::Point(l_px1_orientation[0], l_px1_orientation[1]), 4, green , 4);
        cv::circle(l_img, cv::Point(l_px2_orientation[0], l_px2_orientation[1]), 4, blue  , 4);
        cv::circle(l_img, cv::Point(l_px3_orientation[0], l_px3_orientation[1]), 4, yellow, 4);
        cv::circle(l_img, cv::Point(l_px_goal_[0],        l_px_goal_[1]),        4, red   , 4);
        cv::circle(l_img, cv::Point(l_px_goal_[2],        l_px_goal_[3]),        4, green , 4);
        cv::circle(l_img, cv::Point(l_px_goal_[4],        l_px_goal_[5]),        4, blue  , 4);
        cv::circle(l_img, cv::Point(l_px_goal_[6],        l_px_goal_[7]),        4, yellow, 4);

        port_image_left_out_.write();

        /* Right eye end-effector superimposition */
        ImageOf<PixelRgb>* r_imgin  = port_image_right_in_.read(true);
        ImageOf<PixelRgb>& r_imgout = port_image_right_out_.prepare();
        r_imgout = *r_imgin;
        cv::Mat r_img = cv::cvarrToMat(r_imgout.getIplImage());

        cv::circle(r_img, cv::Point(r_px0_position[0],    r_px0_position[1]),    4, red   , 4);
        cv::circle(r_img, cv::Point(r_px1_position[0],    r_px1_position[1]),    4, green , 4);
        cv::circle(r_img, cv::Point(r_px2_position[0],    r_px2_position[1]),    4, blue  , 4);
        cv::circle(r_img, cv::Point(r_px3_position[0],    r_px3_position[1]),    4, yellow, 4);
        cv::circle(r_img, cv::Point(r_px0_orientation[0], r_px0_orientation[1]), 4, red   , 4);
        cv::circle(r_img, cv::Point(r_px1_orientation[0], r_px1_orientation[1]), 4, green , 4);
        cv::circle(r_img, cv::Point(r_px2_orientation[0], r_px2_orientation[1]), 4, blue  , 4);
        cv::circle(r_img, cv::Point(r_px3_orientation[0], r_px3_orientation[1]), 4, yellow, 4);
        cv::circle(r_img, cv::Point(r_px_goal_[0],        r_px_goal_[1]),        4, red   , 4);
        cv::circle(r_img, cv::Point(r_px_goal_[2],        r_px_goal_[3]),        4, green , 4);
        cv::circle(r_img, cv::Point(r_px_goal_[4],        r_px_goal_[5]),        4, blue  , 4);
        cv::circle(r_img, cv::Point(r_px_goal_[6],        r_px_goal_[7]),        4, yellow, 4);

        port_image_right_out_.write();
        /* *** *** *** *** *** *** *** *** *** */
    }

    itf_rightarm_cart_->stopControl();
    itf_gaze_->stopControl();

    Time::delay(0.5);

    return false;
}


bool ServerVisualServoing::interruptModule()
{
    yInfo() << "Interrupting module...";

    yInfo() << "...blocking controllers...";
    itf_rightarm_cart_->stopControl();
    itf_gaze_->stopControl();

    Time::delay(3.0);

    yInfo() << "...port cleanup...";
    port_pose_left_in_.interrupt();
    port_pose_right_in_.interrupt();
    port_image_left_in_.interrupt();
    port_image_left_out_.interrupt();
    port_click_left_.interrupt();
    port_image_right_in_.interrupt();
    port_image_right_out_.interrupt();
    port_click_right_.interrupt();

    yInfo() << "...done!";
    return true;
}


bool ServerVisualServoing::close()
{
    yInfo() << "Calling close functions...";

    port_pose_left_in_.close();
    port_pose_right_in_.close();
    port_image_left_in_.close();
    port_image_left_out_.close();
    port_click_left_.close();
    port_image_right_in_.close();
    port_image_right_out_.close();
    port_click_right_.close();

    itf_rightarm_cart_->removeTipFrame();

    if (rightarm_cartesian_driver_.isValid()) rightarm_cartesian_driver_.close();
    if (gaze_driver_.isValid())               gaze_driver_.close();

    yInfo() << "...done!";
    return true;
}


std::vector<std::string> ServerVisualServoing::get_info()
{
    std::vector<std::string> info;

    info.push_back("Unimplemented...!");

    return info;
}


/* Go to initial position (open-loop) */
bool ServerVisualServoing::init(const std::string& label)
{
    Vector xd       = zeros(3);
    Vector od       = zeros(4);
    Vector gaze_loc = zeros(3);

    /* Trial 27/04/17 */
    // -0.346 0.133 0.162 0.140 -0.989 0.026 2.693
//    xd[0] = -0.346;
//    xd[1] =  0.133;
//    xd[2] =  0.162;
//    od[0] =  0.140;
//    od[1] = -0.989;
//    od[2] =  0.026;
//    od[3] =  2.693;
//    gaze_loc[0] = -6.706;
//    gaze_loc[1] =  1.394;
//    gaze_loc[2] = -3.618;

    if (label == "t170517")
    {
        /* -0.300 0.088 0.080 -0.245 0.845 -0.473 2.896 */
        xd[0] = -0.300;
        xd[1] =  0.088;
        xd[2] =  0.080;

        od[0] = -0.245;
        od[1] =  0.845;
        od[2] = -0.473;
        od[3] =  2.896;

        gaze_loc[0] = -0.681;
        gaze_loc[1] =  0.112;
        gaze_loc[2] = -0.240;

        unsetTorsoDOF();
    }

    if (label == "sfm300517")
    {
        /* -0.333 0.203 -0.053 0.094 0.937 -0.335 3.111 */
        xd[0] = -0.333;
        xd[1] =  0.203;
        xd[2] = -0.053;

        od[0] =  0.094;
        od[1] =  0.937;
        od[2] = -0.335;
        od[3] =  3.111;

        /* -0.589 0.252 -0.409 */
        gaze_loc[0] = -0.589;
        gaze_loc[1] =  0.252;
        gaze_loc[2] = -0.409;

        setTorsoDOF();
        itf_rightarm_cart_->setLimits(0,  25.0,  25.0);
    }

    /* KARATE */
    // -0.319711 0.128912 0.075052 0.03846 -0.732046 0.680169 2.979943
//    xd[0] = -0.319;
//    xd[1] =  0.128;
//    xd[2] =  0.075;
//    Matrix Od = zeros(3, 3);
//    Od(0, 0) = -1.0;
//    Od(2, 1) = -1.0;
//    Od(1, 2) = -1.0;
//    od = dcm2axis(Od);
//    gaze_loc[0] = xd[0];
//    gaze_loc[1] = xd[1];
//    gaze_loc[2] = xd[2];

    /* GRASPING */
//    xd[0] = -0.370;
//    xd[1] =  0.103;
//    xd[2] =  0.064;
//    od(0) = -0.141;
//    od(1) =  0.612;
//    od(2) = -0.777;
//    od(4) =  3.012;
//    gaze_loc[0] = xd[0];
//    gaze_loc[1] = xd[1];
//    gaze_loc[2] = xd[2];


    /* SIM1 */
//    xd[0] = -0.416;
//    xd[1] =  0.024 + 0.1;
//    xd[2] =  0.055;
//    Matrix Od(3, 3);
//    Od(0, 0) = -1.0;
//    Od(1, 1) = -1.0;
//    Od(2, 2) =  1.0;
//    od = dcm2axis(Od);
//    gaze_loc[0] = xd[0];
//    gaze_loc[1] = xd[1];
//    gaze_loc[2] = xd[2];


    /* SIM2 */
//    xd[0] = -0.35;
//    xd[1] =  0.025 + 0.05;
//    xd[2] =  0.10;
//    Matrix Od(3, 3);
//    Od(0, 0) = -1.0;
//    Od(1, 1) = -1.0;
//    Od(2, 2) =  1.0;
//    od = dcm2axis(Od);
//    gaze_loc[0] = xd[0];
//    gaze_loc[1] = xd[1];
//    gaze_loc[2] = xd[2];


    double traj_time = 0.0;
    itf_rightarm_cart_->getTrajTime(&traj_time);

    if (norm(xd) != 0.0 && norm(od) != 0.0 && norm(gaze_loc) != 0.0 && traj_time == traj_time_)
    {
        yInfo() << "Init position:"    << xd.toString();
        yInfo() << "Init orientation:" << od.toString();

        itf_rightarm_cart_->goToPoseSync(xd, od);
        itf_rightarm_cart_->waitMotionDone(0.1, 10.0);
        itf_rightarm_cart_->stopControl();

        itf_rightarm_cart_->removeTipFrame();

        unsetTorsoDOF();

        itf_rightarm_cart_->storeContext(&ctx_cart_);


        yInfo() << "Fixation point: " << gaze_loc.toString();

        itf_gaze_->lookAtFixationPointSync(gaze_loc);
        itf_gaze_->waitMotionDone(0.1, 10.0);
        itf_gaze_->stopControl();

        itf_gaze_->storeContext(&ctx_gaze_);
    }
    else
        return false;

    return true;
}


/* Set a fixed goal in pixel coordinates */
/* PLUS: Compute again the roto-translation and projection matrices from root to left and right camera planes */
bool ServerVisualServoing::set_goal(const std::string& label)
{
    /* Hand pointing forward, palm looking down */
    //    Matrix R_ee = zeros(3, 3);
    //    R_ee(0, 0) = -1.0;
    //    R_ee(1, 1) =  1.0;
    //    R_ee(2, 2) = -1.0;
    //    Vector ee_o = dcm2axis(R_ee);

    /* Trial 27/04/17 */
    // -0.323 0.018 0.121 0.310 -0.873 0.374 3.008
    //    Vector p = zeros(6);
    //    p[0] = -0.323;
    //    p[1] =  0.018;
    //    p[2] =  0.121;
    //    p[3] =  0.310 * 3.008;
    //    p[4] = -0.873 * 3.008;
    //    p[5] =  0.374 * 3.008;

    /* Trial 17/05/17 */
    // -0.284 0.013 0.104 -0.370 0.799 -0.471 2.781
    Vector p = zeros(6);
    p[0] = -0.284;
    p[1] =  0.013;
    p[2] =  0.104;
    p[3] = -0.370 * 2.781;
    p[4] =  0.799 * 2.781;
    p[5] = -0.471 * 2.781;

    /* KARATE */
    //    Vector p = zeros(6);
    //    p[0] = -0.319;
    //    p[1] =  0.128;
    //    p[2] =  0.075;
    //    p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));

    /* SIM init 1 */
    // -0.416311	-0.026632	 0.055334	-0.381311	-0.036632	 0.055334	-0.381311	-0.016632	 0.055334
    //    Vector p = zeros(6);
    //    p[0] = -0.416;
    //    p[1] = -0.024;
    //    p[2] =  0.055;
    //    p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));

    /* SIM init 2 */
    //    Vector p = zeros(6);
    //    p[0] = -0.35;
    //    p[1] =  0.025;
    //    p[2] =  0.10;
    //    p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));

    goal_pose_ = p;
    yInfo() << "Goal: " << goal_pose_.toString();

    Vector p0 = zeros(4);
    Vector p1 = zeros(4);
    Vector p2 = zeros(4);
    Vector p3 = zeros(4);
    getControlPointsFromPose(p, p0, p1, p2, p3);

    yInfo() << "Goal px: [" << p0.toString() << ";" << p1.toString() << ";" << p2.toString() << ";" << p3.toString() << "];";


    Vector left_eye_x;
    Vector left_eye_o;
    itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

    Vector right_eye_x;
    Vector right_eye_o;
    itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

    yInfo() << "left_eye_o = ["  << left_eye_o.toString()  << "]";
    yInfo() << "right_eye_o = [" << right_eye_o.toString() << "]";


    l_H_eye_to_r_ = axis2dcm(left_eye_o);
    left_eye_x.push_back(1.0);
    l_H_eye_to_r_.setCol(3, left_eye_x);
    l_H_r_to_eye_ = SE3inv(l_H_eye_to_r_);

    r_H_eye_to_r_ = axis2dcm(right_eye_o);
    right_eye_x.push_back(1.0);
    r_H_eye_to_r_.setCol(3, right_eye_x);
    r_H_r_to_eye_ = SE3inv(r_H_eye_to_r_);

    yInfo() << "l_H_r_to_eye_ = [\n" << l_H_r_to_eye_.toString() << "]";
    yInfo() << "r_H_r_to_eye_ = [\n" << r_H_r_to_eye_.toString() << "]";

    l_H_r_to_cam_ = l_proj_ * l_H_r_to_eye_;
    r_H_r_to_cam_ = r_proj_ * r_H_r_to_eye_;


    Vector l_px0_goal = l_H_r_to_cam_ * p0;
    l_px0_goal[0] /= l_px0_goal[2];
    l_px0_goal[1] /= l_px0_goal[2];
    Vector l_px1_goal = l_H_r_to_cam_ * p1;
    l_px1_goal[0] /= l_px1_goal[2];
    l_px1_goal[1] /= l_px1_goal[2];
    Vector l_px2_goal = l_H_r_to_cam_ * p2;
    l_px2_goal[0] /= l_px2_goal[2];
    l_px2_goal[1] /= l_px2_goal[2];
    Vector l_px3_goal = l_H_r_to_cam_ * p3;
    l_px3_goal[0] /= l_px3_goal[2];
    l_px3_goal[1] /= l_px3_goal[2];

    l_px_goal_.resize(8);
    l_px_goal_[0] = l_px0_goal[0];
    l_px_goal_[1] = l_px0_goal[1];
    l_px_goal_[2] = l_px1_goal[0];
    l_px_goal_[3] = l_px1_goal[1];
    l_px_goal_[4] = l_px2_goal[0];
    l_px_goal_[5] = l_px2_goal[1];
    l_px_goal_[6] = l_px3_goal[0];
    l_px_goal_[7] = l_px3_goal[1];


    Vector r_px0_goal = r_H_r_to_cam_ * p0;
    r_px0_goal[0] /= r_px0_goal[2];
    r_px0_goal[1] /= r_px0_goal[2];
    Vector r_px1_goal = r_H_r_to_cam_ * p1;
    r_px1_goal[0] /= r_px1_goal[2];
    r_px1_goal[1] /= r_px1_goal[2];
    Vector r_px2_goal = r_H_r_to_cam_ * p2;
    r_px2_goal[0] /= r_px2_goal[2];
    r_px2_goal[1] /= r_px2_goal[2];
    Vector r_px3_goal = r_H_r_to_cam_ * p3;
    r_px3_goal[0] /= r_px3_goal[2];
    r_px3_goal[1] /= r_px3_goal[2];

    r_px_goal_.resize(8);
    r_px_goal_[0] = r_px0_goal[0];
    r_px_goal_[1] = r_px0_goal[1];
    r_px_goal_[2] = r_px1_goal[0];
    r_px_goal_[3] = r_px1_goal[1];
    r_px_goal_[4] = r_px2_goal[0];
    r_px_goal_[5] = r_px2_goal[1];
    r_px_goal_[6] = r_px3_goal[0];
    r_px_goal_[7] = r_px3_goal[1];

    yInfo() << "l_px_goal_ = [" << l_px_goal_.toString() << "]";
    yInfo() << "r_px_goal_ = [" << r_px_goal_.toString() << "]";

    px_des_.push_back(l_px_goal_[0]);    /* u_ee_l */
    px_des_.push_back(r_px_goal_[0]);    /* u_ee_r */
    px_des_.push_back(l_px_goal_[1]);    /* v_ee_l */

    px_des_.push_back(l_px_goal_[2]);    /* u_x1_l */
    px_des_.push_back(r_px_goal_[2]);    /* u_x1_r */
    px_des_.push_back(l_px_goal_[3]);    /* v_x1_l */

    px_des_.push_back(l_px_goal_[4]);    /* u_x2_l */
    px_des_.push_back(r_px_goal_[4]);    /* u_x2_r */
    px_des_.push_back(l_px_goal_[5]);    /* v_x2_l */

    px_des_.push_back(l_px_goal_[6]);    /* u_x3_l */
    px_des_.push_back(r_px_goal_[6]);    /* u_x3_r */
    px_des_.push_back(l_px_goal_[7]);    /* v_x3_l */

    yInfo() << "px_des_ = ["  << px_des_.toString() << "]";

    return true;
}


/* Get 3D point from Structure From Motion clicking on the left camera image */
/* PLUS: Compute again the roto-translation and projection matrices from root to left and right camera planes */
bool ServerVisualServoing::get_sfm_points()
{
    Vector left_eye_x;
    Vector left_eye_o;
    itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

    Vector right_eye_x;
    Vector right_eye_o;
    itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

    yInfo() << "left_eye_o =" << left_eye_o.toString();
    yInfo() << "right_eye_o =" << right_eye_o.toString();


    l_H_eye_to_r_ = axis2dcm(left_eye_o);
    left_eye_x.push_back(1.0);
    l_H_eye_to_r_.setCol(3, left_eye_x);
    l_H_r_to_eye_ = SE3inv(l_H_eye_to_r_);

    r_H_eye_to_r_ = axis2dcm(right_eye_o);
    right_eye_x.push_back(1.0);
    r_H_eye_to_r_.setCol(3, right_eye_x);
    r_H_r_to_eye_ = SE3inv(r_H_eye_to_r_);

    yInfo() << "l_H_r_to_eye_ =\n" << l_H_r_to_eye_.toString();
    yInfo() << "r_H_r_to_eye_ =\n" << r_H_r_to_eye_.toString();

    l_H_r_to_cam_ = l_proj_ * l_H_r_to_eye_;
    r_H_r_to_cam_ = r_proj_ * r_H_r_to_eye_;


    Network yarp;
    Bottle  cmd;
    Bottle  rep;

    Bottle* click_left = port_click_left_.read(true);
    Vector l_click = zeros(2);
    l_click[0] = click_left->get(0).asDouble();
    l_click[1] = click_left->get(1).asDouble();

    RpcClient port_sfm;
    port_sfm.open("/visual-servoing/tosfm");
    yarp.connect("/visual-servoing/tosfm", "/SFM/rpc");

    cmd.clear();

    cmd.addInt(l_click[0]);
    cmd.addInt(l_click[1]);

    Bottle reply_pos;
    port_sfm.write(cmd, reply_pos);
    if (reply_pos.size() == 5)
    {
        Matrix R_ee = zeros(3, 3);
        R_ee(0, 0) = -1.0;
//        R_ee(1, 1) =  1.0;
        R_ee(1, 2) = -1.0;
//        R_ee(2, 2) = -1.0;
        R_ee(2, 1) = -1.0;
        Vector ee_o = dcm2axis(R_ee);

        Vector sfm_pos = zeros(3);
        sfm_pos[0] = reply_pos.get(0).asDouble();
        sfm_pos[1] = reply_pos.get(1).asDouble();
        sfm_pos[2] = reply_pos.get(2).asDouble();

        Vector p = zeros(7);
        p.setSubvector(0, sfm_pos.subVector(0, 2));
        p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));

        goal_pose_ = p;
        yInfo() << "Goal: " << goal_pose_.toString();

        Vector p0 = zeros(4);
        Vector p1 = zeros(4);
        Vector p2 = zeros(4);
        Vector p3 = zeros(4);
        getControlPointsFromPose(p, p0, p1, p2, p3);

        yInfo() << "goal px: [" << p0.toString() << ";" << p1.toString() << ";" << p2.toString() << ";" << p3.toString() << "];";


        Vector l_px0_goal = l_H_r_to_cam_ * p0;
        l_px0_goal[0] /= l_px0_goal[2];
        l_px0_goal[1] /= l_px0_goal[2];
        Vector l_px1_goal = l_H_r_to_cam_ * p1;
        l_px1_goal[0] /= l_px1_goal[2];
        l_px1_goal[1] /= l_px1_goal[2];
        Vector l_px2_goal = l_H_r_to_cam_ * p2;
        l_px2_goal[0] /= l_px2_goal[2];
        l_px2_goal[1] /= l_px2_goal[2];
        Vector l_px3_goal = l_H_r_to_cam_ * p3;
        l_px3_goal[0] /= l_px3_goal[2];
        l_px3_goal[1] /= l_px3_goal[2];

        l_px_goal_.resize(8);
        l_px_goal_[0] = l_px0_goal[0];
        l_px_goal_[1] = l_px0_goal[1];
        l_px_goal_[2] = l_px1_goal[0];
        l_px_goal_[3] = l_px1_goal[1];
        l_px_goal_[4] = l_px2_goal[0];
        l_px_goal_[5] = l_px2_goal[1];
        l_px_goal_[6] = l_px3_goal[0];
        l_px_goal_[7] = l_px3_goal[1];


        Vector r_px0_goal = r_H_r_to_cam_ * p0;
        r_px0_goal[0] /= r_px0_goal[2];
        r_px0_goal[1] /= r_px0_goal[2];
        Vector r_px1_goal = r_H_r_to_cam_ * p1;
        r_px1_goal[0] /= r_px1_goal[2];
        r_px1_goal[1] /= r_px1_goal[2];
        Vector r_px2_goal = r_H_r_to_cam_ * p2;
        r_px2_goal[0] /= r_px2_goal[2];
        r_px2_goal[1] /= r_px2_goal[2];
        Vector r_px3_goal = r_H_r_to_cam_ * p3;
        r_px3_goal[0] /= r_px3_goal[2];
        r_px3_goal[1] /= r_px3_goal[2];

        r_px_goal_.resize(8);
        r_px_goal_[0] = r_px0_goal[0];
        r_px_goal_[1] = r_px0_goal[1];
        r_px_goal_[2] = r_px1_goal[0];
        r_px_goal_[3] = r_px1_goal[1];
        r_px_goal_[4] = r_px2_goal[0];
        r_px_goal_[5] = r_px2_goal[1];
        r_px_goal_[6] = r_px3_goal[0];
        r_px_goal_[7] = r_px3_goal[1];

        yInfo() << "l_px_goal_ = [" << l_px_goal_.toString() << "]";
        yInfo() << "r_px_goal_ = [" << r_px_goal_.toString() << "]";

        px_des_.push_back(l_px_goal_[0]);    /* u_ee_l */
        px_des_.push_back(r_px_goal_[0]);    /* u_ee_r */
        px_des_.push_back(l_px_goal_[1]);    /* v_ee_l */

        px_des_.push_back(l_px_goal_[2]);    /* u_x1_l */
        px_des_.push_back(r_px_goal_[2]);    /* u_x1_r */
        px_des_.push_back(l_px_goal_[3]);    /* v_x1_l */

        px_des_.push_back(l_px_goal_[4]);    /* u_x2_l */
        px_des_.push_back(r_px_goal_[4]);    /* u_x2_r */
        px_des_.push_back(l_px_goal_[5]);    /* v_x2_l */

        px_des_.push_back(l_px_goal_[6]);    /* u_x3_l */
        px_des_.push_back(r_px_goal_[6]);    /* u_x3_r */
        px_des_.push_back(l_px_goal_[7]);    /* v_x3_l */

        yInfo() << "px_des_ = ["  << px_des_.toString() << "]";
    }
    else
        return false;

    yarp.disconnect("/visual-servoing/tosfm", "/SFM/rpc");
    port_sfm.close();

    return true;
}


/* Set visual servoing operating mode */
bool ServerVisualServoing::set_modality(const std::string& mode)
{
    if (mode == "position")
        op_mode_ = OperatingMode::position;
    else if (mode == "orientation")
        op_mode_ = OperatingMode::orientation;
    else if (mode == "pose")
        op_mode_ = OperatingMode::pose;
    else
        return false;

    return true;
}


bool ServerVisualServoing::set_position_gain(const double k)
{
    K_x_ = k;

    return true;
}


bool ServerVisualServoing::set_orientation_gain(const double k)
{
    K_o_ = k;

    return true;
}


bool ServerVisualServoing::set_position_bound(const double b)
{
    vx_max_ = b;

    return true;
}


bool ServerVisualServoing::set_orientation_bound(const double b)
{
    vo_max_ = b;

    return true;
}


bool ServerVisualServoing::set_goal_tol(const double px)
{
    px_tol_ = px;

    return true;
}


/* Start visual servoing */
bool ServerVisualServoing::go()
{
    shall_go_ = true;

    return true;
}


/* Safely close the application */
bool ServerVisualServoing::quit()
{
    itf_rightarm_cart_->stopControl();
    itf_gaze_->stopControl();

    should_stop_ = true;
    shall_go_    = true;

    stopModule();

    return true;
}


bool ServerVisualServoing::setRightArmCartesianController()
{
    Property rightarm_cartesian_options;
    rightarm_cartesian_options.put("device", "cartesiancontrollerclient");
    rightarm_cartesian_options.put("local",  "/visual-servoing/cart_right_arm");
    rightarm_cartesian_options.put("remote", "/"+robot_name_+"/cartesianController/right_arm");

    rightarm_cartesian_driver_.open(rightarm_cartesian_options);
    if (rightarm_cartesian_driver_.isValid())
    {
        rightarm_cartesian_driver_.view(itf_rightarm_cart_);
        if (!itf_rightarm_cart_)
        {
            yError() << "Error getting ICartesianControl interface.";
            return false;
        }
        yInfo() << "cartesiancontrollerclient succefully opened.";
    }
    else
    {
        yError() << "Error opening cartesiancontrollerclient device.";
        return false;
    }

    if (!itf_rightarm_cart_->setTrajTime(traj_time_))
    {
        yError() << "Error setting ICartesianControl trajectory time.";
        return false;
    }
    yInfo() << "Succesfully set ICartesianControl trajectory time!";

    if (!itf_rightarm_cart_->setInTargetTol(0.01))
    {
        yError() << "Error setting ICartesianControl target tolerance.";
        return false;
    }
    yInfo() << "Succesfully set ICartesianControl target tolerance!";

    return true;
}


bool ServerVisualServoing::setGazeController()
{
    Property gaze_option;
    gaze_option.put("device", "gazecontrollerclient");
    gaze_option.put("local",  "/visual-servoing/gaze");
    gaze_option.put("remote", "/iKinGazeCtrl");

    gaze_driver_.open(gaze_option);
    if (gaze_driver_.isValid())
    {
        gaze_driver_.view(itf_gaze_);
        if (!itf_gaze_)
        {
            yError() << "Error getting IGazeControl interface.";
            return false;
        }
    }
    else
    {
        yError() << "Gaze control device not available.";
        return false;
    }

    return true;
}


bool ServerVisualServoing::attach(yarp::os::Port &source)
{
    return this->yarp().attachAsServer(source);
}


bool ServerVisualServoing::setCommandPort()
{
    std::cout << "Opening RPC command port." << std::endl;
    if (!port_rpc_command_.open("/visual-servoing/cmd:i"))
    {
        std::cerr << "Cannot open the RPC command port." << std::endl;
        return false;
    }
    if (!attach(port_rpc_command_))
    {
        std::cerr << "Cannot attach the RPC command port." << std::endl;
        return false;
    }
    std::cout << "RPC command port opened and attached. Ready to recieve commands!" << std::endl;

    return true;
}


bool ServerVisualServoing::setTorsoDOF()
{
    Vector curDOF;
    itf_rightarm_cart_->getDOF(curDOF);
    yInfo() << "Old DOF: [" + curDOF.toString(0) + "].";

    yInfo() << "Setting iCub to use torso DOF.";
    Vector newDOF(curDOF);
    newDOF[0] = 1;
    newDOF[1] = 1;
    newDOF[2] = 1;
    if (!itf_rightarm_cart_->setDOF(newDOF, curDOF))
    {
        yError() << "Unable to set torso DOF.";
        return false;
    }
    yInfo() << "New DOF: [" + curDOF.toString(0) + "]";
    
    return true;
}


bool ServerVisualServoing::unsetTorsoDOF()
{
    Vector curDOF;
    itf_rightarm_cart_->getDOF(curDOF);
    yInfo() << "Old DOF: [" + curDOF.toString(0) + "].";

    yInfo() << "Setting iCub to block torso DOF.";
    Vector newDOF(curDOF);
    newDOF[0] = 0;
    newDOF[1] = 0;
    newDOF[2] = 0;
    if (!itf_rightarm_cart_->setDOF(newDOF, curDOF))
    {
        yError() << "Unable to set torso DOF.";
        return false;
    }
    yInfo() << "New DOF: [" + curDOF.toString(0) + "]";
    
    return true;
}


void ServerVisualServoing::getControlPixelsFromPose(const Vector& pose, const CamSel cam, const ControlPixelMode mode, Vector& px0, Vector& px1, Vector& px2, Vector& px3)
{
    yAssert(cam == CamSel::left || cam == CamSel::right);
    yAssert(mode == ControlPixelMode::origin || mode == ControlPixelMode::origin_o || mode == ControlPixelMode::origin_x);


    Vector control_pose = pose;
    if (mode == ControlPixelMode::origin_x)
        control_pose.setSubvector(3, goal_pose_.subVector(3, 5));
    else if (mode == ControlPixelMode::origin_o)
        control_pose.setSubvector(0, goal_pose_.subVector(0, 2));


    Vector control_p0 = zeros(4);
    Vector control_p1 = zeros(4);
    Vector control_p2 = zeros(4);
    Vector control_p3 = zeros(4);
    getControlPointsFromPose(control_pose, control_p0, control_p1, control_p2, control_p3);


    px0 = getPixelFromPoint(cam, control_p0);
    px1 = getPixelFromPoint(cam, control_p1);
    px2 = getPixelFromPoint(cam, control_p2);
    px3 = getPixelFromPoint(cam, control_p3);
}


void ServerVisualServoing::getControlPointsFromPose(const Vector& pose, Vector& p0, Vector& p1, Vector& p2, Vector& p3)
{
    Vector ee_x = pose.subVector(0, 2);
    ee_x.push_back(1.0);
    double ang  = norm(pose.subVector(3, 5));
    Vector ee_o = pose.subVector(3, 5) / ang;
    ee_o.push_back(ang);

    Matrix H_ee_to_root = axis2dcm(ee_o);
    H_ee_to_root.setCol(3, ee_x);


    Vector p = zeros(4);

    p(0) =  0;
    p(1) = -0.015;
    p(2) =  0;
    p(3) =  1.0;

    p0 = zeros(4);
    p0 = H_ee_to_root * p;

    p(0) = 0;
    p(1) = 0.015;
    p(2) = 0;
    p(3) = 1.0;

    p1 = zeros(4);
    p1 = H_ee_to_root * p;

    p(0) = -0.035;
    p(1) =  0.015;
    p(2) =  0;
    p(3) =  1.0;

    p2 = zeros(4);
    p2 = H_ee_to_root * p;

    p(0) = -0.035;
    p(1) = -0.015;
    p(2) =  0;
    p(3) =  1.0;

    p3 = zeros(4);
    p3 = H_ee_to_root * p;
}


Vector ServerVisualServoing::getPixelFromPoint(const CamSel cam, const Vector& p) const
{
    yAssert(cam == CamSel::left || cam == CamSel::right);

    Vector px;

    if (cam == CamSel::left)
        px = l_H_r_to_cam_ * p;
    else if (cam == CamSel::right)
        px = r_H_r_to_cam_ * p;

    px[0] /= px[2];
    px[1] /= px[2];

    return px;
}


void ServerVisualServoing::getCurrentStereoFeaturesAndJacobian(const Vector& left_px0,  const Vector& left_px1,  const Vector& left_px2,  const Vector& left_px3,
                                                               const Vector& right_px0, const Vector& right_px1, const Vector& right_px2, const Vector& right_px3,
                                                               Vector& features, Matrix& jacobian)
{
    if (features.length() != 12)
        features.resize(12);

    if (jacobian.rows() != 12 || jacobian.cols() != 6)
        jacobian.resize(12, 6);


    /* FEATURES */
    features[0]  = left_px0 [0];    /* u_ee_l */
    features[1]  = right_px0[0];    /* u_ee_r */
    features[2]  = left_px0 [1];    /* v_ee_l */

    features[3]  = left_px1 [0];    /* u_x1_l */
    features[4]  = right_px1[0];    /* u_x1_r */
    features[5]  = left_px1 [1];    /* v_x1_l */

    features[6]  = left_px2 [0];    /* u_x2_l */
    features[7]  = right_px2[0];    /* u_x2_r */
    features[8]  = left_px2 [1];    /* v_x2_l */

    features[9]  = left_px3 [0];    /* u_x3_l */
    features[10] = right_px3[0];    /* u_x3_r */
    features[11] = left_px3 [1];    /* v_x3_l */


    /* JACOBIAN */
    jacobian.setRow(0,  getJacobianU(CamSel::left,  left_px0));
    jacobian.setRow(1,  getJacobianU(CamSel::right, right_px0));
    jacobian.setRow(2,  getJacobianV(CamSel::left,  left_px0));

    jacobian.setRow(3,  getJacobianU(CamSel::left,  left_px1));
    jacobian.setRow(4,  getJacobianU(CamSel::right, right_px1));
    jacobian.setRow(5,  getJacobianV(CamSel::left,  left_px1));

    jacobian.setRow(6,  getJacobianU(CamSel::left,  left_px2));
    jacobian.setRow(7,  getJacobianU(CamSel::right, right_px2));
    jacobian.setRow(8,  getJacobianV(CamSel::left,  left_px2));

    jacobian.setRow(9,  getJacobianU(CamSel::left,  left_px3));
    jacobian.setRow(10, getJacobianU(CamSel::right, right_px3));
    jacobian.setRow(11, getJacobianV(CamSel::left,  left_px3));
}


Vector ServerVisualServoing::getJacobianU(const CamSel cam, const Vector& px)
{
    Vector jacobian = zeros(6);
    
    if (cam == CamSel::left)
    {
        jacobian(0) = l_proj_(0, 0) / px(2);
        jacobian(2) = - (px(0) - l_proj_(0, 2)) / px(2);
        jacobian(3) = - ((px(0) - l_proj_(0, 2)) * (px(1) - l_proj_(1, 2))) / l_proj_(1, 1);
        jacobian(4) = (pow(l_proj_(0, 0), 2.0) + pow(px(0) - l_proj_(0, 2), 2.0)) / l_proj_(0, 0);
        jacobian(5) = - l_proj_(0, 0) / l_proj_(1, 1) * (px(1) - l_proj_(1, 2));
    }
    else if (cam == CamSel::right)
    {
        jacobian(0) = r_proj_(0, 0) / px(2);
        jacobian(2) = - (px(0) - r_proj_(0, 2)) / px(2);
        jacobian(3) = - ((px(0) - r_proj_(0, 2)) * (px(1) - r_proj_(1, 2))) / r_proj_(1, 1);
        jacobian(4) = (pow(r_proj_(0, 0), 2.0) + pow(px(0) - r_proj_(0, 2), 2.0)) / r_proj_(0, 0);
        jacobian(5) = - r_proj_(0, 0) / r_proj_(1, 1) * (px(1) - r_proj_(1, 2));
    }
    
    return jacobian;
}


Vector ServerVisualServoing::getJacobianV(const CamSel cam, const Vector& px)
{
    Vector jacobian = zeros(6);
    
    if (cam == CamSel::left)
    {
        jacobian(1) = l_proj_(1, 1) / px(2);
        jacobian(2) = - (px(1) - l_proj_(1, 2)) / px(2);
        jacobian(3) = - (pow(l_proj_(1, 1), 2.0) + pow(px(1) - l_proj_(1, 2), 2.0)) / l_proj_(1, 1);
        jacobian(4) = ((px(0) - l_proj_(0, 2)) * (px(1) - l_proj_(1, 2))) / l_proj_(0, 0);
        jacobian(5) = l_proj_(1, 1) / l_proj_(0, 0) * (px(0) - l_proj_(0, 2));
    }
    else if (cam == CamSel::right)
    {
        jacobian(1) = r_proj_(1, 1) / px(2);
        jacobian(2) = - (px(1) - r_proj_(1, 2)) / px(2);
        jacobian(3) = - (pow(r_proj_(1, 1), 2.0) + pow(px(1) - r_proj_(1, 2), 2.0)) / r_proj_(1, 1);
        jacobian(4) = ((px(0) - r_proj_(0, 2)) * (px(1) - r_proj_(1, 2))) / r_proj_(0, 0);
        jacobian(5) = r_proj_(1, 1) / r_proj_(0, 0) * (px(0) - r_proj_(0, 2));
    }
    
    return jacobian;
}


Vector ServerVisualServoing::getAxisAngle(const Vector& v)
{
    double ang  = norm(v);
    Vector aa   = v / ang;
    aa.push_back(ang);
    
    return aa;
}
