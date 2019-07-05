/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <VisualProprioceptionSiamese.h>
#include <ReceiveMasks.h>
#include <ReceiveDepth.h>
#include <utils.h>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

#include <yarp/dev/IRGBDSensor.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>
#include <yarp/os/Property.h>
#include <yarp/os/ResourceFinder.h>

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace Eigen;
using namespace hand_tracking::utils;
using namespace yarp::os;


struct VisualProprioceptionSiamese::ImplData
{
    const std::string log_ID_ = "[VisualProprioceptionSiamese]";

    std::unique_ptr<ReceiveMasks> receive_masks_ = nullptr;

    std::unique_ptr<ReceiveDepth> receive_depth_ = nullptr; 

    bfl::Camera::CameraIntrinsics cam_params_;

    SICAD::ModelPathContainer mesh_paths_;

    std::string shader_folder_;

    std::unique_ptr<SICAD> si_cad_;

    int num_images_;

    yarp::sig::ImageOf<yarp::sig::PixelMono> mask_;

    yarp::sig::ImageOf<yarp::sig::PixelFloat> depth_map_;
};


VisualProprioceptionSiamese::VisualProprioceptionSiamese
(
    std::unique_ptr<ReceiveMasks> receive_masks,
    std::unique_ptr<ReceiveDepth> receive_depth,
    const int num_requested_images,
    const std::string& object_name,
    const std::string& context
) :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    ImplData& rImpl = *pImpl_;

    rImpl.receive_masks_ = std::move(receive_masks);

    rImpl.receive_depth_ = std::move(receive_depth);

    // Get camera intrinsic parameters
    Property properties;
    properties.put("device", "RGBDSensorClient");
    properties.put("localImagePort",  "/visual-proprioception-siamese/RGBDSensorClient/image:i");
    properties.put("localDepthPort",  "/visual-proprioception-siamese/RGBDSensorClient/depth:i");
    properties.put("localRpcPort",    "/visual-proprioception-siamese/RGBDSensorClient/rpc:i");
    properties.put("remoteImagePort", "/depthCamera/rgbImage:o");
    properties.put("remoteDepthPort", "/depthCamera/depthImage:o");
    properties.put("remoteRpcPort",   "/depthCamera/rpc:i");

    yarp::dev::PolyDriver rgbd_drv;
    yarp::dev::IRGBDSensor* irgbd;
    if (rgbd_drv.open(properties) && rgbd_drv.view(irgbd) && (irgbd != nullptr))
    {
        std::cout << "INFO::VISUALPROPRIOCEPTION::CTOR\n Getting intrinsic parameters from camera driver." << std::endl;

        yarp::os::Property camera_intrinsics;
        irgbd->getRgbIntrinsicParam(camera_intrinsics);

        rImpl.cam_params_.width = irgbd->getRgbWidth();
        rImpl.cam_params_.height = irgbd->getRgbHeight();
        rImpl.cam_params_.fx = camera_intrinsics.find("focalLengthX").asFloat64();
        rImpl.cam_params_.fy = camera_intrinsics.find("focalLengthY").asFloat64();
        rImpl.cam_params_.cx = camera_intrinsics.find("principalPointX").asFloat64();
        rImpl.cam_params_.cy = camera_intrinsics.find("principalPointY").asFloat64();

        rgbd_drv.close();
    }
    else
    {
        std::cout << "INFO::VISUALPROPRIOCEPTION::CTOR" << std::endl
                  << "Camera driver not ready, reading intrinsic paramers from fallback configuration" << std::endl;
        std::cout << "INFO::VISUALPROPRIOCEPTION::CTOR" << std::endl
                  << "Using fallback configuration" << std::endl;

        ResourceFinder rf;
        rf.setVerbose(false);
        rf.setDefaultContext(context);
        rf.setDefaultConfigFile("realsense_camera_config.ini");
        rf.configure(0, nullptr);

        ResourceFinder rf_camera = rf.findNestedResourceFinder("ycb_camera_320_240");
        rImpl.cam_params_.width = rf_camera.find("width").asDouble();
        rImpl.cam_params_.height = rf_camera.find("height").asDouble();
        rImpl.cam_params_.fx = rf_camera.find("fx").asDouble();
        rImpl.cam_params_.fy = rf_camera.find("fy").asDouble();
        rImpl.cam_params_.cx = rf_camera.find("cx").asDouble();
        rImpl.cam_params_.cy = rf_camera.find("cy").asDouble();
    }

    std::cout << "INFO::VISUALPROPRIOCEPTION::CTOR" << std::endl
              << "Camera intrinsic parameters are:" << std::endl
              << "- width:" << rImpl.cam_params_.width << std::endl
              << "- height:" << rImpl.cam_params_.height << std::endl
              << "- fx:" << rImpl.cam_params_.fx << std::endl
              << "- fy:" << rImpl.cam_params_.fy << std::endl
              << "- cx:" << rImpl.cam_params_.cx << std::endl
              << "- cy:" << rImpl.cam_params_.cy << std::endl;

    ResourceFinder rf;
    rf.setVerbose(false);

    // Get object mesh path
    rf.setDefaultContext(context + "/mesh");
    std::string mesh_path = rf.findFileByName(object_name + ".obj");
    if (mesh_path.empty())
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTIONSIAMESE::CTOR::DIR\nERROR: mesh path not found!");
    else
        std::cout << "INFO::VISUALPROPRIOCEPTION::CTOR" << std::endl
                  << "Found mesh path:" << mesh_path << std::endl;

    // Get shader path
    rf.setDefaultContext(context + "/shader");
    std::string shader_path = rf.findFileByName("shader_model.vert");
    if (shader_path.empty())
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTIONSIAMESE::CTOR::DIR\n\t ERROR: shader directory not found!");

    size_t rfind_slash = shader_path.rfind("/");
    if (rfind_slash == std::string::npos)
        rfind_slash = 0;
    size_t rfind_backslash = shader_path.rfind("\\");
    if (rfind_backslash == std::string::npos)
        rfind_backslash = 0;

    shader_path = shader_path.substr(0, rfind_slash > rfind_backslash ? rfind_slash : rfind_backslash);
    std::cout << "INFO::VISUALPROPRIOCEPTION::CTOR\n Found shader directory:" << shader_path << std::endl;

    SICAD::ModelPathContainer path_container;
    path_container["object"] = mesh_path;
    try
    {
        rImpl.si_cad_ = std::unique_ptr<SICAD>(new SICAD(path_container,
                                                         rImpl.cam_params_.width, rImpl.cam_params_.height,
                                                         rImpl.cam_params_.fx, rImpl.cam_params_.fy, rImpl.cam_params_.cx, rImpl.cam_params_.cy,
                                                         num_requested_images,
                                                         shader_path,
                                                         { 1.0, 0.0, 0.0, static_cast<float>(M_PI) }));
    }
    catch (const std::runtime_error& e)
    {
        throw std::runtime_error(e.what());
    }

    rImpl.num_images_ = rImpl.si_cad_->getTilesNumber();

}


VisualProprioceptionSiamese::~VisualProprioceptionSiamese()
{ }


std::pair<bool, bfl::Data> VisualProprioceptionSiamese::measure(const bfl::Data& data) const
{
    std::pair<yarp::sig::ImageOf<yarp::sig::PixelMono>,yarp::sig::ImageOf<yarp::sig::PixelFloat>> measure_pair = std::make_pair(std::move(pImpl_->mask_),std::move(pImpl_->depth_map_));
    return std::make_pair(true, measure_pair);
}


std::pair<bool, bfl::Data> VisualProprioceptionSiamese::predictedMeasure(const Ref<const MatrixXd>& cur_states) const
{
    ImplData& rImpl = *pImpl_;

    std::array<double, 3> camera_position;
    std::array<double, 4> camera_orientation;
    camera_position[0] = 0.0;
    camera_position[1] = 0.0;
    camera_position[2] = 0.0;

    camera_orientation[0] = 1.0;
    camera_orientation[1] = 0.0;
    camera_orientation[2] = 0.0;
    camera_orientation[3] = 0.0;

    std::vector<Superimpose::ModelPoseContainer> model_poses(cur_states.cols());
    
    for(int i=0;i<cur_states.cols();i++)
    {
        Superimpose::ModelPoseContainer mesh_poses;
        std::vector<double> mesh_pose(7);

        /*
        * SuperimposeMeshLib requires axis-angle representation,
        * hence the Euler ZYX representation stored in cur_states is converted to axis-angle.
        */
        VectorXd obj_o = euler_to_axis_angle(cur_states.col(i).tail<3>(), AxisOfRotation::UnitZ, AxisOfRotation::UnitY, AxisOfRotation::UnitX);
        

        mesh_pose[0] = cur_states(0,i);
        mesh_pose[1] = cur_states(1,i);
        mesh_pose[2] = cur_states(2,i);
        mesh_pose[3] = obj_o(0);
        mesh_pose[4] = obj_o(1);
        mesh_pose[5] = obj_o(2);
        mesh_pose[6] = obj_o(3);        

        mesh_poses.emplace("object",mesh_pose);
        model_poses[i] = mesh_poses;
    }

    cv::Mat rendered_image;
    cv::Mat rendered_depth;
    if (!(rImpl.si_cad_->superimpose(model_poses, camera_position.data(), camera_orientation.data(), rendered_image, rendered_depth)))
        {
            std::pair<cv::Mat,cv::Mat> rendered_pair = std::make_pair(std::move(rendered_image),std::move(rendered_depth));
            return std::make_pair(false, rendered_pair); // initialize with empty if rendering failed
        }
    // return the rendered pair
    std::pair<cv::Mat,cv::Mat> rendered_pair = std::make_pair(std::move(rendered_image),std::move(rendered_depth));
    return std::make_pair(true, rendered_pair);
}


std::pair<bool, bfl::Data> VisualProprioceptionSiamese::innovation(const bfl::Data& predicted_measurements, const bfl::Data& measurements) const
{
    MatrixXf innovation = -(bfl::any::any_cast<MatrixXf>(predicted_measurements).rowwise() - bfl::any::any_cast<MatrixXf>(measurements).row(0));

    return std::make_pair(true, std::move(innovation));
}


bool VisualProprioceptionSiamese::freeze()
{
    ImplData& rImpl = *pImpl_;

    rImpl.mask_ = bfl::any::any_cast<yarp::sig::ImageOf<yarp::sig::PixelMono>>(rImpl.receive_masks_->GetMask());

    rImpl.depth_map_ = bfl::any::any_cast<yarp::sig::ImageOf<yarp::sig::PixelFloat>>(rImpl.receive_depth_->GetDepth());

    return true;
}


std::pair<std::size_t, std::size_t> VisualProprioceptionSiamese::getOutputSize() const
{
    /* The output size is set to (0, 0) since the measurements are stored in a row major format,
       hence not compatible with the required output format. */
    return std::make_pair(0, 0);
}


int VisualProprioceptionSiamese::getNumberOfUsedParticles() const
{
    ImplData& rImpl = *pImpl_;

    return rImpl.num_images_;
}

