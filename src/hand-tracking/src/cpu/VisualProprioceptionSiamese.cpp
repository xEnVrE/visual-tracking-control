/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <VisualProprioceptionSiamese.h>
#include <ReceiveMasks.h>
#include <utils.h>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

#include <yarp/sig/Vector.h>
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

    bfl::Camera::CameraIntrinsics cam_params_;

    SICAD::ModelPathContainer mesh_paths_;

    std::string shader_folder_;

    std::unique_ptr<SICAD> si_cad_;

    int num_images_;

    yarp::sig::ImageOf<yarp::sig::PixelMono> mask_;
};


VisualProprioceptionSiamese::VisualProprioceptionSiamese
(
    std::unique_ptr<ReceiveMasks> receive_masks,
    const int num_requested_images,
    const std::string& object_name,
    const std::string& context
) :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    ImplData& rImpl = *pImpl_;

    rImpl.receive_masks_ = std::move(receive_masks);

    rImpl.cam_params_.width = 640;
    rImpl.cam_params_.height = 480;
    rImpl.cam_params_.fx = 617.170349121094;
    rImpl.cam_params_.fy = 616.72265625;
    rImpl.cam_params_.cx = 309.493408203124;
    rImpl.cam_params_.cy = 235.852325439454;

    ResourceFinder rf;
    rf.setVerbose(true);

    // Get object mesh path
    rf.setDefaultContext(context + "/mesh");
    std::string mesh_path = rf.findFileByName(object_name + ".obj");
    if (mesh_path.empty())
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTIONSIAMESE::CTOR::DIR\nERROR: mesh path not found!");
    else
        std::cout << "INFO::VISUALPROPRIOCEPTION::CTOR\n Found mesh path:" << mesh_path << std::endl;

    // Get shader path
    rf.setDefaultContext(context + "/shader");
    std::string shader_path = rf.findFileByName("shader_model.vert");
    if (shader_path.empty())
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTIONSIAMESE::CTOR::DIR\nERROR: shader directory not found!");

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


VisualProprioceptionSiamese::~VisualProprioceptionSiamese() noexcept = default;


std::pair<bool, bfl::Data> VisualProprioceptionSiamese::measure(const bfl::Data& data) const
{
    return std::make_pair(true, std::move(pImpl_->mask_));
}


std::pair<bool, bfl::Data> VisualProprioceptionSiamese::predictedMeasure(const Ref<const MatrixXd>& cur_states) const
{
    ImplData& rImpl = *pImpl_;

    std::array<double, 3> camera_position;
    std::array<double, 4> camera_orientation;

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
    bool success = false;
    success &= rImpl.si_cad_->superimpose(model_poses, camera_position.data(), camera_orientation.data(), rendered_image);
    if (!success)
        return std::make_pair(false, rendered_image); // initialize with empty if rendering failed 

    // return the rendered image
    return std::make_pair(true, std::move(rendered_image));
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

