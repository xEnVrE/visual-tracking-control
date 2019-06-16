/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <KinPoseModel.h>

#include <exception>
#include <iostream>
#include <functional>

#include <iCub/ctrl/math.h>

#include <yarp/math/Math.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/Property.h>
#include <yarp/os/LogStream.h>

using namespace bfl;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


KinPoseModel::KinPoseModel() noexcept
{ }


KinPoseModel::~KinPoseModel() noexcept
{ }


void KinPoseModel::propagate(const Ref<const MatrixXd>& cur_state, Ref<MatrixXd> prop_state)
{
    for (std::size_t i = 0; i < cur_state.cols(); i++)
    {
        Transform<double, 3, Affine> pose;
	pose = Translation<double, 3>(cur_state.col(i).head<3>());
	pose.rotate(AngleAxisd(cur_state.col(i)(3), Vector3d::UnitZ()) *
		    AngleAxisd(cur_state.col(i)(4), Vector3d::UnitY()) *
		    AngleAxisd(cur_state.col(i)(5), Vector3d::UnitX()));
	auto transformed = pose * relative_pose_;
	prop_state.col(i).head<3>() = transformed.translation();
	prop_state.col(i).tail<3>() = transformed.rotation().eulerAngles(2, 1, 0);
    }
}


MatrixXd KinPoseModel::getExogenousMatrix()
{
    std::cerr << "ERROR::PFPREDICTION::SETEXOGENOUSMODEL\n";
    std::cerr << "ERROR:\n\tCall to unimplemented base class method." << std::endl;

    return MatrixXd::Zero(1, 1);
}


bool KinPoseModel::setProperty(const std::string& property)
{
    if (property == "kin_pose_delta")
        return setDeltaMotion();

    if (property == "init")
    {
        initialize_delta_ = true;
        return setDeltaMotion();
    }

    return false;
}


std::pair<std::size_t, std::size_t> KinPoseModel::getOutputSize() const
{
    return std::make_pair(3, 3);
}


bool KinPoseModel::setDeltaMotion()
{
    VectorXd ee_pose = readPose();

    if (!initialize_delta_)
    {
        Transform<double, 3, Affine> prev_pose;
	prev_pose = Translation<double, 3>(prev_ee_pose_.head<3>());
	prev_pose.rotate(AngleAxisd(prev_ee_pose_(3), Vector3d::UnitZ()) *
			 AngleAxisd(prev_ee_pose_(4), Vector3d::UnitY()) *
			 AngleAxisd(prev_ee_pose_(5), Vector3d::UnitX()));

	Transform<double, 3, Affine> pose;
	pose = Translation<double, 3>(ee_pose.head<3>());
	pose.rotate(AngleAxisd(ee_pose(3), Vector3d::UnitZ()) *
		    AngleAxisd(ee_pose(4), Vector3d::UnitY()) *
		    AngleAxisd(ee_pose(5), Vector3d::UnitX()));

	relative_pose_ = prev_pose.inverse() * pose;
    }
    else
    {
        relative_pose_ = Transform<double, 3, Affine>::Identity();
        initialize_delta_ = false;
    }

    prev_ee_pose_ = ee_pose;

    return true;
}
