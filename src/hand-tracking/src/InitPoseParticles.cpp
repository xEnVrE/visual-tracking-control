/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <InitPoseParticles.h>

#include <iCub/ctrl/math.h>

#include <chrono>
#include <thread>
#include <iostream>

using namespace bfl;
using namespace Eigen;


InitPoseParticles::InitPoseParticles(std::unique_ptr<Camera> camera) noexcept :
  camera_(std::move(camera))
{ }


bool InitPoseParticles::initialize(ParticleSet& particles)
{
    // Get camera pose
    while(!(camera_->bufferData()))
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::array<double, 3> camera_position;
    std::array<double, 4> camera_orientation;
    std::tie(std::ignore, camera_position, camera_orientation) = bfl::any::any_cast<bfl::Camera::CameraData>(camera_->getData());

    Transform<double, 3, Affine> camera_pose;
    camera_pose = Translation<double, 3>(Vector3d(camera_position.data()));
    camera_pose.rotate(AngleAxisd(camera_orientation[3], Vector3d(camera_orientation.data())));

    // Express the pose in camera frame
    VectorXd pose = readPose();
    VectorXd rotated_pose(6);
    Transform<double, 3, Affine> transform;
    transform = Translation<double, 3>(pose.head<3>());
    transform.rotate(AngleAxisd(pose(3), Vector3d::UnitZ()) *
		     AngleAxisd(pose(4), Vector3d::UnitY()) *
		     AngleAxisd(pose(5), Vector3d::UnitX()));
    auto cam_to_object = camera_pose.inverse() * transform;
    rotated_pose.head<3>() = cam_to_object.translation();
    rotated_pose.tail<3>() = cam_to_object.rotation().eulerAngles(2, 1, 0);

    // Assign pose
    for (int i = 0; i < particles.state().cols(); ++i)
        particles.state(i) << rotated_pose;

    particles.weight().fill(-std::log(particles.state().cols()));

    return true;
}
