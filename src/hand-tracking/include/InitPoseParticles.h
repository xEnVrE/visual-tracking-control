/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef INITPOSEPARTICLES_H
#define INITPOSEPARTICLES_H

#include <BayesFilters/ParticleSet.h>
#include <BayesFilters/ParticleSetInitialization.h>

#include <Camera.h>

#include <iCub/iKin/iKinFwd.h>

#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>


class InitPoseParticles : public bfl::ParticleSetInitialization
{
public:
    InitPoseParticles(std::unique_ptr<bfl::Camera> camera) noexcept;

    virtual ~InitPoseParticles() noexcept { };

    bool initialize(bfl::ParticleSet& particles) override;

protected:
    virtual Eigen::VectorXd readPose() = 0;

private:
    std::unique_ptr<bfl::Camera> camera_;
};


#endif /* INITPOSEPARTICLES_H */
