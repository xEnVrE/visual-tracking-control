/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef INITPOSEPARTICLESSIAMESE_H
#define INITPOSEPARTICLESSIAMESE_H

#include <Eigen/Dense>
#include <InitPoseParticlesAxisAngle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>


class InitPoseParticlesSiamese : public InitPoseParticlesAxisAngle
{
public:
    InitPoseParticlesSiamese(const std::string& port_prefix);

    virtual ~InitPoseParticlesSiamese();
    
protected:
    Eigen::VectorXd readPoseAxisAngle() override;

private:
    yarp::os::BufferedPort<yarp::sig::Vector> SiamesePort;
};


#endif /* INITPOSEPARTICLESSIAMESE_H */
