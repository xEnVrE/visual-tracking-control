/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <InitPoseParticlesSiamese.h>
#include <yarp/eigen/Eigen.h>


using namespace Eigen;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::eigen;


InitPoseParticlesSiamese::InitPoseParticlesSiamese()
{
    SiamesePort.open("/siamese/init_particles:i");
}


InitPoseParticlesSiamese::~InitPoseParticlesSiamese()
{
    SiamesePort.close();
}


Eigen::VectorXd InitPoseParticlesSiamese::readPoseAxisAngle()
{
    Vector *input = SiamesePort.read(true);
    VectorXd Pose;
    
    if (input!=nullptr)
    {
        Pose = toEigen(*input);
    }
    else
    {
        throw(std::runtime_error("Particles initialization was not received!"));
    }

    return Pose;
}
