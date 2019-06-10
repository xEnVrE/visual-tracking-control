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


InitPoseParticlesSiamese::InitPoseParticlesSiamese(const std::string& port_prefix)
{
    SiamesePort.open("/" + port_prefix + "/init_particles:i");
}


InitPoseParticlesSiamese::~InitPoseParticlesSiamese()
{
    SiamesePort.close();
}


Eigen::VectorXd InitPoseParticlesSiamese::readPoseAxisAngle()
{
    VectorXd pose(7);

    pose(0) = 0.00332970008243399;
    pose(1) = 0.0480561846425669;
    pose(2) = 0.470490887414952;
    pose(3) = 0.110957019138624;
    pose(4) = 0.851492242463462;
    pose(5) = -0.512493415497619;
    pose(6) = 2.92096676220956;

    return pose;
}
