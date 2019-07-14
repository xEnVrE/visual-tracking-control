/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef RECEIVEGT_H
#define RECEIVEGT_H

#include <yarp/os/BufferedPort.h>
#include <yarp/sig/all.h>
#include <BayesFilters/Data.h>
#include <yarp/eigen/Eigen.h>

using namespace Eigen;

class ReceiveGT
{
public:
    ReceiveGT();

    virtual ~ReceiveGT();
    
    bfl::Data GetGT();

private:
    yarp::os::BufferedPort<yarp::sig::Vector> port_gt_in_;

    bool FirstGT = false;

    yarp::sig::Vector inputGT;

    yarp::sig::Vector* input_ptr;
};


#endif /* RECEIVEGT_H */
