/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef RECEIVEDEPTHYCB_H
#define RECEIVEDEPTHYCB_H

#include <yarp/os/BufferedPort.h>
#include <yarp/sig/all.h>
#include <BayesFilters/Data.h>

class ReceiveDepthYCB
{
public:
    ReceiveDepthYCB();

    virtual ~ReceiveDepthYCB();
    
    bfl::Data GetDepth();

private:
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> SiamesePort;

    bool FirstDepth = false;

    yarp::sig::ImageOf<yarp::sig::PixelRgb> inputDepth;

    yarp::sig::ImageOf<yarp::sig::PixelRgb>* inputPtr;
};


#endif /* RECEIVEDEPTHYCB_H */
