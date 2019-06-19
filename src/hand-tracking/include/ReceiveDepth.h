/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef RECEIVEDEPTH_H
#define RECEIVEDEPTH_H

#include <yarp/os/BufferedPort.h>
#include <yarp/sig/all.h>
#include <BayesFilters/Data.h>

class ReceiveDepth
{
public:
    ReceiveDepth();

    virtual ~ReceiveDepth();
    
    bfl::Data GetDepth();

private:
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelFloat>> SiamesePort;

    bool FirstDepth = false;

    yarp::sig::ImageOf<yarp::sig::PixelFloat> inputDepth;

    yarp::sig::ImageOf<yarp::sig::PixelFloat>* inputPtr;
};


#endif /* RECEIVEDEPTH_H */
