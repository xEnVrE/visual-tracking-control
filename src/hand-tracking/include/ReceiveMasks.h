/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef RECEIVEMASKS_H
#define RECEIVEMASKS_H

#include <yarp/os/BufferedPort.h>
#include <yarp/sig/all.h>
#include <BayesFilters/Data.h>

class ReceiveMasks
{
public:
    ReceiveMasks();

    virtual ~ReceiveMasks();
    
    bfl::Data GetMask();

private:
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelMono>> SiamesePort;

    bool FirstMask = false;

    yarp::sig::ImageOf<yarp::sig::PixelMono> inputImage;

    yarp::sig::ImageOf<yarp::sig::PixelMono>* inputImagePtr;
};


#endif /* RECEIVEMASKS_H */
