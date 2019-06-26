/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/Data.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/all.h>
#include <ReceiveDepthYCB.h>


using namespace bfl;
using namespace yarp::os;
using namespace yarp::sig;


ReceiveDepthYCB::ReceiveDepthYCB()
{
    SiamesePort.open("/siamese/receive_depth:i");
}


ReceiveDepthYCB::~ReceiveDepthYCB()
{
    SiamesePort.close();
}


Data ReceiveDepthYCB::GetDepth()
{
    ImageOf<PixelRgb>* inputPtr = SiamesePort.read(false);
    
    if (inputPtr != nullptr)
    {
        inputDepth = *inputPtr;
        FirstDepth = true;
        return inputDepth;
    }
    else
    {
        if (FirstDepth)
        {
            return inputDepth;
        }
        else
        {
            ImageOf<PixelRgb>* inputPtr = SiamesePort.read(true);
            inputDepth = *inputPtr;
            FirstDepth = true;
            return inputDepth;
        }
    }
}

