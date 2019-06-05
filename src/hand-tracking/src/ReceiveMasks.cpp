/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/Data.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/all.h>
#include <ReceiveMasks.h>


using namespace bfl;
using namespace yarp::os;
using namespace yarp::sig;


ReceiveMasks::ReceiveMasks()
{
    SiamesePort.open("/siamese/read_masks:i");
}


ReceiveMasks::~ReceiveMasks()
{
    SiamesePort.close();
}


Data ReceiveMasks::GetMask()
{
    ImageOf<PixelMono>* inputImagePtr = SiamesePort.read(false);
    
    if (inputImagePtr != nullptr)
    {
        inputImage = *inputImagePtr;
        FirstMask = true;
        return inputImage;
    }
    else
    {
        if (FirstMask)
        {
            return inputImage;
        }
        else
        {
            ImageOf<PixelMono>* inputImagePtr = SiamesePort.read(true);
            inputImage = *inputImagePtr;
            FirstMask = true;
            return inputImage;
        }
    }
}

