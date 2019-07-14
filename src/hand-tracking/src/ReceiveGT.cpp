/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/Data.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/all.h>
#include <ReceiveGT.h>
#include <iostream>


using namespace bfl;
using namespace yarp::os;
using namespace yarp::sig;


ReceiveGT::ReceiveGT()
{
    port_gt_in_.open("/object-tracking-siamese/ground-truth:i");
}


ReceiveGT::~ReceiveGT()
{
    port_gt_in_.close();
}


Data ReceiveGT::GetGT()
{
    input_ptr = port_gt_in_.read(false);

    if (input_ptr != nullptr)
    {
        inputGT = *input_ptr;
        FirstGT = true;
        return inputGT;
    }
    else
    {
        if (FirstGT)
        {
            return inputGT;
        }
        else
        {
            input_ptr = port_gt_in_.read(true);
            inputGT = *input_ptr;
            FirstGT = true;
            return inputGT;
        }
    }
}

