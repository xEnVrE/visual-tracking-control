/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <IoU.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#include <yarp/cv/Cv.h>
#include <yarp/sig/all.h>

using namespace bfl;
using namespace Eigen;
using namespace yarp::cv;
using namespace yarp::os;
using namespace yarp::sig;


struct IoU::ImplData
{
    double likelihood_gain_;
};


IoU::IoU(const double likelihood_gain) noexcept :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    pImpl_->likelihood_gain_ = likelihood_gain;
}


IoU::~IoU() = default;


std::pair<bool, VectorXd> IoU::likelihood(const MeasurementModel& measurement_model, const Ref<const MatrixXd>& pred_states)
{
    bool valid_measurements;
    Data data_3d_measurements;
    std::tie(valid_measurements, data_3d_measurements) = measurement_model.measure();

    if (!valid_measurements)
        return std::make_pair(false, VectorXd::Zero(1));

    std::pair<yarp::sig::ImageOf<yarp::sig::PixelMono>,yarp::sig::ImageOf<yarp::sig::PixelFloat>> measurements_pair = any::any_cast<std::pair<yarp::sig::ImageOf<yarp::sig::PixelMono>,yarp::sig::ImageOf<yarp::sig::PixelFloat>>>(data_3d_measurements);
    
    cv::Mat mask_measurements = toCvMat(measurements_pair.first);

    bool valid_predicted_measurements;
    Data data_3d_predicted_measurements;
    std::tie(valid_predicted_measurements, data_3d_predicted_measurements) = measurement_model.predictedMeasure(pred_states);

    std::pair<cv::Mat,cv::Mat> predicted_measurements_pair = any::any_cast<std::pair<cv::Mat,cv::Mat>>(data_3d_predicted_measurements);
    cv::Mat predicted_measurements = predicted_measurements_pair.first;

    try
    {
        cv::cvtColor(predicted_measurements, predicted_measurements, cv::COLOR_RGB2GRAY);
    }
    catch(const any::bad_any_cast& e)
    {
        std::cerr << e.what() << std::endl;

        valid_predicted_measurements = false;
    }

    if (!valid_predicted_measurements)
        return std::make_pair(false, VectorXd::Zero(1));


    VectorXd likelihood(pred_states.cols());
    cv::Mat Intersection;
    cv::Mat Union;
    cv::Mat rendered_image;
    double intersection_sum;
    double union_sum;
    int particle = 0;

    for (int y = 0; y < predicted_measurements.rows; y += mask_measurements.rows)
    {
        for (int x = 0; x < predicted_measurements.cols; x += mask_measurements.cols)
        {
            rendered_image = cv::Mat(predicted_measurements, cv::Rect(x,y, mask_measurements.cols, mask_measurements.rows));

            cv::bitwise_and(mask_measurements,rendered_image,Intersection);
            cv::bitwise_or(mask_measurements,rendered_image,Union);

            intersection_sum = cv::countNonZero(Intersection);
            union_sum = cv::countNonZero(Union);

            likelihood(particle++) = 1 - (intersection_sum/union_sum);
        }
    }

    likelihood = (-(pImpl_->likelihood_gain_) * likelihood).array().exp();

    return std::make_pair(true, std::move(likelihood));
}