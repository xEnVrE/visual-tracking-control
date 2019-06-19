/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <IoU_depth.h>

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


struct IoU_depth::ImplData
{
    double likelihood_gain_;
};


IoU_depth::IoU_depth(const double likelihood_gain) noexcept :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    pImpl_->likelihood_gain_ = likelihood_gain;
}


IoU_depth::~IoU_depth() = default;


std::pair<bool, VectorXd> IoU_depth::likelihood(const MeasurementModel& measurement_model, const Ref<const MatrixXd>& pred_states)
{
    bool valid_measurements;
    Data data_3d_measurements;    
    std::tie(valid_measurements, data_3d_measurements) = measurement_model.measure();

    if (!valid_measurements)
        return std::make_pair(false, VectorXd::Zero(1));

    std::pair<yarp::sig::ImageOf<yarp::sig::PixelMono>,yarp::sig::ImageOf<yarp::sig::PixelFloat>> measurements_pair = any::any_cast<std::pair<yarp::sig::ImageOf<yarp::sig::PixelMono>,yarp::sig::ImageOf<yarp::sig::PixelFloat>>>(data_3d_measurements);
    
    cv::Mat mask_measurements = toCvMat(measurements_pair.first);
    cv::Mat depth_measurements_unprocessed = toCvMat(measurements_pair.second);
    cv::Mat depth_measurements;
    depth_measurements_unprocessed.copyTo(depth_measurements, mask_measurements);

    bool valid_predicted_measurements;
    Data data_3d_predicted_measurements;
    std::tie(valid_predicted_measurements, data_3d_predicted_measurements) = measurement_model.predictedMeasure(pred_states);

    std::pair<cv::Mat,cv::Mat> predicted_measurements_pair = any::any_cast<std::pair<cv::Mat,cv::Mat>>(data_3d_predicted_measurements);
    cv::Mat predicted_mask_measurements = predicted_measurements_pair.first;
    cv::Mat predicted_depth_measurements = predicted_measurements_pair.second;

    try
    {
        cv::cvtColor(predicted_mask_measurements, predicted_mask_measurements, cv::COLOR_RGB2GRAY);
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
    cv::Mat rendered_depth;
    double intersection_sum;
    double union_sum;
    int particle = 0;
    int num_of_pixels;
    cv::Mat distance;
    cv::Scalar_<double> depth_likelihood;

    for (int y = 0; y < predicted_mask_measurements.rows; y += mask_measurements.rows)
    {
        for (int x = 0; x < predicted_mask_measurements.cols; x += mask_measurements.cols)
        {
            rendered_image = cv::Mat(predicted_mask_measurements, cv::Rect(x,y, mask_measurements.cols, mask_measurements.rows));
            rendered_depth = cv::Mat(predicted_depth_measurements, cv::Rect(x,y, depth_measurements.cols, depth_measurements.rows));

            // Apply mask by Siamese Mask R-CNN to the rendered mask
            num_of_pixels = cv::countNonZero(mask_measurements);
            cv::subtract(depth_measurements, rendered_depth, distance,mask_measurements);
            depth_likelihood = cv::sum(distance)/(num_of_pixels*255);

            cv::bitwise_and(mask_measurements,rendered_image,Intersection);
            cv::bitwise_or(mask_measurements,rendered_image,Union);

            intersection_sum = cv::countNonZero(Intersection);
            union_sum = cv::countNonZero(Union);

            likelihood(particle++) = 1 - (intersection_sum/union_sum) - 3*depth_likelihood[0];
        }
    }

    likelihood = (-(pImpl_->likelihood_gain_) * likelihood).array().exp();

    return std::make_pair(true, std::move(likelihood));
}
