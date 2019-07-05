/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <IoU_exp.h>

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
using namespace std;


struct IoU_exp::ImplData
{
    double likelihood_gain_iou_;
    double likelihood_gain_depth_;
    double likelihood_step_;
};


IoU_exp::IoU_exp(const double likelihood_gain_iou, const double likelihood_gain_depth, const double likelihood_step) noexcept :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    pImpl_->likelihood_gain_iou_ = likelihood_gain_iou;
    pImpl_->likelihood_gain_depth_ = likelihood_gain_depth;
    pImpl_->likelihood_step_ = likelihood_step;
}


IoU_exp::~IoU_exp() = default;


std::pair<bool, VectorXd> IoU_exp::likelihood(const MeasurementModel& measurement_model, const Ref<const MatrixXd>& pred_states)
{
    bool valid_measurements;
    Data data_3d_measurements;    
    std::tie(valid_measurements, data_3d_measurements) = measurement_model.measure();

    if (!valid_measurements)
        return std::make_pair(false, VectorXd::Zero(1));

    std::pair<yarp::sig::ImageOf<yarp::sig::PixelMono>,yarp::sig::ImageOf<yarp::sig::PixelFloat>> measurements_pair = any::any_cast<std::pair<yarp::sig::ImageOf<yarp::sig::PixelMono>,yarp::sig::ImageOf<yarp::sig::PixelFloat>>>(data_3d_measurements);
    
    cv::Mat mask_measurements = toCvMat(measurements_pair.first);
    cv::Mat depth_measurements_unprocessed = toCvMat(measurements_pair.second);

    cv::Mat mask2 = mask_measurements;

    // Find the BBox
    cv::Rect bbox = boundingRect(mask2);

    cv::Point pt1, pt2;
    pt1.x = bbox.x;
    pt1.y = bbox.y;
    pt2.x = bbox.x + bbox.width;
    pt2.y = bbox.y + bbox.height;

    // Apply mask from Siamese Mask R-CNN to the depth map obtained from Camera and save it in depth_measurements
    cv::Mat depth_measurements;
    depth_measurements_unprocessed.copyTo(depth_measurements, mask_measurements);

    bool valid_predicted_measurements;
    Data data_3d_predicted_measurements;
    std::tie(valid_predicted_measurements, data_3d_predicted_measurements) = measurement_model.predictedMeasure(pred_states);

    std::pair<cv::Mat,cv::Mat> predicted_measurements_pair = any::any_cast<std::pair<cv::Mat,cv::Mat>>(data_3d_predicted_measurements);
    cv::Mat predicted_mask_measurements = predicted_measurements_pair.first;
    cv::Mat predicted_depth_measurements = predicted_measurements_pair.second;

    // Invert the rendered depth's background (from 1024 to 0)
    predicted_depth_measurements.setTo(0, predicted_depth_measurements == 1024);

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


    VectorXd likelihood(pred_states.cols()), likelihood_iou(pred_states.cols()), likelihood_depth(pred_states.cols());
    cv::Mat Intersection, Union, rendered_image, rendered_depth, distance, depth_difference, depth_intersection;

    double intersection_sum, union_sum, print_sum, particle_depth_likelihood = 0, depth_sum = 0;
    int num_of_pixels, particle = 0;


    for (int y = 0; y < predicted_mask_measurements.rows; y += mask_measurements.rows)
    {
        for (int x = 0; x < predicted_mask_measurements.cols; x += mask_measurements.cols)
        {
            rendered_image = cv::Mat(predicted_mask_measurements, cv::Rect(x,y, mask_measurements.cols, mask_measurements.rows));
            rendered_depth = cv::Mat(predicted_depth_measurements, cv::Rect(x,y, depth_measurements.cols, depth_measurements.rows));

            // Depth part

            // Find depth intersection mask
            cv::bitwise_and(depth_measurements,rendered_depth,depth_intersection); 
            // Find depth difference
            depth_difference = cv::Mat::zeros(cv::Size(depth_measurements.rows, depth_measurements.cols), CV_64FC1);
            for(int i=0;i<depth_measurements.rows;i++)
                for(int j=0;j<depth_measurements.cols;j++)
                    if(depth_intersection.at<float>(i,j) != 0)
                        {
                        depth_difference.at<float>(i,j) = cv::abs(rendered_depth.at<float>(i,j) - depth_measurements.at<float>(i,j));
                        depth_sum += depth_difference.at<float>(i,j);
                        }

            num_of_pixels = cv::countNonZero(depth_difference);
            
            if (num_of_pixels != 0)
                particle_depth_likelihood = depth_sum/num_of_pixels;
            else
                particle_depth_likelihood = 1.0;

            print_sum = depth_sum;
            depth_sum = 0;

            // IoU part
            cv::bitwise_and(mask_measurements,rendered_image,Intersection);
            cv::bitwise_or(mask_measurements,rendered_image,Union);

            intersection_sum = cv::countNonZero(Intersection);
            union_sum = cv::countNonZero(Union);

            likelihood_iou(particle) = 1 - (intersection_sum/union_sum);
            likelihood_depth(particle++) = particle_depth_likelihood;
        }
    }

    likelihood_iou = (-(pImpl_->likelihood_gain_iou_) * likelihood_iou).array().exp();
    likelihood_depth = (-(pImpl_->likelihood_gain_depth_) * likelihood_depth).array().exp();
//    likelihood_iou.normalize();
//    likelihood_depth.normalize();

    likelihood = likelihood_iou + likelihood_depth;
//likelihood = (-(pImpl_->likelihood_gain_) * (likelihood_iou)).array().exp();
// Try to normalize the total likelihood!
// likelihood.normalize();


    (pImpl_->likelihood_step_)++;
    if (int(pImpl_->likelihood_step_) % 100 == 0)
    {
        const char* depth_diff = "depth_diff.txt";
        const char* real = "real_depth.txt";
        const char* lik = "Likelihood.txt";
        const char* intersect = "intersection.txt";

        ofstream fout1(depth_diff);
        ofstream fout2(real);
        ofstream fout3(lik);
        ofstream fout4(intersect);
        
        for(int i=0; i<mask_measurements.rows; i++)
        {
            for(int j=0; j<mask_measurements.cols; j++)
            {
                fout1<<depth_difference.at<float>(i,j)<<"\t";
                fout2<<depth_measurements.at<float>(i,j)<<"\t";
                fout4<<depth_intersection.at<float>(i,j)<<"\t";
            }

            fout1<<endl;
            fout2<<endl;
            fout4<<endl;
        }

        fout3<<"Distance sum: "<<print_sum<<" Number of pixels: "<<num_of_pixels<<" Particle depth likelihood: "<<particle_depth_likelihood<< "Pt1.x: "<< pt1.x << "Pt1.y: "<< pt1.y << "Pt2.x: "<< pt2.x << "Pt2.y: "<< pt2.y <<"\nIoU likelihood:\n"<<likelihood_iou<<"\nDepth likelihood:\n"<<likelihood_depth<<"\nTotal likelihood:\n"<<likelihood<<endl;

        fout1.close();
        fout2.close();
        fout3.close();
        fout4.close();
    }

    return std::make_pair(true, std::move(likelihood));
}
