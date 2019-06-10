/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef IOU_H
#define IOU_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>

#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Image.h>


class IoU : public bfl::LikelihoodModel
{
public:
    IoU(const double likelihood_gain) noexcept;

    ~IoU() noexcept;

    std::pair<bool, Eigen::VectorXd> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXd>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* IOU_H */
