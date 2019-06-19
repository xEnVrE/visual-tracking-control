/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef IOU_DEPTH_H
#define IOU_DEPTH_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class IoU_depth : public bfl::LikelihoodModel
{
public:
    IoU_depth(const double likelihood_gain) noexcept;

    ~IoU_depth() noexcept;

    std::pair<bool, Eigen::VectorXd> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXd>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* IOU_DEPTH_H */
