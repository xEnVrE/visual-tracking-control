/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef IOU_EXP_H
#define IOU_EXP_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class IoU_exp : public bfl::LikelihoodModel
{
public:
    IoU_exp(const double likelihood_gain_iou, const double likelihood_gain_depth, const double likelihood_step) noexcept;

    ~IoU_exp() noexcept;

    std::pair<bool, Eigen::VectorXd> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXd>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* IOU_EXP_H */
