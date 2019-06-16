/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef FWDPOSEMODEL_H
#define FWDPOSEMODEL_H

#include <BayesFilters/ExogenousModel.h>

#include <Eigen/Dense>

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/sig/Vector.h>


class KinPoseModel : public bfl::ExogenousModel
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KinPoseModel() noexcept;

    ~KinPoseModel() noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> prop_state) override;

    Eigen::MatrixXd getExogenousMatrix() override;

    bool setProperty(const std::string& property) override;

    std::pair<std::size_t, std::size_t> getOutputSize() const override;

protected:
    virtual Eigen::VectorXd readPose() = 0;

    bool initialize_delta_ = true;

    bool setDeltaMotion();

private:
    Eigen::VectorXd prev_ee_pose_ = Eigen::VectorXd::Zero(6);

    Eigen::Transform<double, 3, Eigen::Affine> relative_pose_ = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
};

#endif /* FWDPOSEMODEL_H */
