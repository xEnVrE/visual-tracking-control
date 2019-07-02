/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef BROWNIANMOTIONPOSE_H
#define BROWNIANMOTIONPOSE_H

#include <functional>
#include <random>

#include <BayesFilters/StateModel.h>

#include <thrift/BrownianMotionPoseIDL.h>

#include <yarp/os/Mutex.h>
#include <yarp/os/Port.h>


class BrownianMotionPose : public bfl::StateModel,
                           public BrownianMotionPoseIDL
{
public:
    BrownianMotionPose(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll, const unsigned int seed) noexcept;

    BrownianMotionPose(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll) noexcept;

    BrownianMotionPose() noexcept;

    BrownianMotionPose(const BrownianMotionPose& bm);

    BrownianMotionPose(BrownianMotionPose&& bm) noexcept;

    ~BrownianMotionPose() noexcept;

    BrownianMotionPose& operator=(const BrownianMotionPose& bm);

    BrownianMotionPose& operator=(BrownianMotionPose&& bm) noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> prop_state) override;

    void motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> mot_state) override;

    Eigen::MatrixXd getNoiseSample(const std::size_t num);

    Eigen::MatrixXd getNoiseCovarianceMatrix() override { return Eigen::MatrixXd::Zero(1, 1); };

    bool setProperty(const std::string& property) override { return false; };

    std::pair<std::size_t, std::size_t> getOutputSize() const override;

protected:
    void updateNoiseDistribution();

    /**
     * Thrift interface for parameters tuning.
     */
    bool enable() override;

    bool disable() override;

    bool set_q_x(const double q) override;

    bool set_q_y(const double q) override;

    bool set_q_z(const double q) override;

    bool set_q_yaw(const double q) override;

    bool set_q_pitch(const double q) override;

    bool set_q_roll(const double q) override;

    std::string get_q_x() override;

    std::string get_q_y() override;

    std::string get_q_z() override;

    std::string get_q_yaw() override;

    std::string get_q_pitch() override;

    std::string get_q_roll() override;

    double                                 q_x_;         /* Noise standard deviation for x 3D position */
    double                                 q_y_;         /* Noise standard deviation for y 3D position */
    double                                 q_z_;         /* Noise standard deviation for z 3D position */
    double                                 q_yaw_;       /* Noise standard deviation for yaw   Euler angle */
    double                                 q_pitch_;     /* Noise standard deviation for pitch Euler angle */
    double                                 q_roll_;      /* Noise standard deviation for roll  Euler angle */

    std::mt19937_64                        generator_;
    std::normal_distribution<double>       distribution_pos_x_;
    std::normal_distribution<double>       distribution_pos_y_;
    std::normal_distribution<double>       distribution_pos_z_;
    std::normal_distribution<double>       distribution_yaw_;
    std::normal_distribution<double>       distribution_pitch_;
    std::normal_distribution<double>       distribution_roll_;
    std::function<double()>                gaussian_random_pos_x_;
    std::function<double()>                gaussian_random_pos_y_;
    std::function<double()>                gaussian_random_pos_z_;
    std::function<double()>                gaussian_random_yaw_;
    std::function<double()>                gaussian_random_pitch_;
    std::function<double()>                gaussian_random_roll_;

    /* Whether the noise generation is enabled or not */
    bool enable_ = true;

    yarp::os::Port port_rpc_command_;

    yarp::os::Mutex rpc_mutex_;
};

#endif /* BROWNIANMOTIONPOSE_H */
