#include <BrownianMotionPoseWithTimeDynamics.h>

#include <cmath>
#include <iostream>
#include <utility>

#include <BayesFilters/directional_statistics.h>
#include <BayesFilters/utils.h>

using namespace bfl;
using namespace Eigen;


struct BrownianMotionPoseWithTimeDynamics::ImplData
{
    enum class Modality
    {
        Iteration,
        Time
    };

    Modality modality_;

    unsigned int iterations_;

    unsigned int current_iterations_ = 0;

    double seconds_;

    double current_seconds_ = 0.0;

    bfl::utils::CpuTimer<> timer_;
};


BrownianMotionPoseWithTimeDynamics::BrownianMotionPoseWithTimeDynamics(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll, const int max_iterations) noexcept :
    BrownianMotionPoseWithTimeDynamics(q_x, q_y, q_z, q_yaw, q_pitch, q_roll, max_iterations, 1)
{ }


BrownianMotionPoseWithTimeDynamics::BrownianMotionPoseWithTimeDynamics(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll, const double max_seconds) noexcept :
    BrownianMotionPoseWithTimeDynamics(q_x, q_y, q_z, q_yaw, q_pitch, q_roll, max_seconds, 1)
{ }


BrownianMotionPoseWithTimeDynamics::BrownianMotionPoseWithTimeDynamics(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll, const int max_iterations, const unsigned int seed) noexcept :
    BrownianMotionPoseWithTimeDynamics(q_x, q_y, q_z, q_yaw, q_pitch, q_roll, seed)
{
    ImplData& rImpl = *pImpl_;


    rImpl.modality_ = ImplData::Modality::Iteration;

    rImpl.iterations_ = max_iterations;

    rImpl.seconds_ = std::numeric_limits<double>::infinity();
}


BrownianMotionPoseWithTimeDynamics::BrownianMotionPoseWithTimeDynamics(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll, const double max_seconds, const unsigned int seed) noexcept :
    BrownianMotionPoseWithTimeDynamics(q_x, q_y, q_z, q_yaw, q_pitch, q_roll, seed)
{
    ImplData& rImpl = *pImpl_;


    rImpl.modality_ = ImplData::Modality::Time;

    rImpl.iterations_ = std::numeric_limits<int>::infinity();

    rImpl.seconds_ = std::abs(max_seconds);

    if (max_seconds < 0)
    {
        std::cerr << "WARNING::BROWNIANMOTIONWITHTIMEDYNAMICS::CTOR\n";
        std::cerr << "WARNING::LOG:\n\tInput parameter `max_seconds` is negative. Used as positive.\n";
        std::cerr << "WARNING::LOG:\n\tProvided: " << max_seconds << ". Used " << rImpl.seconds_ << "." << std::endl;
    }
}


BrownianMotionPoseWithTimeDynamics::BrownianMotionPoseWithTimeDynamics(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll, const unsigned int seed) noexcept :
    q_x_(q_x),
    q_y_(q_y),
    q_z_(q_z),
    q_yaw_(q_yaw),
    q_pitch_(q_pitch),
    q_roll_(q_roll),
    generator_(std::mt19937_64(seed)),
    distribution_pos_x_(std::normal_distribution<double>(0.0, q_x)),
    distribution_pos_y_(std::normal_distribution<double>(0.0, q_y)),
    distribution_pos_z_(std::normal_distribution<double>(0.0, q_z)),
    distribution_yaw_(std::normal_distribution<double>(0.0, q_yaw_)),
    distribution_pitch_(std::normal_distribution<double>(0.0, q_pitch_)),
    distribution_roll_(std::normal_distribution<double>(0.0, q_roll_)),
    gaussian_random_pos_x_([&] { return (distribution_pos_x_)(generator_); }),
    gaussian_random_pos_y_([&] { return (distribution_pos_y_)(generator_); }),
    gaussian_random_pos_z_([&] { return (distribution_pos_z_)(generator_); }),
    gaussian_random_yaw_([&] { return (distribution_yaw_)(generator_); }),
    gaussian_random_pitch_([&] { return (distribution_pitch_)(generator_); }),
    gaussian_random_roll_([&] { return (distribution_roll_)(generator_); }),
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{ }


BrownianMotionPoseWithTimeDynamics::BrownianMotionPoseWithTimeDynamics(const BrownianMotionPoseWithTimeDynamics& brown) :
    q_x_(brown.q_x_),
    q_y_(brown.q_y_),
    q_z_(brown.q_z_),
    q_yaw_(brown.q_yaw_),
    q_pitch_(brown.q_pitch_),
    q_roll_(brown.q_roll_),
    generator_(brown.generator_),
    distribution_pos_x_(brown.distribution_pos_x_),
    distribution_pos_y_(brown.distribution_pos_y_),
    distribution_pos_z_(brown.distribution_pos_z_),
    distribution_yaw_(brown.distribution_yaw_),
    distribution_pitch_(brown.distribution_pitch_),
    distribution_roll_(brown.distribution_roll_),
    gaussian_random_pos_x_(brown.gaussian_random_pos_x_),
    gaussian_random_pos_y_(brown.gaussian_random_pos_y_),
    gaussian_random_pos_z_(brown.gaussian_random_pos_z_),
    gaussian_random_yaw_(brown.gaussian_random_yaw_),
    gaussian_random_pitch_(brown.gaussian_random_pitch_),
    gaussian_random_roll_(brown.gaussian_random_roll_) { }

BrownianMotionPoseWithTimeDynamics::BrownianMotionPoseWithTimeDynamics(BrownianMotionPoseWithTimeDynamics&& brown) noexcept :
    q_x_(brown.q_x_),
    q_y_(brown.q_y_),
    q_z_(brown.q_z_),
    q_yaw_(brown.q_yaw_),
    q_pitch_(brown.q_pitch_),
    q_roll_(brown.q_roll_),
    generator_(std::move(brown.generator_)),
    distribution_pos_x_(std::move(brown.distribution_pos_x_)),
    distribution_pos_y_(std::move(brown.distribution_pos_y_)),
    distribution_pos_z_(std::move(brown.distribution_pos_z_)),
    distribution_yaw_(std::move(brown.distribution_yaw_)),
    distribution_pitch_(std::move(brown.distribution_pitch_)),
    distribution_roll_(std::move(brown.distribution_roll_)),
    gaussian_random_pos_x_(std::move(brown.gaussian_random_pos_x_)),
    gaussian_random_pos_y_(std::move(brown.gaussian_random_pos_y_)),
    gaussian_random_pos_z_(std::move(brown.gaussian_random_pos_z_)),
    gaussian_random_yaw_(std::move(brown.gaussian_random_yaw_)),
    gaussian_random_pitch_(std::move(brown.gaussian_random_pitch_)),
    gaussian_random_roll_(std::move(brown.gaussian_random_roll_))
{
    brown.q_x_        = 0.0;
    brown.q_y_        = 0.0;
    brown.q_z_        = 0.0;
    brown.q_yaw_      = 0.0;
    brown.q_pitch_    = 0.0;
    brown.q_roll_     = 0.0;
}


BrownianMotionPoseWithTimeDynamics::~BrownianMotionPoseWithTimeDynamics() noexcept { }


BrownianMotionPoseWithTimeDynamics& BrownianMotionPoseWithTimeDynamics::operator=(const BrownianMotionPoseWithTimeDynamics& brown)
{
    BrownianMotionPoseWithTimeDynamics tmp(brown);
    *this = std::move(tmp);

    return *this;
}


BrownianMotionPoseWithTimeDynamics& BrownianMotionPoseWithTimeDynamics::operator=(BrownianMotionPoseWithTimeDynamics&& brown) noexcept
{
    q_x_        = brown.q_x_;
    q_y_        = brown.q_y_;
    q_z_        = brown.q_z_;
    q_yaw_      = brown.q_yaw_;
    q_pitch_    = brown.q_pitch_;
    q_roll_     = brown.q_roll_;

    generator_              = std::move(brown.generator_);
    distribution_pos_x_     = std::move(brown.distribution_pos_x_);
    distribution_pos_y_     = std::move(brown.distribution_pos_y_);
    distribution_pos_z_     = std::move(brown.distribution_pos_z_);
    distribution_yaw_       = std::move(brown.distribution_yaw_);
    distribution_pitch_     = std::move(brown.distribution_pitch_);
    distribution_roll_       = std::move(brown.distribution_roll_);
    gaussian_random_pos_x_  = std::move(brown.gaussian_random_pos_x_);
    gaussian_random_pos_y_  = std::move(brown.gaussian_random_pos_y_);
    gaussian_random_pos_z_  = std::move(brown.gaussian_random_pos_z_);
    gaussian_random_yaw_  = std::move(brown.gaussian_random_yaw_);
    gaussian_random_pitch_  = std::move(brown.gaussian_random_pitch_);
    gaussian_random_roll_  = std::move(brown.gaussian_random_roll_);

    brown.q_x_        = 0.0;
    brown.q_y_        = 0.0;
    brown.q_z_        = 0.0;
    brown.q_yaw_      = 0.0;
    brown.q_pitch_    = 0.0;
    brown.q_roll_     = 0.0;

    return *this;
}


void BrownianMotionPoseWithTimeDynamics::propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> prop_state)
{
    prop_state = cur_state;
}


void BrownianMotionPoseWithTimeDynamics::motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> mot_state)
{
    propagate(cur_state, mot_state);

    MatrixXd sample(6, mot_state.cols());
    sample = getNoiseSample(mot_state.cols());

    mot_state.topRows<3>() += sample.topRows<3>();
    mot_state.bottomRows<3>() = directional_statistics::directional_add(mot_state.bottomRows<3>(), sample.bottomRows<3>());
}


Eigen::MatrixXd BrownianMotionPoseWithTimeDynamics::getNoiseSample(const std::size_t num)
{
    ImplData& rImpl = *pImpl_;


    double damper = 1.0;

    switch (rImpl.modality_)
    {
        case ImplData::Modality::Iteration:
        {
            damper = (rImpl.current_iterations_ <= rImpl.iterations_) ? std::exp(-rImpl.current_iterations_) : 0.0;

            break;
        }

        case ImplData::Modality::Time:
        {
            std::cout << rImpl.current_seconds_ << std::endl;
            double tau = 5.0;
            damper = (rImpl.current_seconds_ <= rImpl.seconds_) ? std::exp(-rImpl.current_seconds_ / tau) : 0.0;

            break;
        }

        default:
            return getUndampedNoiseSample(num);
    }

    std::cout << damper << std::endl;

    return getUndampedNoiseSample(num) * damper;
}


bool BrownianMotionPoseWithTimeDynamics::setProperty(const std::string& property)
{
    ImplData& rImpl = *pImpl_;


    if (property == "tdd_reset")
    {
        switch (rImpl.modality_)
        {
        case ImplData::Modality::Iteration:
        {
            rImpl.current_iterations_ = 0;

            break;
        }

        case ImplData::Modality::Time:
        {
            rImpl.timer_.stop();

            break;
        }

        default:
            return false;
        }

        return true;
    }


    if (property == "tdd_advance")
    {
        switch (rImpl.modality_)
        {
        case ImplData::Modality::Iteration:
        {
            ++rImpl.current_iterations_;

            break;
        }

        case ImplData::Modality::Time:
        {
            rImpl.current_seconds_ = rImpl.timer_.elapsed() / 1000.0;

            if (!rImpl.timer_.is_running())
                rImpl.timer_.start();

            break;
        }

        default:
            return false;
        }

        return true;
    }
}


Eigen::MatrixXd BrownianMotionPoseWithTimeDynamics::getUndampedNoiseSample(const std::size_t num)
{
    MatrixXd sample(6, num);

    for (std::size_t i = 0; i < num; ++i)
    {
        sample(0, i) = gaussian_random_pos_x_();
        sample(1, i) = gaussian_random_pos_y_();
        sample(2, i) = gaussian_random_pos_z_();
        sample(3, i) = gaussian_random_yaw_();
        sample(4, i) = gaussian_random_pitch_();
        sample(5, i) = gaussian_random_roll_();
    }

    return sample;
}


std::pair<std::size_t, std::size_t> BrownianMotionPoseWithTimeDynamics::getOutputSize() const
{
    return std::make_pair(3, 3);
}
