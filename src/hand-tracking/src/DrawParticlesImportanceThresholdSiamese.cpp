/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <DrawParticlesImportanceThresholdSiamese.h>

#include <utility>

using namespace bfl;
using namespace Eigen;


DrawParticlesImportanceThresholdSiamese::DrawParticlesImportanceThresholdSiamese() noexcept { }


DrawParticlesImportanceThresholdSiamese::~DrawParticlesImportanceThresholdSiamese() noexcept { }


DrawParticlesImportanceThresholdSiamese::DrawParticlesImportanceThresholdSiamese(DrawParticlesImportanceThresholdSiamese&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)) { };


DrawParticlesImportanceThresholdSiamese& DrawParticlesImportanceThresholdSiamese::operator=(DrawParticlesImportanceThresholdSiamese&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    return *this;
}


StateModel& DrawParticlesImportanceThresholdSiamese::getStateModel()
{
    return *state_model_;
}


void DrawParticlesImportanceThresholdSiamese::setStateModel(std::unique_ptr<StateModel> state_model)
{
    state_model_ = std::move(state_model);
}


void DrawParticlesImportanceThresholdSiamese::predictStep(const ParticleSet& prev_particles, ParticleSet& pred_particles)
{
    VectorXd sorted_cor = prev_particles.weight().array().exp();

    std::sort(sorted_cor.data(), sorted_cor.data() + sorted_cor.size());
    double threshold = sorted_cor.tail(6)(0);

    for (int j = 0; j < prev_particles.weight().size(); ++j)
    {
        VectorXd tmp_state = VectorXd::Zero(prev_particles.state().rows());
        tmp_state = prev_particles.state(j);

        if (!getSkipState() && std::exp(prev_particles.weight(j)) <= threshold)
            state_model_->motion(tmp_state, pred_particles.state(j));
        else
            pred_particles.state(j) = tmp_state;
    }

    pred_particles.weight() = prev_particles.weight();
}
