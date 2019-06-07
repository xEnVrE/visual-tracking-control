/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef DRAWPARTICLESIMPORTANCETHRESHOLDSIAMESE_H
#define DRAWPARTICLESIMPORTANCETHRESHOLDSIAMESE_H

#include <BayesFilters/ParticleSet.h>
#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

#include <memory>
#include <random>

namespace bfl {
    class DrawParticlesImportanceThresholdSiamese;
}


class bfl::DrawParticlesImportanceThresholdSiamese : public bfl::PFPrediction
{
public:
    DrawParticlesImportanceThresholdSiamese() noexcept;

    DrawParticlesImportanceThresholdSiamese(DrawParticlesImportanceThresholdSiamese&& pf_prediction) noexcept;

    ~DrawParticlesImportanceThresholdSiamese() noexcept;

    DrawParticlesImportanceThresholdSiamese& operator=(DrawParticlesImportanceThresholdSiamese&& pf_prediction) noexcept;

    StateModel& getStateModel() override;

    void setStateModel(std::unique_ptr<StateModel> state_model) override;

protected:
    void predictStep(const bfl::ParticleSet& prev_particles, bfl::ParticleSet& pred_particles) override;

    std::unique_ptr<StateModel> state_model_;
};

#endif /* DRAWPARTICLESIMPORTANCETHRESHOLDSIAMESE_H */
