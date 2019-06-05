/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef VisualProprioceptionSiamese_H
#define VisualProprioceptionSiamese_H

#include <BayesFilters/MeasurementModel.h>
#include <ReceiveMasks.h>

#include <Camera.h>
#include <MeshModel.h>

#include <array>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>

#include <SuperimposeMesh/SICAD.h>


class VisualProprioceptionSiamese : public bfl::MeasurementModel
{
public:
    VisualProprioceptionSiamese(std::unique_ptr<ReceiveMasks> receive_masks, const int num_requested_images, std::unique_ptr<bfl::MeshModel> mesh_model, const std::string object_mesh_path, const std::string shader_folder);

    virtual ~VisualProprioceptionSiamese() noexcept;

    std::pair<bool, bfl::Data> measure(const bfl::Data& data = bfl::Data()) const override;

    std::pair<bool, bfl::Data> predictedMeasure(const Eigen::Ref<const Eigen::MatrixXd>& cur_states) const override;

    std::pair<bool, bfl::Data> innovation(const bfl::Data& predicted_measurements, const bfl::Data& measurements) const override;

    bool freeze() override;

    int getNumberOfUsedParticles() const;

    std::pair<std::size_t, std::size_t> getOutputSize() const override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* VisualProprioceptionSiamese_H */
