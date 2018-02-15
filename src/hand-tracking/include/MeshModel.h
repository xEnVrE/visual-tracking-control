#ifndef MESHMODEL_H
#define MESHMODEL_H

#include <tuple>

#include <Eigen/Dense>
#include <SuperimposeMesh/SICAD.h>

namespace bfl {
    class MeshModel;
}


class bfl::MeshModel
{
public:
    virtual ~MeshModel() noexcept { };

    virtual std::tuple<bool, SICAD::ModelPathContainer> readMeshPaths() = 0;

    virtual std::tuple<bool, std::string> readShaderPaths() = 0;

    virtual std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> getModelPose(const Eigen::Ref<const Eigen::MatrixXf>& cur_states) = 0;
};

#endif /* MESHMODEL_H */
