/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <string>

#include <BayesFilters/BootstrapCorrection.h>
#include <BayesFilters/ResamplingWithPrior.h>

#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Value.h>

#include <opencv2/core/core.hpp>

#include <BrownianMotionPose.h>
#include <ChiSquare.h>
#include <DrawParticlesImportanceThresholdSiamese.h>
#include <iCubCamera.h>
#include <iCubArmModel.h>
#include <iCubGatePose.h>
#include <iCubFwdKinModel.h>
#include <InitiCubArm.h>
#include <InitPoseParticlesSiamese.h>
#include <InitWalkmanArm.h>
#include <IoU.h>
#include <IoU_depth.h>
#include <KLD.h>
#include <NormOne.h>
#include <NormTwo.h>
#include <NormTwoChiSquare.h>
#include <NormTwoKLD.h>
#include <NormTwoKLDChiSquare.h>
#include <PlayiCubFwdKinModel.h>
#include <PlayWalkmanPoseModel.h>
#include <PlayGatePose.h>
#include <ReceiveMasks.h>
#include <ReceiveDepth.h>
#include <VisualProprioception.h>
#include <VisualProprioceptionSiamese.h>
#include <VisualSIS.h>
#include <WalkmanArmModel.h>
#include <WalkmanCamera.h>

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace yarp::os;


/* MAIN */
int main(int argc, char *argv[])
{
    const std::string log_ID = "[Main]";
    yInfo() << log_ID << "Configuring and starting module...";

    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError() << "YARP seems unavailable!";
        return EXIT_FAILURE;
    }

    ResourceFinder rf;
    rf.setVerbose();
    rf.setDefaultContext("object-tracking-siamese");
    rf.setDefaultConfigFile("config.ini");
    rf.configure(argc, argv);

/*
    // Path to shader directory
    rf.setDefaultContext(context + "/shader");
    const std::string shader_path_ = rf.findFileByName("shader_model.vert");
    if (!file_found(shader_path_))
        throw std::runtime_error("ERROR::CTOR::DIR\nERROR: shader directory not found!");

    size_t rfind_slash = shader_path_.rfind("/");
    if (rfind_slash == std::string::npos)
        rfind_slash = 0;
    size_t rfind_backslash = shader_path_.rfind("\\");
    if (rfind_backslash == std::string::npos)
        rfind_backslash = 0;
    shader_path_ = shader_path_.substr(0, rfind_slash > rfind_backslash ? rfind_slash : rfind_backslash);


    // Path to mesh directory
    rf.setDefaultContext(context + "/mesh");
    const std::string object_mesh_path_ = rf.findFileByName("object.obj");
    if (!file_found(object_mesh_path_))
        throw std::runtime_error("ERROR::CTOR::DIR\nERROR: mesh directory not found!");

    size_t rfind_slash = object_mesh_path_.rfind("/");
    if (rfind_slash == std::string::npos)
        rfind_slash = 0;
    size_t rfind_backslash = object_mesh_path_.rfind("\\");
    if (rfind_backslash == std::string::npos)
        rfind_backslash = 0;
    object_mesh_path_ = object_mesh_path_.substr(0, rfind_slash > rfind_backslash ? rfind_slash : rfind_backslash);
*/

    FilteringParamtersD paramsd;
    FilteringParamtersS paramss;

    /* Get Particle Filter parameters */
    yarp::os::Bottle bottle_pf_params = rf.findGroup("PF");
    paramsd["num_particles"]    = bottle_pf_params.check("num_particles",    Value(50)).asInt();
    paramsd["gpu_count"]        = bottle_pf_params.check("gpu_count",        Value(1.0)).asInt();
    paramsd["resample_prior"]   = bottle_pf_params.check("resample_prior",   Value(1.0)).asInt();
    paramsd["gate_pose"]        = bottle_pf_params.check("gate_pose",        Value(0.0)).asInt();
    paramsd["resolution_ratio"] = bottle_pf_params.check("resolution_ratio", Value(1.0)).asInt();
    paramss["laterality"]       = bottle_pf_params.check("laterality",       Value("right")).asString();

    paramsd["num_images"]       = paramsd["num_particles"] / paramsd["gpu_count"];

    if (rf.check("play"))
        paramsd["play"] = 1.0;
    else
        paramsd["play"] = bottle_pf_params.check("play", Value(1.0)).asDouble();

    if (rf.check("robot"))
        paramss["robot"] = rf.find("robot").asString();
    else
        paramss["robot"] = bottle_pf_params.check("robot", Value("icub")).asString();

    if (rf.check("cam"))
        paramss["cam_sel"] = rf.find("cam").asString();
    else
        paramss["cam_sel"] = bottle_pf_params.check("cam_sel", Value("left")).asString();


    /* Get Brownian Motion parameters */
    yarp::os::Bottle bottle_brownianmotion_params = rf.findGroup("BROWNIANMOTION");
    paramsd["q_x"]     = bottle_brownianmotion_params.check("q_x",     Value(0.005)).asDouble();
    paramsd["q_y"]     = bottle_brownianmotion_params.check("q_y",     Value(0.005)).asDouble();
    paramsd["q_z"]     = bottle_brownianmotion_params.check("q_z",     Value(0.005)).asDouble();
    paramsd["q_yaw"]   = bottle_brownianmotion_params.check("q_yaw",   Value(0.001)).asDouble();
    paramsd["q_pitch"] = bottle_brownianmotion_params.check("q_pitch", Value(0.001)).asDouble();
    paramsd["q_roll"]  = bottle_brownianmotion_params.check("q_roll",  Value(0.001)).asDouble();

    /* Get Visual Proprioception parameters */
    yarp::os::Bottle bottle_visualproprioception_params = rf.findGroup("VISUALPROPRIOCEPTION");
    paramsd["use_thumb"]   = bottle_visualproprioception_params.check("use_thumb", Value(0.0)).asDouble();
    paramsd["use_forearm"] = bottle_visualproprioception_params.check("use_forearm", Value(0.0)).asDouble();
    paramss["object_name"] = bottle_visualproprioception_params.check("object_name", Value("mustard_bottle")).asString();
    // paramss["object_mesh_path"] = bottle_visualproprioception_params.check("object_mesh_path", Value("/home/yuriy/robot-code/visual-tracking-control/src/hand-tracking/mesh/object.obj")).asString();
    // paramss["shader_path"] = bottle_visualproprioception_params.check("shader_path", Value("/home/yuriy/robot-code/visual-tracking-control/src/hand-tracking/shader")).asString();


    /* Get Likelihood parameters */
    yarp::os::Bottle bottle_likelihood_params = rf.findGroup("LIKELIHOOD");
    paramss["likelihood_type"] = bottle_likelihood_params.check("likelihood_type", Value("norm_one")).asString();
    paramsd["likelihood_gain"] = bottle_likelihood_params.check("likelihood_gain", Value(0.001)).asDouble();


    /* Get Gate Pose parameters */
    yarp::os::Bottle bottle_gatepose_params = rf.findGroup("GATEPOSE");
    paramsd["gate_x"]        = bottle_gatepose_params.check("gate_x",        Value(0.1)).asDouble();
    paramsd["gate_y"]        = bottle_gatepose_params.check("gate_y",        Value(0.1)).asDouble();
    paramsd["gate_z"]        = bottle_gatepose_params.check("gate_z",        Value(0.1)).asDouble();
    paramsd["gate_aperture"] = bottle_gatepose_params.check("gate_aperture", Value(15.0)).asDouble();
    paramsd["gate_rotation"] = bottle_gatepose_params.check("gate_rotation", Value(30.0)).asDouble();


    /* Get Resampling parameters */
    yarp::os::Bottle bottle_resampling_params = rf.findGroup("RESAMPLING");
    paramsd["resample_ratio"] = bottle_resampling_params.check("resample_ratio", Value(0.3)).asDouble();
    paramsd["prior_ratio"]    = bottle_resampling_params.check("prior_ratio",    Value(0.5)).asDouble();


    /* Log parameters */
    yInfo() << log_ID << "General PF parameters:";
    yInfo() << log_ID << " - robot:"          << paramss["robot"];
    yInfo() << log_ID << " - cam_sel:"        << paramss["cam_sel"];
    yInfo() << log_ID << " - laterality:"     << paramss["laterality"];
    yInfo() << log_ID << " - num_particles:"  << paramsd["num_particles"];
    yInfo() << log_ID << " - gpu_count:"      << paramsd["gpu_count"];
    yInfo() << log_ID << " - num_images:"     << paramsd["num_images"];
    yInfo() << log_ID << " - resample_prior:" << paramsd["resample_prior"];
    yInfo() << log_ID << " - gate_pose:"      << paramsd["gate_pose"];
    yInfo() << log_ID << " - play:"           << (paramsd["play"] == 1.0 ? "true" : "false");

    yInfo() << log_ID << "Motion modle parameters:";
    yInfo() << log_ID << " - q_x:"        << paramsd["q_x"];
    yInfo() << log_ID << " - q_y:"        << paramsd["q_y"];
    yInfo() << log_ID << " - q_z:"        << paramsd["q_z"];
    yInfo() << log_ID << " - q_yaw:"      << paramsd["q_yaw"];
    yInfo() << log_ID << " - q_pitch:"    << paramsd["q_pitch"];
    yInfo() << log_ID << " - q_roll:"      << paramsd["q_roll"];

    yInfo() << log_ID << "Sensor model parameters:";
    yInfo() << log_ID << " - use_thumb:"   << paramsd["use_thumb"];
    yInfo() << log_ID << " - use_forearm:" << paramsd["use_forearm"];

    yInfo() << log_ID << "Correction parameters:";
    yInfo() << log_ID << " - likelihood_type:" << paramss["likelihood_type"];
    yInfo() << log_ID << " - likelihood_gain:" << paramsd["likelihood_gain"];

    yInfo() << log_ID << "Resampling parameters:";
    yInfo() << log_ID << " - resample_ratio:" << paramsd["resample_ratio"];

    if (paramsd["resample_prior"] == 1.0)
    {
        yInfo() << log_ID << "Resampling with prior parameters:";
        yInfo() << log_ID << " - prior_ratio:" << paramsd["prior_ratio"];
    }

    if (paramsd["gate_pose"] == 1.0)
    {
        yInfo() << log_ID << "Pose gating parameters:";
        yInfo() << log_ID << " - gate_x:"        << paramsd["gate_x"];
        yInfo() << log_ID << " - gate_y:"        << paramsd["gate_y"];
        yInfo() << log_ID << " - gate_z:"        << paramsd["gate_z"];
        yInfo() << log_ID << " - gate_aperture:" << paramsd["gate_aperture"];
        yInfo() << log_ID << " - gate_rotation:" << paramsd["gate_rotation"];
    }


    /* INITIALIZATION */
    std::unique_ptr<ParticleSetInitialization> init_arm;
    if (paramss["robot"] == "icub")
        init_arm = std::unique_ptr<InitiCubArm>(new InitiCubArm(paramss["laterality"], "handTracking/InitiCubArm/" + paramss["cam_sel"]));
    else if (paramss["robot"] == "walkman")
        init_arm = std::unique_ptr<InitWalkmanArm>(new InitWalkmanArm(paramss["laterality"], "handTracking/InitWalkmanArm/" + paramss["cam_sel"]));

    // Added initialization for Siamese Model
    else if (paramss["robot"] == "siamese")
        init_arm = std::unique_ptr<InitPoseParticlesSiamese>(new InitPoseParticlesSiamese("siamese"));


    /* MOTION MODEL */
    std::unique_ptr<StateModel> brown(new BrownianMotionPose(paramsd["q_x"], paramsd["q_y"], paramsd["q_z"], paramsd["q_yaw"], paramsd["q_pitch"], paramsd["q_roll"], paramsd["seed"]));


    /* PREDICTION */
    std::unique_ptr<DrawParticlesImportanceThresholdSiamese> pf_prediction(new DrawParticlesImportanceThresholdSiamese());
    pf_prediction->setStateModel(std::move(brown));


    /* PROCESS MODEL */
    std::unique_ptr<Camera> camera;
    std::unique_ptr<MeshModel> mesh_model;
    std::unique_ptr<ReceiveMasks> receive_masks;
    std::unique_ptr<ReceiveDepth> receive_depth;
    if (paramss["robot"] == "icub")
    {
        camera = std::unique_ptr<Camera>(new iCubCamera(paramss["cam_sel"],
                                                        paramsd["resolution_ratio"],
                                                        rf.getContext(),
                                                        "handTracking/Process/iCubCamera/" + paramss["cam_sel"]));

        mesh_model = std::unique_ptr<MeshModel>(new iCubArmModel(paramsd["use_thumb"],
                                                                 paramsd["use_forearm"],
                                                                 paramss["laterality"],
                                                                 rf.getContext(),
                                                                 "handTracking/MeshModel/iCubArmModel/" + paramss["cam_sel"]));
    }
    else if (paramss["robot"] == "walkman")
    {
        camera = std::unique_ptr<Camera>(new WalkmanCamera(paramss["cam_sel"],
                                                           paramsd["resolution_ratio"],
                                                           rf.getContext(),
                                                           "handTracking/Process/WalkmanCamera/" + paramss["cam_sel"]));

        mesh_model = std::unique_ptr<MeshModel>(new WalkmanArmModel(paramss["laterality"],
                                                                    rf.getContext(),
                                                                    "handTracking/MeshModel/WalkmanArmModel/" + paramss["cam_sel"]));
    }
    else if (paramss["robot"] == "siamese")
    {
        // Added arguments to VisualProprioceptionSiamese
        receive_masks = std::unique_ptr<ReceiveMasks>(new ReceiveMasks());

        receive_depth = std::unique_ptr<ReceiveDepth>(new ReceiveDepth());
    }
    else
    {
        yError() << log_ID << "Wrong robot name. Provided: " << paramss["robot"] << ". Shall be either 'icub', 'walkman' or 'siamese'.";
        return EXIT_FAILURE;
    }

    /* SENSOR MODEL */
    std::unique_ptr<VisualProprioceptionSiamese> proprio;
    try
    {
        proprio = std::unique_ptr<VisualProprioceptionSiamese>(new VisualProprioceptionSiamese(std::move(receive_masks),
                                                                                               std::move(receive_depth),
                                                                                               paramsd["num_images"],
                                                                                               paramss["object_name"],
                                                                                               rf.getContext()));
                                                                                               // paramss["object_mesh_path"],
                                                                                               // paramss["shader_path"]));
   
        paramsd["num_particles"] = proprio->getNumberOfUsedParticles();

        yInfo() << log_ID << "General PF parameters changed after constructing VisualProprioception:";
        yInfo() << log_ID << " - num_particles:" << paramsd["num_particles"];
    }
    catch (const std::runtime_error& e)
    {
        yError() << e.what();
        return EXIT_FAILURE;
    }

    /* LIKELIHOOD */
    std::unique_ptr<LikelihoodModel> likelihood;
    if (paramss["likelihood_type"] == "chi")
        likelihood = std::unique_ptr<ChiSquare>(new ChiSquare(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "iou")
        likelihood = std::unique_ptr<IoU>(new IoU(paramsd["likelihood_gain"]));
    else if (paramss["likelihood_type"] == "iou_depth")
        likelihood = std::unique_ptr<IoU_depth>(new IoU_depth(paramsd["likelihood_gain"]));
    else if (paramss["likelihood_type"] == "kld")
        likelihood = std::unique_ptr<KLD>(new KLD(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_one")
        likelihood = std::unique_ptr<NormOne>(new NormOne(paramsd["likelihood_gain"]));
    else if (paramss["likelihood_type"] == "norm_two")
        likelihood = std::unique_ptr<NormTwo>(new NormTwo(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_two_chi")
        likelihood = std::unique_ptr<NormTwoChiSquare>(new NormTwoChiSquare(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_two_kld")
        likelihood = std::unique_ptr<NormTwoKLD>(new NormTwoKLD(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_two_kld_chi")
        likelihood = std::unique_ptr<NormTwoKLDChiSquare>(new NormTwoKLDChiSquare(paramsd["likelihood_gain"], 36));
    else
    {
        yError() << log_ID << "Wrong likelihood type. Provided: " << paramss["likelihood_type"] << ". Shalle be either 'norm_one' or 'norm_two_chi'.";
        return EXIT_FAILURE;
    }

    /* CORRECTION */
    std::unique_ptr<PFCorrection> vpf_correction(new BootstrapCorrection());
    vpf_correction->setLikelihoodModel(std::move(likelihood));
    vpf_correction->setMeasurementModel(std::move(proprio));

    /* RESAMPLING */
    std::unique_ptr<Resampling> pf_resampling;
    if (paramsd["resample_prior"] != 1.0)
        pf_resampling = std::unique_ptr<Resampling>(new Resampling());
    else
    {
        std::unique_ptr<ParticleSetInitialization> resample_init;

        if (paramss["robot"] == "icub")
            resample_init = std::unique_ptr<InitiCubArm>(new InitiCubArm(paramss["laterality"], "handTracking/ResamplingWithPrior/InitiCubArm/" + paramss["cam_sel"]));
        else if (paramss["robot"] == "walkman")
            resample_init = std::unique_ptr<InitWalkmanArm>(new InitWalkmanArm(paramss["laterality"], "handTracking/ResamplingWithPrior/InitWalkmanArm/" + paramss["cam_sel"]));
        else if (paramss["robot"] == "siamese")
            resample_init = std::unique_ptr<InitPoseParticlesSiamese>(new InitPoseParticlesSiamese("resampling-with-prior"));

        pf_resampling = std::unique_ptr<Resampling>(new ResamplingWithPrior(std::move(resample_init), paramsd["prior_ratio"]));
    }

    /* PARTICLE FILTER */
    VisualSIS vsis_pf(std::move(init_arm),
                      std::move(pf_prediction),
                      std::move(vpf_correction),
                      std::move(pf_resampling),
                      paramss["cam_sel"],
                      paramsd["num_particles"],
                      paramsd["resample_ratio"],
                      rf.getContext());

    yInfo() << log_ID << "Booting filter...";

    vsis_pf.boot();
    vsis_pf.wait();

    yInfo() << log_ID << "Application closed succesfully.";
    return EXIT_SUCCESS;
}
