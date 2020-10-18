/**TODO:  Add copyright*/

#include <EvoNet/ml/ModelInterpreterGpu.h>
#include <EvoNet/io/ModelInterpreterFileGpu.h>
#include <EvoNet/simulator/MetabolomicsLatentUnsClassDataSimulator.h>
#include <EvoNet/models/CVAEFullyConnGpu.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Metabolomics_CVAE.h"

using namespace EvoNet;
using namespace EvoNetMetabolomics;

template<class ...ParameterTypes>
void main_(const ParameterTypes& ...args) {
  auto parameters = std::make_tuple(args...);

  // define the data simulator
  std::cout << "Making the training and validation data..." << std::endl;
  MetabolomicsLatentUnsClassDataSimulator<float> data_simulator;
  const int n_features = makeDataSimulator(data_simulator, args...);

  // Make the input nodes
  std::vector<std::string> input_nodes;
  makeInputNodes(input_nodes, n_features);

  // Make the output nodes
  std::vector<std::string> encoding_nodes_mu = makeMuEncodingNodes(args...);
  std::vector<std::string> encoding_nodes_logvar = makeLogVarEncodingNodes(args...);
  std::vector<std::string> encoding_nodes_logalpha = makeAlphaEncodingNodes(args...);
  std::vector<std::string> categorical_softmax_nodes = makeCategoricalSoftmaxNodes(args...);

  // define the model trainer
  CVAEFullyConnGpu<float> model_trainer;
  makeModelTrainer<float>(model_trainer, std::vector<std::string>(), std::vector<std::string>(), std::vector<std::string>(), std::vector<std::string>(), categorical_softmax_nodes, args...);

  // define the model and resources
  Model<float> model;
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  ModelInterpreterFileGpu<float> model_interpreter_file;
  makeModelAndInterpreters(model, model_trainer, model_interpreters, model_interpreter_file, n_features, args...);

  // define the model logger
  ModelLogger<float> model_logger(true, true, true, false, false, true, false, true);

  // Validate the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.validateModel(model, data_simulator,
    input_nodes, model_logger, model_interpreters.front());
}

// Main
int main(int argc, char** argv)
{
  // Parse the user commands
  int id_int = -1;
  std::string parameters_filename = "";
  parseCommandLineArguments(argc, argv, id_int, parameters_filename);

  // Set the parameter names and defaults
  EvoNetParameters::General::ID id("id", -1);
  EvoNetParameters::General::DataDir data_dir("data_dir", std::string(""));
  EvoNetParameters::General::OutputDir output_dir("output_dir", std::string(""));
  EvoNetParameters::Main::DeviceId device_id("device_id", 0);
  EvoNetParameters::Main::ModelName model_name("model_name", "");
  EvoNetParameters::Main::MakeModel make_model("make_model", true);
  EvoNetParameters::Main::LoadModelCsv load_model_csv("load_model_csv", false);
  EvoNetParameters::Main::LoadModelBinary load_model_binary("load_model_binary", false);
  EvoNetParameters::Main::TrainModel train_model("train_model", true);
  EvoNetParameters::Main::EvolveModel evolve_model("evolve_model", false);
  EvoNetParameters::Main::EvaluateModel evaluate_model("evaluate_model", false);
  EvoNetParameters::Main::EvaluateModels evaluate_models("evaluate_models", false);
  EvoNetParameters::Examples::ModelType model_type("model_type", "EncDec"); // Options include EncDec, Enc, Dec
  EvoNetParameters::Examples::SimulationType simulation_type("simulation_type", "");
  EvoNetParameters::Examples::BiochemicalRxnsFilename biochemical_rxns_filename("biochemical_rxns_filename", "");
  EvoNetParameters::Examples::MetaboDataTrainFilename metabo_data_train_filename("metabo_data_train_filename", "");
  EvoNetParameters::Examples::MetaboDataTestFilename metabo_data_test_filename("metabo_data_test_filename", "");
  EvoNetParameters::Examples::MetaDataTrainFilename meta_data_train_filename("meta_data_train_filename", "");
  EvoNetParameters::Examples::MetaDataTestFilename meta_data_test_filename("meta_data_test_filename", "");
  EvoNetParameters::Examples::UseConcentrations use_concentrations("use_concentrations", true);
  EvoNetParameters::Examples::UseMARs use_MARs("use_MARs", false);
  EvoNetParameters::Examples::SampleValues sample_values("sample_values", true);
  EvoNetParameters::Examples::IterValues iter_values("iter_values", false);
  EvoNetParameters::Examples::FillSampling fill_sampling("fill_sampling", false);
  EvoNetParameters::Examples::FillMean fill_mean("fill_mean", false);
  EvoNetParameters::Examples::FillZero fill_zero("fill_zero", false);
  EvoNetParameters::Examples::ApplyFoldChange apply_fold_change("apply_fold_change", false);
  EvoNetParameters::Examples::FoldChangeRef fold_change_ref("fold_change_ref", "Evo04");
  EvoNetParameters::Examples::FoldChangeLogBase fold_change_log_base("fold_change_log_base", 10);
  EvoNetParameters::Examples::OfflineLinearScaleInput offline_linear_scale_input("offline_linear_scale_input", true);
  EvoNetParameters::Examples::OfflineLogTransformInput offline_log_transform_input("offline_log_transform_input", false);
  EvoNetParameters::Examples::OfflineStandardizeInput offline_standardize_input("offline_standardize_input", false);
  EvoNetParameters::Examples::OnlineLinearScaleInput online_linear_scale_input("online_linear_scale_input", false);
  EvoNetParameters::Examples::OnlineLogTransformInput online_log_transform_input("online_log_transform_input", false);
  EvoNetParameters::Examples::OnlineStandardizeInput online_standardize_input("online_standardize_input", false);
  EvoNetParameters::PopulationTrainer::PopulationName population_name("population_name", "");
  EvoNetParameters::PopulationTrainer::NGenerations n_generations("n_generations", 1);
  EvoNetParameters::PopulationTrainer::NInterpreters n_interpreters("n_interpreters", 1);
  EvoNetParameters::PopulationTrainer::PruneModelNum prune_model_num("prune_model_num", 10);
  EvoNetParameters::PopulationTrainer::RemoveIsolatedNodes remove_isolated_nodes("remove_isolated_nodes", true);
  EvoNetParameters::PopulationTrainer::CheckCompleteModelInputToOutput check_complete_model_input_to_output("check_complete_model_input_to_output", true);
  EvoNetParameters::PopulationTrainer::PopulationSize population_size("population_size", 128);
  EvoNetParameters::PopulationTrainer::NTop n_top("n_top", 8);
  EvoNetParameters::PopulationTrainer::NRandom n_random("n_random", 8);
  EvoNetParameters::PopulationTrainer::NReplicatesPerModel n_replicates_per_model("n_replicates_per_model", 1);
  EvoNetParameters::PopulationTrainer::ResetModelCopyWeights reset_model_copy_weights("reset_model_copy_weights", true);
  EvoNetParameters::PopulationTrainer::ResetModelTemplateWeights reset_model_template_weights("reset_model_template_weights", true);
  EvoNetParameters::PopulationTrainer::Logging population_logging("population_logging", true);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeFixed set_population_size_fixed("set_population_size_fixed", false);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeDoubling set_population_size_doubling("set_population_size_doubling", true);
  EvoNetParameters::PopulationTrainer::SetTrainingStepsByModelSize set_training_steps_by_model_size("set_training_steps_by_model_size", false);
  EvoNetParameters::ModelTrainer::BatchSize batch_size("batch_size", 32);
  EvoNetParameters::ModelTrainer::MemorySize memory_size("memory_size", 64);
  EvoNetParameters::ModelTrainer::NEpochsTraining n_epochs_training("n_epochs_training", 1000);
  EvoNetParameters::ModelTrainer::NEpochsValidation n_epochs_validation("n_epochs_validation", 25);
  EvoNetParameters::ModelTrainer::NEpochsEvaluation n_epochs_evaluation("n_epochs_evaluation", 10);
  EvoNetParameters::ModelTrainer::NTBTTSteps n_tbtt_steps("n_tbtt_steps", 64);
  EvoNetParameters::ModelTrainer::NTETTSteps n_tett_steps("n_tett_steps", 64);
  EvoNetParameters::ModelTrainer::Verbosity verbosity("verbosity", 1);
  EvoNetParameters::ModelTrainer::LoggingTraining logging_training("logging_training", true);
  EvoNetParameters::ModelTrainer::LoggingValidation logging_validation("logging_validation", false);
  EvoNetParameters::ModelTrainer::LoggingEvaluation logging_evaluation("logging_evaluation", true);
  EvoNetParameters::ModelTrainer::FindCycles find_cycles("find_cycles", true);
  EvoNetParameters::ModelTrainer::FastInterpreter fast_interpreter("fast_interpreter", true);
  EvoNetParameters::ModelTrainer::PreserveOoO preserve_ooo("preserve_ooo", true);
  EvoNetParameters::ModelTrainer::InterpretModel interpret_model("interpret_model", true);
  EvoNetParameters::ModelTrainer::ResetModel reset_model("reset_model", false);
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 16);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 0);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 0);
  EvoNetParameters::ModelTrainer::LossFncWeight0 loss_fnc_weight_0("loss_fnc_weight_0", 1); // Classification loss
  EvoNetParameters::ModelTrainer::LossFncWeight1 loss_fnc_weight_1("loss_fnc_weight_1", 1); // Reconstruction loss
  EvoNetParameters::ModelTrainer::LossFncWeight2 loss_fnc_weight_2("loss_fnc_weight_2", 0);
  EvoNetParameters::ModelTrainer::LearningRate learning_rate("learning_rate", 1e-5);
  EvoNetParameters::ModelTrainer::GradientClipping gradient_clipping("gradient_clipping", 10);
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
  EvoNetParameters::ModelTrainer::LossFunction loss_function("loss_function", "MSE");
  EvoNetParameters::ModelTrainer::KLDivergenceWarmup KL_divergence_warmup("KL_divergence_warmup", true);
  EvoNetParameters::ModelTrainer::NEncodingsContinuous n_encodings_continuous("n_encodings_continuous", 8);
  EvoNetParameters::ModelTrainer::NEncodingsCategorical n_encodings_categorical("n_encodings_categorical", 8);
  EvoNetParameters::ModelTrainer::Beta beta("beta", 30);
  EvoNetParameters::ModelTrainer::CapacityC capacity_c("capacity_c", 5);
  EvoNetParameters::ModelTrainer::CapacityD capacity_d("capacity_d", 5);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsLB n_node_down_additions_lb("n_node_down_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsLB n_node_right_additions_lb("n_node_right_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesLB n_node_down_copies_lb("n_node_down_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesLB n_node_right_copies_lb("n_node_right_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsLB n_link_additons_lb("n_link_additons_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesLB n_link_copies_lb("n_link_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsLB n_node_deletions_lb("n_node_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsLB n_link_deletions_lb("n_link_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesLB n_node_activation_changes_lb("n_node_activation_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesLB n_node_integration_changes_lb("n_node_integration_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsLB n_module_additions_lb("n_module_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesLB n_module_copies_lb("n_module_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsLB n_module_deletions_lb("n_module_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsUB n_node_down_additions_ub("n_node_down_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsUB n_node_right_additions_ub("n_node_right_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesUB n_node_down_copies_ub("n_node_down_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesUB n_node_right_copies_ub("n_node_right_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsUB n_link_additons_ub("n_link_additons_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesUB n_link_copies_ub("n_link_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsUB n_node_deletions_ub("n_node_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsUB n_link_deletions_ub("n_link_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesUB n_node_activation_changes_ub("n_node_activation_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesUB n_node_integration_changes_ub("n_node_integration_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsUB n_module_additions_ub("n_module_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesUB n_module_copies_ub("n_module_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsUB n_module_deletions_ub("n_module_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::SetModificationRateFixed set_modification_rate_fixed("set_modification_rate_fixed", false);
  EvoNetParameters::ModelReplicator::SetModificationRateByPrevError set_modification_rate_by_prev_error("set_modification_rate_by_prev_error", false);
  auto parameters = std::make_tuple(id, data_dir, output_dir,
    device_id, model_name, make_model, load_model_csv, load_model_binary, train_model, evolve_model, evaluate_model, evaluate_models,
    model_type, simulation_type, biochemical_rxns_filename, metabo_data_train_filename, metabo_data_test_filename, meta_data_train_filename, meta_data_test_filename, use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base, offline_linear_scale_input, offline_log_transform_input, offline_standardize_input, online_linear_scale_input, online_log_transform_input, online_standardize_input,
    population_name, n_generations, n_interpreters, /*prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling, set_training_steps_by_model_size,*/
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, n_hidden_0, n_hidden_1, n_hidden_2, loss_fnc_weight_0, loss_fnc_weight_1, loss_fnc_weight_2, learning_rate, gradient_clipping, reset_interpreter, loss_function, KL_divergence_warmup, n_encodings_continuous, n_encodings_categorical, beta, capacity_c, capacity_d/*,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error*/);

    // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = EvoNet::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  EvoNet::apply([](auto&& ...args) { main_(args ...); }, parameters);
  return 0;
}