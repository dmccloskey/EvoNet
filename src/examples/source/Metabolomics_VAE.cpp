/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/simulator/MetabolomicsReconstructionDataSimulator.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Other extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief Basic VAE with	Xavier-like initialization

  References:
  Based on Kingma et al, 2014: https://arxiv.org/pdf/1312.6114
  https://github.com/pytorch/examples/blob/master/vae/main.py

  @param[in, out] model The network model
  @param[in] n_inputs The number of input pixels
  @param[in] n_encodings The length of the encodings layer
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeVAEFullyConn(Model<TensorT>& model,
    const int& n_inputs = 784, const int& n_encodings = 64, const int& n_hidden_0 = 512, const int& n_hidden_1 = 256, const int& n_hidden_2 = 64,
    const bool& add_bias = true, const bool& specify_layers = false) {
    model.setId(0);
    model.setName("VAE");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-5, 0.9, 0.999, 1e-8, 10));

    // Add the Endocer FC layers
    std::vector<std::string> node_names_mu, node_names_logvar;
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    node_names_mu = model_builder.addFullyConnected(model, "MuEnc", "MuEnc", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names_logvar = model_builder.addFullyConnected(model, "LogVarEnc", "LogVarEnc", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Add the Encoding layers
    node_names = model_builder.addGaussianEncoding(model, "Encoding", "Encoding", node_names_mu, node_names_logvar, specify_layers);

    // Add the Decoder FC layers
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    node_names = model_builder.addFullyConnected(model, "DE-Output", "DE-Output", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, add_bias, true);

    // Add the actual output nodes
    node_names_mu = model_builder.addSinglyConnected(model, "Mu", "Mu", node_names_mu, node_names_mu.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    node_names_logvar = model_builder.addSinglyConnected(model, "LogVar", "LogVar", node_names_logvar, node_names_logvar.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_mu)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_logvar)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) override {
    // Check point the model every 1000 epochs
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false, false);
      //// save the model weights
      //WeightFile<float> weight_data;
      //weight_data.storeWeightValuesCsv(model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model.weights_);
      // save the model and tensors to binary
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);

      // Increase the KL divergence beta and capacity
      TensorT beta = 1 / 2.5e4 * n_epochs;
      if (beta > 1) beta = 1;
      TensorT capacity_z = 0.0 / 2.5e4 * n_epochs;
      if (capacity_z > 5) capacity_z = 5;
      this->getLossFunctions().at(1) = std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, beta, capacity_z));
      this->getLossFunctions().at(2) = std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, beta, capacity_z));
      this->getLossFunctionGrads().at(1) = std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, beta, capacity_z));
      this->getLossFunctionGrads().at(2) = std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, beta, capacity_z));
    }
  }
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test,
    const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) { // FIXME
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_train_headers = { "Train_Error" };
    std::vector<std::string> log_test_headers = { "Test_Error" };
    std::vector<TensorT> log_train_values = { model_error_train };
    std::vector<TensorT> log_test_values = { model_error_test };
    int metric_iter = 0;
    for (const std::string& metric_name : this->metric_names_) {
      log_train_headers.push_back(metric_name);
      log_test_headers.push_back(metric_name);
      log_train_values.push_back(model_metrics_train(metric_iter));
      log_test_values.push_back(model_metrics_test(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
};

/// Script to run the reconstruction network
void main_reconstruction(const std::string& data_dir, const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
  const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
  const bool& make_model, const bool& train_model,
  const bool& use_concentrations, const bool& use_MARs,
  const bool& sample_values, const bool& iter_values,
  const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero,
  const bool& apply_fold_change, const std::string& fold_change_ref, const float& fold_change_log_base,
  const bool& offline_linear_scale_input, const bool& offline_log_transform_input, const bool& offline_standardize_input,
  const bool& online_linear_scale_input, const bool& online_log_transform_input, const bool& online_standardize_input,
  const std::string& loss_function,
  const int& device_id)
{
  // global local variables
  const int n_epochs = 20000;
  const int batch_size = 64;
  const int memory_size = 1;
  //const int n_reps_per_sample = 10000;
  const int n_encodings_continuous = 8;

  // prior to using shuffle when making the data caches
  const int n_labels = 7; // IndustrialStrains0103
  const int n_reps_per_sample = n_epochs*batch_size/n_labels;

  //std::string model_name = "MetClass_" + std::to_string(use_concentrations) + "-" + std::to_string(use_MARs) + "-" + std::to_string(sample_values) + "-" + std::to_string(iter_values) + "-"
  //  + std::to_string(fill_sampling) + "-" + std::to_string(fill_mean) + "-" + std::to_string(fill_zero) + "-" + std::to_string(apply_fold_change) + "-" + std::to_string(fold_change_log_base) + "-"
  //  + std::to_string(offline_linear_scale_input) + "-" + std::to_string(offline_log_transform_input) + "-" + std::to_string(offline_standardize_input) + "-"
  //  + std::to_string(online_linear_scale_input) + "-" + std::to_string(online_log_transform_input) + "-" + std::to_string(online_standardize_input);
  std::string model_name = "VAE";

  // define the data simulator
  std::cout << "Making the training and validation data..." << std::endl;
  MetabolomicsReconstructionDataSimulator<float> metabolomics_data;
  metabolomics_data.n_encodings_continuous_ = n_encodings_continuous;
  int n_reaction_ids_training, n_labels_training, n_component_group_names_training;
  int n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation;
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base,
    offline_linear_scale_input, offline_log_transform_input, offline_standardize_input, online_linear_scale_input, online_log_transform_input, online_standardize_input,
    n_reps_per_sample, true, false, n_epochs, batch_size, memory_size);

  // define the model input/output nodes
  int n_input_nodes;
  if (use_MARs) n_input_nodes = n_reaction_ids_training;
  else n_input_nodes = n_component_group_names_training;
  const int n_output_nodes = n_labels_training;

  //// Balance the sample group names
  //metabolomics_data.model_training_.sample_group_names_ = {
  //"Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04",
  //"Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP",
  //"Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd",
  //"Evo04gndEvo01EP", "Evo04gndEvo01EP", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04gndEvo02EP", "Evo04gndEvo02EP",
  //"Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB",
  //"Evo04sdhCBEvo01EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04sdhCBEvo02EP", "Evo04sdhCBEvo02EP",
  //"Evo04pgi", "Evo04pgi", "Evo04pgi", "Evo04pgi", "Evo04pgi", "Evo04pgi",
  //"Evo04pgiEvo01EP", "Evo04pgiEvo02EP", "Evo04pgiEvo03EP", "Evo04pgiEvo04EP", "Evo04pgiEvo05EP", "Evo04pgiEvo06EP",
  //"Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr",
  //"Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04ptsHIcrrEvo03EP", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04ptsHIcrrEvo03EP",
  //"Evo04tpiA", "Evo04tpiA", "Evo04tpiA", "Evo04tpiA", "Evo04tpiA", "Evo04tpiA",
  //"Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP", "Evo04tpiAEvo03EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP", "Evo04tpiAEvo03EP"
  //};
  ////metabolomics_data.model_training_.sample_group_names_ = {
  ////"S01_D01_PLT_25C_22hr","S01_D01_PLT_25C_6.5hr","S01_D01_PLT_25C_0hr","S01_D02_PLT_25C_22hr","S01_D02_PLT_25C_6.5hr","S01_D02_PLT_25C_0hr","S01_D05_PLT_25C_0hr","S01_D05_PLT_25C_22hr","S01_D05_PLT_25C_6.5hr","S01_D01_PLT_37C_22hr","S01_D02_PLT_37C_22hr","S01_D05_PLT_37C_22hr"
  ////};
  ////metabolomics_data.model_validation_.sample_group_names_ = {
  ////"S02_D01_PLT_25C_22hr","S02_D01_PLT_25C_6.5hr","S02_D01_PLT_25C_0hr","S02_D02_PLT_25C_22hr","S02_D02_PLT_25C_6.5hr","S02_D02_PLT_25C_0hr","S02_D05_PLT_25C_0hr","S02_D05_PLT_25C_22hr","S02_D05_PLT_25C_6.5hr","S02_D01_PLT_37C_22hr","S02_D02_PLT_37C_22hr","S02_D05_PLT_37C_22hr"
  ////};

  // Make the input nodes
  std::vector<std::string> input_nodes;
  std::vector<std::string> met_input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
    met_input_nodes.push_back(name);
  }

  // Make the encoding nodes and add them to the input
  for (int i = 0; i < n_encodings_continuous; ++i) {
    char name_char[512];
    sprintf(name_char, "Encoding_%012d-Sampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the reconstruction nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_output_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // Make the mu nodes
  std::vector<std::string> encoding_nodes_mu;
  for (int i = 0; i < n_encodings_continuous; ++i) {
    char name_char[512];
    sprintf(name_char, "Mu_%012d", i);
    std::string name(name_char);
    encoding_nodes_mu.push_back(name);
  }

  // Make the encoding nodes
  std::vector<std::string> encoding_nodes_logvar;
  for (int i = 0; i < n_encodings_continuous; ++i) {
    char name_char[512];
    sprintf(name_char, "LogVar_%012d", i);
    std::string name(name_char);
    encoding_nodes_logvar.push_back(name);
  }

  // define the model trainers and resources for the trainers
  ModelResources model_resources = { ModelDevice(device_id, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(batch_size);
  model_trainer.setMemorySize(memory_size);
  model_trainer.setNEpochsTraining(n_epochs * 5); // Iterate through the stored data 5 times
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  std::shared_ptr<LossFunctionOp<float>> loss_function_op;
  std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad_op;
  if (loss_function == std::string("MSE")) {
    loss_function_op = std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0));
  }
  else if (loss_function == std::string("MAE")) {
    loss_function_op = std::make_shared<MAELossOp<float>>(MAELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MAELossGradOp<float>>(MAELossGradOp<float>(1e-6, 1.0));
  }
  else if (loss_function == std::string("MLE")) {
    loss_function_op = std::make_shared<MLELossOp<float>>(MLELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MLELossGradOp<float>>(MLELossGradOp<float>(1e-6, 1.0));
  }
  else if (loss_function == std::string("MAPE")) {
    loss_function_op = std::make_shared<MAPELossOp<float>>(MAPELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MAPELossGradOp<float>>(MAPELossGradOp<float>(1e-6, 1.0));
  }
  else if (loss_function == std::string("BCEWithLogits")) {
    loss_function_op = std::make_shared<BCEWithLogitsLossOp<float>>(BCEWithLogitsLossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<BCEWithLogitsLossGradOp<float>>(BCEWithLogitsLossGradOp<float>(1e-6, 1.0));
  }
  model_trainer.setLossFunctions({
    loss_function_op,
    std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, 0.0, 0.0)),
    std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 0.0, 0.0))
    });
  model_trainer.setLossFunctionGrads({
    loss_function_grad_op,
    std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, 0.0, 0.0)),
    std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 0.0, 0.0))
    });
  model_trainer.setLossOutputNodes({ output_nodes, encoding_nodes_mu, encoding_nodes_logvar });
  model_trainer.setMetricFunctions({ 
    std::make_shared<CosineSimilarityOp<float>>(CosineSimilarityOp<float>("Mean")), std::make_shared<CosineSimilarityOp<float>>(CosineSimilarityOp<float>("Var")), 
    std::make_shared<PearsonROp<float>>(PearsonROp<float>("Mean")), std::make_shared<PearsonROp<float>>(PearsonROp<float>("Var")), 
    std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Mean")), std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Var")), 
    std::make_shared<ManhattanDistOp<float>>(ManhattanDistOp<float>("Mean")), std::make_shared<ManhattanDistOp<float>>(ManhattanDistOp<float>("Var")), 
    std::make_shared<JeffreysAndMatusitaDistOp<float>>(JeffreysAndMatusitaDistOp<float>("Mean")), std::make_shared<JeffreysAndMatusitaDistOp<float>>(JeffreysAndMatusitaDistOp<float>("Var")), 
    std::make_shared<LogarithmicDistOp<float>>(LogarithmicDistOp<float>("Mean")), std::make_shared<LogarithmicDistOp<float>>(LogarithmicDistOp<float>("Var")),
    std::make_shared<PercentDifferenceOp<float>>(PercentDifferenceOp<float>("Mean")), std::make_shared<PercentDifferenceOp<float>>(PercentDifferenceOp<float>("Var"))});
  model_trainer.setMetricOutputNodes({ 
    output_nodes, output_nodes,
    output_nodes, output_nodes,
    output_nodes, output_nodes,
    output_nodes, output_nodes,
    output_nodes, output_nodes,
    output_nodes, output_nodes,
    output_nodes, output_nodes });
  model_trainer.setMetricNames({ 
    "CosineSimilarity-Mean", "CosineSimilarity-Var",
    "PearsonR-Mean", "PearsonR-Var",
    "EuclideanDist-Mean", "EuclideanDist-Var",
    "ManhattanDist-Mean", "ManhattanDist-Var",
    "JeffreysAndMatusitaDist-Mean", "JeffreysAndMatusitaDist-Var",
    "LogarithmicDist-Mean", "LogarithmicDist-Var",
    "PercentDifference-Mean", "PercentDifference-Var" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  Model<float> model;
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    model_trainer.makeVAEFullyConn(model, n_input_nodes, n_encodings_continuous, 8, 0, 0, false, true);
  }
  else {
    // TODO
  }
  model.setName(data_dir + model_name); //So that all output will be written to a specific directory

  // Train the model
  std::cout << "Training the model..." << std::endl;
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, metabolomics_data,
    input_nodes, model_logger, model_interpreter);
}

void main_loadBinaryModelAndStoreWeightsCsv(const std::string& model_filename) {
  // load the binarized model
  Model<float> model;
  ModelFile<float> model_file;
  model_file.loadModelBinary(model_filename, model);

  // save the model weights
  WeightFile<float> data;
  data.storeWeightValuesCsv(model.getName() + "_weights.csv", model.weights_);
}

// Main
int main(int argc, char** argv)
{

  // Initialize the defaults
  std::string data_dir = "";
  std::string biochem_rxns_filename = data_dir + "iJO1366.csv";
  std::string metabo_data_filename_train = data_dir + "ALEsKOs01_Metabolomics_train.csv"; // IndustrialStrains0103_
  std::string meta_data_filename_train = data_dir + "ALEsKOs01_MetaData_train.csv";
  std::string metabo_data_filename_test = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  std::string meta_data_filename_test = data_dir + "ALEsKOs01_MetaData_test.csv";
  bool make_model = true;
  bool train_model = true;
  bool use_concentrations = true;
  bool use_MARs = false;
  bool sample_values = true;
  bool iter_values = false;
  bool fill_sampling = false;
  bool fill_mean = false;
  bool fill_zero = false;
  bool apply_fold_change = false;
  std::string fold_change_ref = "Evo04";
  float fold_change_log_base = 10;
  bool offline_linear_scale_input = true;
  bool offline_log_transform_input = false;
  bool offline_standardize_input = false;
  bool online_linear_scale_input = false;
  bool online_log_transform_input = false;
  bool online_standardize_input = false;
  int device_id = 1;
  std::string loss_function = "MSE";

  // Parse the input
  std::cout << "Parsing the user input..." << std::endl;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    biochem_rxns_filename = argv[2];
  }
  if (argc >= 4) {
    metabo_data_filename_train = argv[3];
  }
  if (argc >= 5) {
    meta_data_filename_train = argv[4];
  }
  if (argc >= 6) {
    metabo_data_filename_test = argv[5];
  }
  if (argc >= 7) {
    meta_data_filename_test = argv[6];
  }
  if (argc >= 8) {
    make_model = (argv[7] == std::string("true")) ? true : false;
  }
  if (argc >= 9) {
    train_model = (argv[8] == std::string("true")) ? true : false;
  }
  if (argc >= 10) {
    use_concentrations = (argv[9] == std::string("true")) ? true : false;
  }
  if (argc >= 11) {
    use_MARs = (argv[10] == std::string("true")) ? true : false;
  }
  if (argc >= 12) {
    sample_values = (argv[11] == std::string("true")) ? true : false;
  }
  if (argc >= 13) {
    iter_values = (argv[12] == std::string("true")) ? true : false;
  }
  if (argc >= 14) {
    fill_sampling = (argv[13] == std::string("true")) ? true : false;
  }
  if (argc >= 15) {
    fill_mean = (argv[14] == std::string("true")) ? true : false;
  }
  if (argc >= 16) {
    fill_zero = (argv[15] == std::string("true")) ? true : false;
  }
  if (argc >= 17) {
    apply_fold_change = (argv[16] == std::string("true")) ? true : false;
  }
  if (argc >= 18) {
    fold_change_ref = argv[17];
  }
  if (argc >= 19) {
    try {
      fold_change_log_base = std::stof(argv[18]);
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }
  if (argc >= 20) {
    offline_linear_scale_input = (argv[19] == std::string("true")) ? true : false;
  }
  if (argc >= 21) {
    offline_log_transform_input = (argv[20] == std::string("true")) ? true : false;
  }
  if (argc >= 22) {
    offline_standardize_input = (argv[21] == std::string("true")) ? true : false;
  }
  if (argc >= 23) {
    online_linear_scale_input = (argv[22] == std::string("true")) ? true : false;
  }
  if (argc >= 24) {
    online_log_transform_input = (argv[23] == std::string("true")) ? true : false;
  }
  if (argc >= 25) {
    online_standardize_input = (argv[24] == std::string("true")) ? true : false;
  }
  if (argc >= 26) {
    if (argv[25] == std::string("MSE"))
      loss_function = "MSE";
    else if (argv[25] == std::string("MAE"))
      loss_function = "MAE";
    else if (argv[25] == std::string("MLE"))
      loss_function = "MLE";
    else if (argv[25] == std::string("MAPE"))
      loss_function = "MAPE";
    else if (argv[25] == std::string("BCEWithLogits"))
      loss_function = "BCEWithLogits";
  }
  if (argc >= 27) {
    try {
      device_id = std::stoi(argv[26]);
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }

  // Cout the parsed input
  std::cout << "data_dir: " << data_dir << std::endl;
  std::cout << "biochem_rxns_filename: " << biochem_rxns_filename << std::endl;
  std::cout << "metabo_data_filename_train: " << metabo_data_filename_train << std::endl;
  std::cout << "meta_data_filename_train: " << meta_data_filename_train << std::endl;
  std::cout << "metabo_data_filename_test: " << metabo_data_filename_test << std::endl;
  std::cout << "meta_data_filename_test: " << meta_data_filename_test << std::endl;
  std::cout << "make_model: " << make_model << std::endl;
  std::cout << "train_model: " << train_model << std::endl;
  std::cout << "use_concentrations: " << use_concentrations << std::endl;
  std::cout << "use_MARs: " << use_MARs << std::endl;
  std::cout << "sample_values: " << sample_values << std::endl;
  std::cout << "iter_values: " << iter_values << std::endl;
  std::cout << "fill_sampling: " << fill_sampling << std::endl;
  std::cout << "fill_mean: " << fill_mean << std::endl;
  std::cout << "fill_zero: " << fill_zero << std::endl;
  std::cout << "apply_fold_change: " << apply_fold_change << std::endl;
  std::cout << "fold_change_ref: " << fold_change_ref << std::endl;
  std::cout << "fold_change_log_base: " << fold_change_log_base << std::endl;
  std::cout << "offline_linear_scale_input: " << offline_linear_scale_input << std::endl;
  std::cout << "offline_log_transform_input: " << offline_log_transform_input << std::endl;
  std::cout << "offline_standardize_input: " << offline_standardize_input << std::endl;
  std::cout << "online_linear_scale_input: " << online_linear_scale_input << std::endl;
  std::cout << "online_log_transform_input: " << online_log_transform_input << std::endl;
  std::cout << "online_standardize_input: " << online_standardize_input << std::endl;
  std::cout << "loss_function: " << loss_function << std::endl;
  std::cout << "device_id: " << device_id << std::endl;

  // Run the classification
  main_reconstruction(data_dir, biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    make_model, train_model,
    use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero,
    apply_fold_change, fold_change_ref, fold_change_log_base,
    offline_linear_scale_input, offline_log_transform_input, offline_standardize_input,
    online_linear_scale_input, online_log_transform_input, online_standardize_input, loss_function, device_id
  );
  return 0;
}