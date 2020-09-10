/**TODO:  Add copyright*/

#include <EvoNet/ml/PopulationTrainerGpu.h>
#include <EvoNet/ml/ModelTrainerGpu.h>
#include <EvoNet/ml/ModelReplicator.h>
#include <EvoNet/ml/ModelBuilder.h>
#include <EvoNet/io/PopulationTrainerFile.h>
#include <EvoNet/io/ModelInterpreterFileGpu.h>
#include <EvoNet/simulator/MetabolomicsReconstructionDataSimulator.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace EvoNet;

// Other extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief Variational autoencoder that encodes the labels using a concrete distribution
    and style using a gaussian distribution

  References:
    arXiv:1804.00104
    https://github.com/Schlumberger/joint-vae

  @param[in, out] model The network model
  @param[in] n_pixels The number of input/output pixels
  @param[in] n_categorical The length of the categorical layer
  @param[in] n_encodings The length of the encodings layer
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeVAEFullyConn(Model<TensorT>& model,
    const int& n_inputs = 784, const int& n_encodings = 64, const int& n_categorical = 10, const int& n_hidden_0 = 512, const int& n_hidden_1 = 256, const int& n_hidden_2 = 64,
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
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_logalpha;
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
    node_names_logalpha = model_builder.addFullyConnected(model, "LogAlphaEnc", "LogAlphaEnc", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Add the Encoding layers
    std::vector<std::string> node_names_Gencoder = model_builder.addGaussianEncoding(model, "Gaussian_encoding", "Gaussian_encoding", node_names_mu, node_names_logvar, true);
    std::vector<std::string> node_names_Cencoder = model_builder.addCategoricalEncoding(model, "Categorical_encoding", "Categorical_encoding", node_names_logalpha, true);

    // Add the Decoder FC layers
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names_Gencoder, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE2", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    if (n_hidden_1 > 0 && n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    else if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names_Gencoder, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE1", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    if (n_hidden_0 > 0 && n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    else if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names_Gencoder, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE1", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, specify_layers);
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
    node_names_logalpha = model_builder.addSinglyConnected(model, "LogAlpha", "LogAlpha", node_names_logalpha, node_names_logalpha.size(),
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
    for (const std::string& node_name : node_names_logalpha)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_Cencoder)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) override {
    // Check point the model every 1000 epochs
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false, false);
      // save the model weights (Not needed if binary is working fine)
      //WeightFile<float> weight_data;
      //weight_data.storeWeightValuesCsv(model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model.weights_);

      // save the model and tensors to binary
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);

      // Increase the KL divergence beta and capacity
      TensorT beta = 1;
      TensorT capacity_c = 0;
      TensorT capacity_d = 0;
      if (this->KL_divergence_warmup_) {
        TensorT scale_factor1 = (n_epochs - 100 > 0) ? n_epochs - 100 : 1;
        beta = 30 / 2.5e4 * scale_factor1;
        if (beta > 30) beta = 30;
        TensorT scale_factor2 = (n_epochs - 1.0e4 > 0) ? n_epochs - 1.0e4 : 1;
        capacity_c = 5 / 1.5e4 * scale_factor2;
        if (capacity_c > 5) capacity_c = 5;
        capacity_d = 5 / 1.5e4 * scale_factor2;
        if (capacity_d > 5) capacity_d = 5;
      }
      this->getLossFunctionHelpers().at(1).loss_functions_.at(0) = std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(2).loss_functions_.at(0) = std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(3).loss_functions_.at(0) = std::make_shared<KLDivergenceCatLossOp<float>>(KLDivergenceCatLossOp<float>(1e-6, beta, capacity_d));
      this->getLossFunctionHelpers().at(1).loss_function_grads_.at(0) = std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(2).loss_function_grads_.at(0) = std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(3).loss_function_grads_.at(0) = std::make_shared<KLDivergenceCatLossGradOp<float>>(KLDivergenceCatLossGradOp<float>(1e-6, beta, capacity_d));
    }
  }
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
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
    for (const std::string& metric_name : this->getMetricNamesLinearized()) {
      log_train_headers.push_back(metric_name);
      log_test_headers.push_back(metric_name);
      log_train_values.push_back(model_metrics_train(metric_iter));
      log_test_values.push_back(model_metrics_test(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
  bool KL_divergence_warmup_ = true;
};

template<class ...ParameterTypes>
void main_(const ParameterTypes& ...args) {
  auto parameters = std::make_tuple(args...);

  // define the model logger
  ModelLogger<float> model_logger(true, true, true, false, false, false, false);

  // define the data simulator

  // prior to using shuffle when making the data caches
  const int n_labels = 7; // IndustrialStrains0103
  const int n_reps_per_sample = std::get<EvoNetParameters::ModelTrainer::BatchSize>(parameters).get()
    * std::get<EvoNetParameters::ModelTrainer::NEpochsTraining>(parameters).get() / n_labels;

  // define the data simulator
  std::cout << "Making the training and validation data..." << std::endl;
  MetabolomicsReconstructionDataSimulator<float> data_simulator;
  data_simulator.n_encodings_continuous_ = std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get();
  data_simulator.n_encodings_discrete_ = std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get();
  int n_reaction_ids_training, n_labels_training, n_component_group_names_training;
  int n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation;
  data_simulator.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    std::get<EvoNetParameters::Examples::BiochemicalRxnsFilename>(parameters).get(),
    std::get<EvoNetParameters::Examples::MetaboDataTrainFilename>(parameters).get(),
    std::get<EvoNetParameters::Examples::MetaDataTrainFilename>(parameters).get(),
    std::get<EvoNetParameters::Examples::MetaboDataTestFilename>(parameters).get(),
    std::get<EvoNetParameters::Examples::MetaDataTestFilename>(parameters).get(),
    std::get<EvoNetParameters::Examples::UseConcentrations>(parameters).get(),
    std::get<EvoNetParameters::Examples::UseMARs>(parameters).get(),
    std::get<EvoNetParameters::Examples::SampleValues>(parameters).get(),
    std::get<EvoNetParameters::Examples::IterValues>(parameters).get(),
    std::get<EvoNetParameters::Examples::FillSampling>(parameters).get(),
    std::get<EvoNetParameters::Examples::FillMean>(parameters).get(),
    std::get<EvoNetParameters::Examples::FillZero>(parameters).get(),
    std::get<EvoNetParameters::Examples::ApplyFoldChange>(parameters).get(),
    std::get<EvoNetParameters::Examples::FoldChangeRef>(parameters).get(),
    std::get<EvoNetParameters::Examples::FoldChangeLogBase>(parameters).get(),
    std::get<EvoNetParameters::Examples::OfflineLinearScaleInput>(parameters).get(),
    std::get<EvoNetParameters::Examples::OfflineLogTransformInput>(parameters).get(),
    std::get<EvoNetParameters::Examples::OfflineStandardizeInput>(parameters).get(),
    std::get<EvoNetParameters::Examples::OnlineLinearScaleInput>(parameters).get(),
    std::get<EvoNetParameters::Examples::OnlineLogTransformInput>(parameters).get(),
    std::get<EvoNetParameters::Examples::OnlineStandardizeInput>(parameters).get(),
    n_reps_per_sample, true, false,
    std::get<EvoNetParameters::ModelTrainer::NEpochsTraining>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::BatchSize>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::MemorySize>(parameters).get());

  // Update the number of discrete encodings if necessary
  if (n_labels_training != std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get()) {
    std::cout << "Over-riding the number of labels " << n_labels_training << " does not match the number of discrete encodings " << std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get() << std::endl;
    std::cout << "Ensure that classification losses and metric weights are set to 0. " << std::endl;
  }

  // define the model input/output nodes
  int n_input_nodes;
  if (std::get<EvoNetParameters::Examples::UseMARs>(parameters).get()) n_input_nodes = n_reaction_ids_training;
  else n_input_nodes = n_component_group_names_training;
  const int n_output_nodes = n_input_nodes;

  //// Balance the sample group names
  //data_simulator.model_training_.sample_group_names_ = {
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
  ////data_simulator.model_training_.sample_group_names_ = {
  ////"S01_D01_PLT_25C_22hr","S01_D01_PLT_25C_6.5hr","S01_D01_PLT_25C_0hr","S01_D02_PLT_25C_22hr","S01_D02_PLT_25C_6.5hr","S01_D02_PLT_25C_0hr","S01_D05_PLT_25C_0hr","S01_D05_PLT_25C_22hr","S01_D05_PLT_25C_6.5hr","S01_D01_PLT_37C_22hr","S01_D02_PLT_37C_22hr","S01_D05_PLT_37C_22hr"
  ////};
  ////data_simulator.model_validation_.sample_group_names_ = {
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
  for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
    char name_char[512];
    sprintf(name_char, "Gaussian_encoding_%012d-Sampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding_%012d-GumbelSampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding_%012d-InverseTau", i);
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
  for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
    char name_char[512];
    sprintf(name_char, "Mu_%012d", i);
    std::string name(name_char);
    encoding_nodes_mu.push_back(name);
  }

  // Make the encoding nodes
  std::vector<std::string> encoding_nodes_logvar;
  for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
    char name_char[512];
    sprintf(name_char, "LogVar_%012d", i);
    std::string name(name_char);
    encoding_nodes_logvar.push_back(name);
  }

  // Make the alpha nodes
  std::vector<std::string> encoding_nodes_logalpha;
  for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
    char name_char[512];
    sprintf(name_char, "LogAlpha_%012d", i);
    std::string name(name_char);
    encoding_nodes_logalpha.push_back(name);
  }

  // Softmax nodes
  std::vector<std::string> categorical_softmax_nodes;
  for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding-SoftMax-Out_%012d", i);
    std::string name(name_char);
    categorical_softmax_nodes.push_back(name);
  }

  // define the model interpreters
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  setModelInterpreterParameters(model_interpreters, args...);

  // define the model trainer
  ModelTrainerExt<float> model_trainer;
  setModelTrainerParameters(model_trainer, args...);
  model_trainer.setNEpochsTraining(std::get<EvoNetParameters::ModelTrainer::NEpochsTraining>(parameters).get() * 5); // iterate through the cache 5x

  std::shared_ptr<LossFunctionOp<float>> loss_function_op;
  std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad_op;
  if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MSE")) {
    loss_function_op = std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MAE")) {
    loss_function_op = std::make_shared<MAELossOp<float>>(MAELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MAELossGradOp<float>>(MAELossGradOp<float>(1e-6, 1.0));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MLE")) {
    loss_function_op = std::make_shared<MLELossOp<float>>(MLELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MLELossGradOp<float>>(MLELossGradOp<float>(1e-6, 1.0));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MAPE")) {
    loss_function_op = std::make_shared<MAPELossOp<float>>(MAPELossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<MAPELossGradOp<float>>(MAPELossGradOp<float>(1e-6, 1.0));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("BCEWithLogits")) {
    loss_function_op = std::make_shared<BCEWithLogitsLossOp<float>>(BCEWithLogitsLossOp<float>(1e-6, 1.0));
    loss_function_grad_op = std::make_shared<BCEWithLogitsLossGradOp<float>>(BCEWithLogitsLossGradOp<float>(1e-6, 1.0));
  }

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3, loss_function_helper4, loss_function_helper5;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { loss_function_op };
  loss_function_helper1.loss_function_grads_ = { loss_function_grad_op };
  loss_function_helpers.push_back(loss_function_helper1);
  loss_function_helper2.output_nodes_ = encoding_nodes_mu;
  loss_function_helper2.loss_functions_ = { std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper2.loss_function_grads_ = { std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper2);
  loss_function_helper3.output_nodes_ = encoding_nodes_logvar;
  loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper3);
  loss_function_helper4.output_nodes_ = encoding_nodes_logalpha;
  loss_function_helper4.loss_functions_ = { std::make_shared<KLDivergenceCatLossOp<float>>(KLDivergenceCatLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper4.loss_function_grads_ = { std::make_shared<KLDivergenceCatLossGradOp<float>>(KLDivergenceCatLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper4);
  if (std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get() > 0) {
    loss_function_helper5.output_nodes_ = categorical_softmax_nodes;
    loss_function_helper5.loss_functions_ = { std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-8, std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get())) };
    loss_function_helper5.loss_function_grads_ = { std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-8, std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get())) };
    loss_function_helpers.push_back(loss_function_helper5);
  }
  model_trainer.setLossFunctionHelpers(loss_function_helpers);
  model_trainer.KL_divergence_warmup_ = std::get<EvoNetParameters::ModelTrainer::KLDivergenceWarmup>(parameters).get();

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1, metric_function_helper2;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<CosineSimilarityOp<float>>(CosineSimilarityOp<float>("Mean")), std::make_shared<CosineSimilarityOp<float>>(CosineSimilarityOp<float>("Var")),
    std::make_shared<PearsonROp<float>>(PearsonROp<float>("Mean")), std::make_shared<PearsonROp<float>>(PearsonROp<float>("Var")),
    std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Mean")), std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Var")),
    std::make_shared<ManhattanDistOp<float>>(ManhattanDistOp<float>("Mean")), std::make_shared<ManhattanDistOp<float>>(ManhattanDistOp<float>("Var")),
    std::make_shared<JeffreysAndMatusitaDistOp<float>>(JeffreysAndMatusitaDistOp<float>("Mean")), std::make_shared<JeffreysAndMatusitaDistOp<float>>(JeffreysAndMatusitaDistOp<float>("Var")),
    std::make_shared<LogarithmicDistOp<float>>(LogarithmicDistOp<float>("Mean")), std::make_shared<LogarithmicDistOp<float>>(LogarithmicDistOp<float>("Var")),
    std::make_shared<PercentDifferenceOp<float>>(PercentDifferenceOp<float>("Mean")), std::make_shared<PercentDifferenceOp<float>>(PercentDifferenceOp<float>("Var")) };
  metric_function_helper1.metric_names_ = { "CosineSimilarity-Mean", "CosineSimilarity-Var",
    "PearsonR-Mean", "PearsonR-Var",
    "EuclideanDist-Mean", "EuclideanDist-Var",
    "ManhattanDist-Mean", "ManhattanDist-Var",
    "JeffreysAndMatusitaDist-Mean", "JeffreysAndMatusitaDist-Var",
    "LogarithmicDist-Mean", "LogarithmicDist-Var",
    "PercentDifference-Mean", "PercentDifference-Var" };
  metric_function_helpers.push_back(metric_function_helper1);
  if (std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get() > 0) {
    metric_function_helper2.output_nodes_ = categorical_softmax_nodes;
    metric_function_helper2.metric_functions_ = { std::make_shared<AccuracyMCMicroOp<float>>(AccuracyMCMicroOp<float>()), std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>()) };
    metric_function_helper2.metric_names_ = { "AccuracyMCMicro", "PrecisionMCMicro" };
    metric_function_helpers.push_back(metric_function_helper2);
  }
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  Model<float> model;
  if (std::get<EvoNetParameters::Main::MakeModel>(parameters).get()) {
    std::cout << "Making the model..." << std::endl;
    model_trainer.makeVAEFullyConn(model, n_input_nodes, 
      std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
  }
  else {
    ModelFile<float> model_file;
    ModelInterpreterFileGpu<float> model_interpreter_file;
    loadModelFromParameters(model, model_interpreters.at(0), model_file, model_interpreter_file, args...);
  }
  model.setName(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
    input_nodes, model_logger, model_interpreters.front());
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
  EvoNetParameters::Examples::ModelType model_type("model_type", "Solution");
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
  EvoNetParameters::ModelTrainer::LossFncWeight0 loss_fnc_weight_0("loss_fnc_weight_0", 1);
  EvoNetParameters::ModelTrainer::LossFncWeight1 loss_fnc_weight_1("loss_fnc_weight_1", 0);
  EvoNetParameters::ModelTrainer::LossFncWeight2 loss_fnc_weight_2("loss_fnc_weight_2", 0);
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
  EvoNetParameters::ModelTrainer::LossFunction loss_function("loss_function", "MSE");
  EvoNetParameters::ModelTrainer::KLDivergenceWarmup KL_divergence_warmup("KL_divergence_warmup", true);
  EvoNetParameters::ModelTrainer::NEncodingsContinuous n_encodings_continuous("n_encodings_continuous", 8);
  EvoNetParameters::ModelTrainer::NEncodingsCategorical n_encodings_categorical("n_encodings_categorical", 8);
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
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, n_hidden_0, n_hidden_1, n_hidden_2, reset_interpreter, loss_function, KL_divergence_warmup, n_encodings_continuous, n_encodings_categorical/*,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error*/);
    
  // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = EvoNet::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  EvoNet::apply([](auto&& ...args) { main_(args ...); }, parameters);
  return 0;
}