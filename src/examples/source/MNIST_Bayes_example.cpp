/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/io/ModelFile.h>
#include <SmartPeak/io/Parameters.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief Fully Connected Bayesian model with Xavier-like initialization

  Reference:
  Blundell 2015 Weight uncertainty in neural networks arXiv:1505.05424

  @param[in, out] model The network model
  @param[in] n_inputs The number of input pixels
  @param[in] n_outputs The number of output labels
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeFullyConnBayes(Model<TensorT>& model, const int& n_inputs = 784, const int& n_outputs = 10, const int& n_hidden_0 = 512, const int& n_hidden_1 = 512, const int& n_hidden_2 = 512, const bool& add_gaussian = false, const bool& add_mixed_gaussian = false, const bool& specify_layers = false, const TensorT& learning_rate = 1e-3, const TensorT& gradient_clipping = 100) {
    model.setId(0);
    model.setName("FullyConnectedBayesClassifier");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_feature_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    auto activation_linear = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
    auto activation_linear_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(learning_rate, 0.9, 0.999, 1e-8, gradient_clipping));
    auto solver_dummy_op = std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>());

    // Define the nodes
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_encoding, node_names_input, node_names_mu_out, node_names_logvar_out;

    // Add the 1st FC layer
    if (n_hidden_0 > 0) {
      node_names_input = node_names;
      if (add_gaussian || add_mixed_gaussian) {
        // Add the gaussian nodes
        node_names_mu = model_builder.addFullyConnected(model, "EN0MuEnc0", "EN0MuEnc0", node_names_input, n_hidden_0,
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        node_names_logvar = model_builder.addFullyConnected(model, "EN0LogVarEnc0", "EN0LogVarEnc0", node_names_input, n_hidden_0,
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        node_names_encoding = model_builder.addGaussianEncoding(model, "EN0EncodingEnc0", "EN0EncodingEnc0", node_names_mu, node_names_logvar, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "EN0", "EN0", node_names_encoding, node_names_encoding.size(),
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0, 0.0, true, specify_layers);
        // Add the actual output nodes
        node_names_mu_out = model_builder.addSinglyConnected(model, "EN0Mu0", "EN0Mu0", node_names_mu, node_names_mu.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_logvar_out = model_builder.addSinglyConnected(model, "EN0LogVar0", "EN0LogVar0", node_names_logvar, node_names_logvar.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        // Specify the output node types manually
        for (const std::string& node_name : node_names_mu_out)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_logvar_out)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names)
        if (add_mixed_gaussian) {
          // Add the mixed gaussian nodes
          node_names_logvar = model_builder.addFullyConnected(model, "EN0LogVarEnc1", "EN0LogVarEnc1", node_names_input, n_hidden_0,
            activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
            solver_op, 0.0f, 0.0f, false, specify_layers);
          node_names_encoding = model_builder.addGaussianEncoding(model, "EN0EncodingEnc1", "EN0EncodingEnc1", node_names_mu, node_names_logvar, specify_layers);
          model_builder.addSinglyConnected(model, "EN0", node_names_encoding, node_names,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1e-2)), // Sigma 2 << 1 and Sigma 1 > Sigma 2
            solver_dummy_op, 0.0, specify_layers);
          // Add the actual output nodes
          node_names_logvar_out = model_builder.addSinglyConnected(model, "EN0LogVar1", "EN0LogVar1", node_names_logvar, node_names_logvar.size(),
            activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
            solver_dummy_op, 0.0f, 0.0f, false, true);
          // Specify the output node types manually
          for (const std::string& node_name : node_names_logvar_out)
            model.nodes_.at(node_name)->setType(NodeType::output);
          for (const std::string& node_name : node_names)
            model.nodes_.at(node_name)->setType(NodeType::output);
        }
      }
      else {
        node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
      }
    }

    // Add the 2nd FC layer
    if (n_hidden_1 > 0) {
      node_names_input = node_names;
      if (add_gaussian || add_mixed_gaussian) {
        // Add the gaussian nodes
        node_names_mu = model_builder.addFullyConnected(model, "EN1MuEnc0", "EN1MuEnc0", node_names_input, n_hidden_0,
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        node_names_logvar = model_builder.addFullyConnected(model, "EN1LogVarEnc0", "EN1LogVarEnc0", node_names_input, n_hidden_0,
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        node_names_encoding = model_builder.addGaussianEncoding(model, "EN1EncodingEnc0", "EN1EncodingEnc0", node_names_mu, node_names_logvar, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "EN1", "EN1", node_names_encoding, node_names_encoding.size(),
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0, 0.0, true, specify_layers);
        // Add the actual output nodes
        node_names_mu_out = model_builder.addSinglyConnected(model, "EN1Mu0", "EN1Mu0", node_names_mu, node_names_mu.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_logvar_out = model_builder.addSinglyConnected(model, "EN1LogVar0", "EN1LogVar0", node_names_logvar, node_names_logvar.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        // Specify the output node types manually
        for (const std::string& node_name : node_names_mu_out)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_logvar_out)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names)
        if (add_mixed_gaussian) {
          // Add the mixed gaussian nodes
          node_names_logvar = model_builder.addFullyConnected(model, "EN1LogVarEnc1", "EN1LogVarEnc1", node_names_input, n_hidden_0,
            activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
            solver_op, 0.0f, 0.0f, false, specify_layers);
          node_names_encoding = model_builder.addGaussianEncoding(model, "EN1EncodingEnc1", "EN1EncodingEnc1", node_names_mu, node_names_logvar, specify_layers);
          model_builder.addSinglyConnected(model, "EN1", node_names_encoding, node_names,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1e-2)), // Sigma 2 << 1 and Sigma 1 > Sigma 2
            solver_dummy_op, 0.0, specify_layers);
          // Add the actual output nodes
          node_names_logvar_out = model_builder.addSinglyConnected(model, "EN1LogVar1", "EN1LogVar1", node_names_logvar, node_names_logvar.size(),
            activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
            solver_dummy_op, 0.0f, 0.0f, false, true);
          // Specify the output node types manually
          for (const std::string& node_name : node_names_logvar_out)
            model.nodes_.at(node_name)->setType(NodeType::output);
          for (const std::string& node_name : node_names)
            model.nodes_.at(node_name)->setType(NodeType::output);
        }
      }
      else {
        node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names_input, n_hidden_1,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_1) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
      }
    }

    // Add the 3nd FC layer
    if (n_hidden_2 > 0) {
      node_names_input = node_names;
      if (add_gaussian || add_mixed_gaussian) {
        // Add the gaussian nodes
        node_names_mu = model_builder.addFullyConnected(model, "EN2MuEnc0", "EN2MuEnc0", node_names_input, n_hidden_0,
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        node_names_logvar = model_builder.addFullyConnected(model, "EN2LogVarEnc0", "EN2LogVarEnc0", node_names_input, n_hidden_0,
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        node_names_encoding = model_builder.addGaussianEncoding(model, "EN2EncodingEnc0", "EN2EncodingEnc0", node_names_mu, node_names_logvar, specify_layers);
        node_names = model_builder.addSinglyConnected(model, "EN2", "EN2", node_names_encoding, node_names_encoding.size(),
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0, 0.0, true, specify_layers);
        // Add the actual output nodes
        node_names_mu_out = model_builder.addSinglyConnected(model, "EN1Mu0", "EN1Mu0", node_names_mu, node_names_mu.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_logvar_out = model_builder.addSinglyConnected(model, "EN1LogVar0", "EN1LogVar0", node_names_logvar, node_names_logvar.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        // Specify the output node types manually
        for (const std::string& node_name : node_names_mu_out)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_logvar_out)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names)
        if (add_mixed_gaussian) {
          // Add the mixed gaussian nodes
          node_names_logvar = model_builder.addFullyConnected(model, "EN2LogVarEnc1", "EN2LogVarEnc1", node_names_input, n_hidden_0,
            activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
            solver_op, 0.0f, 0.0f, false, specify_layers);
          node_names_encoding = model_builder.addGaussianEncoding(model, "EN2EncodingEnc1", "EN2EncodingEnc1", node_names_mu, node_names_logvar, specify_layers);
          model_builder.addSinglyConnected(model, "EN2", node_names_encoding, node_names,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1e-2)), // Sigma 2 << 1 and Sigma 1 > Sigma 2
            solver_dummy_op, 0.0, specify_layers);
          // Add the actual output nodes
          node_names_logvar_out = model_builder.addSinglyConnected(model, "EN2LogVar1", "EN2LogVar1", node_names_logvar, node_names_logvar.size(),
            activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
            solver_dummy_op, 0.0f, 0.0f, false, true);
          // Specify the output node types manually
          for (const std::string& node_name : node_names_logvar_out)
            model.nodes_.at(node_name)->setType(NodeType::output);
          for (const std::string& node_name : node_names)
            model.nodes_.at(node_name)->setType(NodeType::output);
        }
      }
      else {
        node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names_input, n_hidden_2,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_2) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
      }
    }

    // Add the actual output nodes
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      solver_dummy_op, 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  void adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, const std::vector<float>& model_errors) override {
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;

      // Save to binary
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
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
    model_logger.setLogNodeOutputsEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, false);
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
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  bool add_gaussian_ = false;
  bool add_mixed_gaussian_ = false;
  int n_hidden_0_ = 0;
  int n_hidden_1_ = 0;
  int n_hidden_2_ = 0;
  void simulateData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& is_train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices(this->training_data.dimension(1));
    if (is_train)
      sample_indices = this->getTrainingIndices(batch_size, 1);
    else
      sample_indices = this->getValidationIndices(batch_size, 1);

    // pull out the training data and labels
    Eigen::Tensor<TensorT, 3> training_data(batch_size, memory_size, this->training_data.dimension(1));
    Eigen::Tensor<TensorT, 3> training_labels(batch_size, memory_size, this->training_labels.dimension(1));
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < this->training_data.dimension(1); ++nodes_iter) {
          if (is_train) {
            training_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
            training_labels(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
          }
          else {
            training_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
            training_labels(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
          }
        }
      }
    }

    // Assign the input data
    input_data.setZero();
    input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) })) = training_data;

    // Assign the input data
    loss_output_data.setZero();
    loss_output_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_labels.dimension(1) })) = training_labels;

    // Assign the input data
    metric_output_data.setZero();
    metric_output_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_labels.dimension(1) })) = training_labels;
    metric_output_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_labels.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_labels.dimension(1) })) = training_labels;

    assert(memory_size == 1);
    if (add_gaussian_) {
      if (n_hidden_0_ > 0 && n_hidden_1_ > 0 && n_hidden_2_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_ + 2 * n_hidden_2_);
        assert(n_metric_output_nodes == 2 * this->training_labels.dimension(1) + n_hidden_0_ + n_hidden_1_ + n_hidden_2_);
        assert(n_input_nodes == this->training_data.dimension(1) + n_hidden_0_ + n_hidden_1_ + n_hidden_2_);

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, n_hidden_0_ + n_hidden_1_ + n_hidden_2_)
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ + n_hidden_1_ + n_hidden_2_ }));

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + n_hidden_0_ + n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_2_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_2_ }));
      }
      if (n_hidden_0_ > 0 && n_hidden_1_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_);
        assert(n_metric_output_nodes == 2 * this->training_labels.dimension(1) + n_hidden_0_ + n_hidden_1_);
        assert(n_input_nodes == this->training_data.dimension(1) + n_hidden_0_ + n_hidden_1_);

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, n_hidden_0_ + n_hidden_1_)
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ + n_hidden_1_ }));

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ }));
      }
      if (n_hidden_0_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 2 * n_hidden_0_);
        assert(n_metric_output_nodes == 2 * this->training_labels.dimension(1) + n_hidden_0_);
        assert(n_input_nodes == this->training_data.dimension(1) + n_hidden_0_);

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, n_hidden_0_)
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
      }
    }
    else if (add_mixed_gaussian_) {
      if (n_hidden_0_ > 0 && n_hidden_1_ > 0 && n_hidden_2_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 3 * n_hidden_0_ + 3 * n_hidden_1_ + 3 * n_hidden_2_);
        assert(n_metric_output_nodes == 2 * this->training_labels.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_ + 2 * n_hidden_2_);
        assert(n_input_nodes == this->training_data.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_ + 2 * n_hidden_2_);

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, 2 * n_hidden_0_ + 2 * n_hidden_1_ + 2 * n_hidden_2_)
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, 2 * n_hidden_0_ + 2 * n_hidden_1_ + 2 * n_hidden_2_ }));

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + 2 * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 2 * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + 2 * n_hidden_0_ + n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 2 * n_hidden_0_ + n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_2_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 2 * n_hidden_0_ + 2 * n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_2_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_ + n_hidden_2_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_2_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 2 * n_hidden_0_ + 2 * n_hidden_1_ + n_hidden_2_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_2_ }));
      }
      if (n_hidden_0_ > 0 && n_hidden_1_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 3 * n_hidden_0_ + 3 * n_hidden_1_);
        assert(n_metric_output_nodes == 2 * this->training_labels.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_);
        assert(n_input_nodes == this->training_data.dimension(1) + 2 * n_hidden_0_ + 2 * n_hidden_1_);

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, 2 * n_hidden_0_ + 2 * n_hidden_1_)
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, 2 * n_hidden_0_ + 2 * n_hidden_1_ }));

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + 2 * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 2 * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + 2 * n_hidden_0_ + n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 2 * n_hidden_0_ + n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ }));
      }
      if (n_hidden_0_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 3 * n_hidden_0_);
        assert(n_metric_output_nodes == 2 * this->training_labels.dimension(1) + 2 * n_hidden_0_);
        assert(n_input_nodes == this->training_data.dimension(1) + 2 * n_hidden_0_);

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, 2 * n_hidden_0_)
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, 2 * n_hidden_0_ }));

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ }));
      }
    }
    else {
      assert(n_output_nodes == this->training_labels.dimension(1));
      assert(n_metric_output_nodes == 2 * this->training_labels.dimension(1));
      assert(n_input_nodes == this->training_data.dimension(1));
    }
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    simulateData(input_data, loss_output_data, metric_output_data, time_steps, true);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    simulateData(input_data, loss_output_data, metric_output_data, time_steps, false);
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerDefaultDevice<TensorT>
{};

/**
 @brief Image classification MNIST example whereby all pixels are
  linearized and read into the model.  The model then attempts to
  classify the image using a CovNet architecture

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
  - classifier (1 hot vector from 0 to 9)
 */
template<class ...ParameterTypes>
void main_MNIST(const ParameterTypes& ...args) {
  auto parameters = std::make_tuple(args...);

  // define the population trainer parameters
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(std::get<EvoNetParameters::PopulationTrainer::NGenerations>(parameters).get());
  population_trainer.setPopulationSize(std::get<EvoNetParameters::PopulationTrainer::PopulationSize>(parameters).get());
  population_trainer.setNReplicatesPerModel(std::get<EvoNetParameters::PopulationTrainer::NReplicatesPerModel>(parameters).get());
  population_trainer.setNTop(std::get<EvoNetParameters::PopulationTrainer::NTop>(parameters).get());
  population_trainer.setNRandom(std::get<EvoNetParameters::PopulationTrainer::NRandom>(parameters).get());
  population_trainer.setLogging(std::get<EvoNetParameters::PopulationTrainer::Logging>(parameters).get());
  population_trainer.setRemoveIsolatedNodes(std::get<EvoNetParameters::PopulationTrainer::RemoveIsolatedNodes>(parameters).get());
  population_trainer.setPruneModelNum(std::get<EvoNetParameters::PopulationTrainer::PruneModelNum>(parameters).get());
  population_trainer.setCheckCompleteModelInputToOutput(std::get<EvoNetParameters::PopulationTrainer::CheckCompleteModelInputToOutput>(parameters).get());
  population_trainer.setResetModelCopyWeights(std::get<EvoNetParameters::PopulationTrainer::ResetModelCopyWeights>(parameters).get());
  population_trainer.setResetModelTemplateWeights(std::get<EvoNetParameters::PopulationTrainer::ResetModelTemplateWeights>(parameters).get());
  //population_trainer.set_population_size_fixed_ = std::get<EvoNetParameters::PopulationTrainer::SetPopulationSizeFixed>(parameters).get();
  //population_trainer.set_population_size_doubling_ = std::get<EvoNetParameters::PopulationTrainer::SetPopulationSizeDoubling>(parameters).get();

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the multithreading parameters
  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = (std::get<EvoNetParameters::PopulationTrainer::NInterpreters>(parameters).get() > n_hard_threads) ? n_hard_threads : std::get<EvoNetParameters::PopulationTrainer::NInterpreters>(parameters).get(); // the number of threads

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // define the data simulator
  const std::size_t input_size = 784;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;
  data_simulator.n_hidden_0_ = std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get();
  data_simulator.n_hidden_1_ = std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get();
  data_simulator.n_hidden_2_ = std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get();
  data_simulator.add_gaussian_ = std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get();
  data_simulator.add_mixed_gaussian_ = std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get();

  // read in the training data
  std::string training_data_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "train-images.idx3-ubyte";
  std::string training_labels_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "train-labels.idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // read in the validation data
  std::string validation_data_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "t10k-images.idx3-ubyte";
  std::string validation_labels_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "t10k-labels.idx1-ubyte";
  data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);
  data_simulator.unitScaleData();

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the encoding nodes and add them to the input
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get() || std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "En0EncodingEnc0_%012d-Sampler", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "En1EncodingEnc0_%012d-Sampler", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "En1EncodingEnc0_%012d-Sampler", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
    if (std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
        char name_char[512];
        sprintf(name_char, "En0EncodingEnc1_%012d-Sampler", i);
        std::string name(name_char);
        input_nodes.push_back(name);
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++i) {
        char name_char[512];
        sprintf(name_char, "En1EncodingEnc1_%012d-Sampler", i);
        std::string name(name_char);
        input_nodes.push_back(name);
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
        char name_char[512];
        sprintf(name_char, "En1EncodingEnc1_%012d-Sampler", i);
        std::string name(name_char);
        input_nodes.push_back(name);
      }
    }
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < data_simulator.mnist_labels.size(); ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // Make the mu nodes and logvar nodes
  std::vector<std::string> encoding_nodes_en0mu0, encoding_nodes_en1mu0, encoding_nodes_en2mu0;
  std::vector<std::string> encoding_nodes_en0logvar0, encoding_nodes_en0logvar1, encoding_nodes_en1logvar0, encoding_nodes_en1logvar1, encoding_nodes_en2logvar0, encoding_nodes_en2logvar1;
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get() || std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
      char* name_char = new char[512];
      sprintf(name_char, "EN0Mu0_%012d", i);
      std::string name(name_char);
      encoding_nodes_en0mu0.push_back(name);
      name_char = new char[512];
      sprintf(name_char, "EN0LogVar0_%012d", i);
      name = name_char;
      encoding_nodes_en0logvar0.push_back(name);
      delete[] name_char;
    }
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++i) {
      char* name_char = new char[512];
      sprintf(name_char, "EN1Mu0_%012d", i);
      std::string name(name_char);
      encoding_nodes_en1mu0.push_back(name);
      name_char = new char[512];
      sprintf(name_char, "EN1LogVar0_%012d", i);
      name = name_char;
      encoding_nodes_en1logvar0.push_back(name);
      delete[] name_char;
    }
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(); ++i) {
      char* name_char = new char[512];
      sprintf(name_char, "EN2Mu0_%012d", i);
      std::string name(name_char);
      encoding_nodes_en2mu0.push_back(name);
      name_char = new char[512];
      sprintf(name_char, "EN2LogVar0_%012d", i);
      name = name_char;
      encoding_nodes_en2logvar0.push_back(name);
      delete[] name_char;
    }
    if (std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
        char* name_char = new char[512];
        sprintf(name_char, "EN0LogVar1_%012d", i);
        std::string name(name_char);
        encoding_nodes_en0logvar1.push_back(name);
        delete[] name_char;
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++i) {
        char* name_char = new char[512];
        sprintf(name_char, "EN1LogVar1_%012d", i);
        std::string name(name_char);
        encoding_nodes_en1logvar1.push_back(name);
        delete[] name_char;
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(); ++i) {
        char* name_char = new char[512];
        sprintf(name_char, "EN2LogVar1_%012d", i);
        std::string name(name_char);
        encoding_nodes_en2logvar1.push_back(name);
        delete[] name_char;
      }
    }
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(std::get<EvoNetParameters::Main::DeviceId>(parameters).get(), 1) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(std::get<EvoNetParameters::ModelTrainer::BatchSize>(parameters).get());
  model_trainer.setMemorySize(std::get<EvoNetParameters::ModelTrainer::MemorySize>(parameters).get());
  model_trainer.setNEpochsTraining(std::get<EvoNetParameters::ModelTrainer::NEpochsTraining>(parameters).get());
  model_trainer.setNEpochsValidation(std::get<EvoNetParameters::ModelTrainer::NEpochsValidation>(parameters).get());
  model_trainer.setNEpochsEvaluation(std::get<EvoNetParameters::ModelTrainer::NEpochsEvaluation>(parameters).get());
  model_trainer.setNTBPTTSteps(std::get<EvoNetParameters::ModelTrainer::NTBTTSteps>(parameters).get());
  model_trainer.setNTETTSteps(std::get<EvoNetParameters::ModelTrainer::NTETTSteps>(parameters).get());
  model_trainer.setVerbosityLevel(std::get<EvoNetParameters::ModelTrainer::Verbosity>(parameters).get());
  model_trainer.setLogging(std::get<EvoNetParameters::ModelTrainer::LoggingTraining>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::LoggingValidation>(parameters).get(),
    std::get<EvoNetParameters::ModelTrainer::LoggingEvaluation>(parameters).get());
  model_trainer.setFindCycles(std::get<EvoNetParameters::ModelTrainer::FindCycles>(parameters).get()); //true
  model_trainer.setFastInterpreter(std::get<EvoNetParameters::ModelTrainer::FastInterpreter>(parameters).get()); //false
  model_trainer.setPreserveOoO(std::get<EvoNetParameters::ModelTrainer::PreserveOoO>(parameters).get());
  model_trainer.setInterpretModel(std::get<EvoNetParameters::ModelTrainer::InterpretModel>(parameters).get());
  model_trainer.setResetModel(std::get<EvoNetParameters::ModelTrainer::ResetModel>(parameters).get());
  model_trainer.setResetInterpreter(std::get<EvoNetParameters::ModelTrainer::ResetInterpreter>(parameters).get());

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-24, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-24, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper1);
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get() || std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
    if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0) {
      loss_function_helper2.output_nodes_ = encoding_nodes_en0mu0;
      loss_function_helper2.loss_functions_ = { std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helper2.loss_function_grads_ = { std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helpers.push_back(loss_function_helper2);
      loss_function_helper3.output_nodes_ = encoding_nodes_en0logvar0;
      loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helpers.push_back(loss_function_helper3);
    }
    if (std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
      loss_function_helper2.output_nodes_ = encoding_nodes_en1mu0;
      loss_function_helper2.loss_functions_ = { std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helper2.loss_function_grads_ = { std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helpers.push_back(loss_function_helper2);
      loss_function_helper3.output_nodes_ = encoding_nodes_en1logvar0;
      loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helpers.push_back(loss_function_helper3);
    }
    if (std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get() > 0) {
      loss_function_helper2.output_nodes_ = encoding_nodes_en2mu0;
      loss_function_helper2.loss_functions_ = { std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helper2.loss_function_grads_ = { std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helpers.push_back(loss_function_helper2);
      loss_function_helper3.output_nodes_ = encoding_nodes_en2logvar0;
      loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
      loss_function_helpers.push_back(loss_function_helper3);
    }
    if (std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
      if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0) {
        loss_function_helper3.output_nodes_ = encoding_nodes_en0logvar1;
        loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
        loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
        loss_function_helpers.push_back(loss_function_helper3);
      }
      if (std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
        loss_function_helper3.output_nodes_ = encoding_nodes_en1logvar1;
        loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
        loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
        loss_function_helpers.push_back(loss_function_helper3);
      }
      if (std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get() > 0) {
        loss_function_helper3.output_nodes_ = encoding_nodes_en2logvar1;
        loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
        loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize(), 0.0)) };
        loss_function_helpers.push_back(loss_function_helper3);
      }
    }
  }
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1, metric_function_helper2;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<AccuracyMCMicroOp<float>>(AccuracyMCMicroOp<float>()), std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>()) };
  metric_function_helper1.metric_names_ = { "AccuracyMCMicro", "PrecisionMCMicro" };
  metric_function_helpers.push_back(metric_function_helper1);
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get() || std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
    if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0) {
      metric_function_helper1.output_nodes_ = encoding_nodes_en0logvar0;
      metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
      metric_function_helper1.metric_names_ = { "MAE" };
      metric_function_helpers.push_back(metric_function_helper1);
    }
    if (std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
      metric_function_helper1.output_nodes_ = encoding_nodes_en1logvar0;
      metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
      metric_function_helper1.metric_names_ = { "MAE" };
      metric_function_helpers.push_back(metric_function_helper1);
    }
    if (std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get() > 0) {
      metric_function_helper1.output_nodes_ = encoding_nodes_en2logvar0;
      metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
      metric_function_helper1.metric_names_ = { "MAE" };
      metric_function_helpers.push_back(metric_function_helper1);
    }
    if (std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get()) {
      if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0) {
        metric_function_helper1.output_nodes_ = encoding_nodes_en0logvar1;
        metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
        metric_function_helper1.metric_names_ = { "MAE" };
        metric_function_helpers.push_back(metric_function_helper1);
      }
      if (std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
        metric_function_helper1.output_nodes_ = encoding_nodes_en1logvar1;
        metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
        metric_function_helper1.metric_names_ = { "MAE" };
        metric_function_helpers.push_back(metric_function_helper1);
      }
      if (std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get() > 0) {
        metric_function_helper1.output_nodes_ = encoding_nodes_en2logvar1;
        metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
        metric_function_helper1.metric_names_ = { "MAE" };
        metric_function_helpers.push_back(metric_function_helper1);
      }
    }
  }
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
    std::make_pair(std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>())),
    std::make_pair(std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>())),
    std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
    std::make_pair(std::make_shared<TanHOp<float>>(TanHOp<float>()), std::make_shared<TanHGradOp<float>>(TanHGradOp<float>()))//,
    });
  model_replicator.setNodeIntegrations({ std::make_tuple(std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>())),
    std::make_tuple(std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>())),
    });
  //model_replicator.set_modification_rate_by_prev_error_ = std::get<EvoNetParameters::ModelReplicator::SetModificationRateByPrevError>(parameters).get();
  //model_replicator.set_modification_rate_fixed_ = std::get<EvoNetParameters::ModelReplicator::SetModificationRateFixed>(parameters).get();
  model_replicator.setRandomModifications(
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeDownAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeDownAdditionsUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeRightAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeRightAdditionsUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeDownCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeDownCopiesUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeRightCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeRightCopiesUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NLinkAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NLinkAdditionsUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NLinkCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NLinkCopiesUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeDeletionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeDeletionsUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NLinkDeletionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NLinkDeletionsUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeActivationChangesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeActivationChangesUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeIntegrationChangesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeIntegrationChangesUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NModuleAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NModuleAdditionsUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NModuleCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NModuleCopiesUB>(parameters).get()),
    std::make_pair(std::get<EvoNetParameters::ModelReplicator::NModuleDeletionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NModuleDeletionsUB>(parameters).get()));

  // define the initial population
  Model<float> model;
  if (std::get<EvoNetParameters::Main::MakeModel>(parameters).get()) {
    std::cout << "Making the model..." << std::endl;
    model_trainer.makeFullyConnBayes(model, input_nodes.size(), output_nodes.size(), std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::AddMixedGaussian>(parameters).get(),
      true, std::get<EvoNetParameters::ModelTrainer::LearningRate>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::GradientClipping>(parameters).get());  // Baseline
    model.setId(0);
  }
  else if (std::get<EvoNetParameters::Main::LoadModelBinary>(parameters).get()) {
    // read in the trained model
    std::cout << "Reading in the model from binary..." << std::endl;
    ModelFile<float> model_file;
    model_file.loadModelBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model);
    model.setId(1);
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_interpreter.binary", model_interpreters[0]); // FIX ME!
  }
  else if (std::get<EvoNetParameters::Main::LoadModelCsv>(parameters).get()) {
    // read in the trained model
    std::cout << "Reading in the model from csv..." << std::endl;
    ModelFile<float> model_file;
    model_file.loadModelCsv(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_nodes.csv", std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_links.csv", std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_weights.csv", model, true, true, true);
    model.setId(1);
  }
  model.setName(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory

  if (std::get<EvoNetParameters::Main::TrainModel>(parameters).get()) {
    // Train the model
    model.setName(model.getName() + "_train");
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());
  }
  else if (std::get<EvoNetParameters::Main::EvolveModel>(parameters).get()) {
    // Evolve the population
    std::vector<Model<float>> population = { model };
    std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
      population, std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get(), //So that all output will be written to a specific directory
      model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    PopulationTrainerFile<float> population_trainer_file;
    population_trainer_file.storeModels(population, std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get());
    population_trainer_file.storeModelValidations(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get() + "Errors.csv", models_validation_errors_per_generation);
  }
  else if (std::get<EvoNetParameters::Main::EvaluateModel>(parameters).get()) {
    //// Evaluate the population
    //std::vector<Model<float>> population = { model };
    //population_trainer.evaluateModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
    // Evaluate the model
    model.setName(model.getName() + "_evaluation");
    Eigen::Tensor<float, 4> model_output = model_trainer.evaluateModel(model, data_simulator, input_nodes, model_logger, model_interpreters.front());
  }

}

/// MNIST_CovNet_example 0 C:/Users/dmccloskey/Documents/GitHub/mnist/Parameters.csv
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
  EvoNetParameters::Examples::ModelType model_type("model_type", "Solution");
  EvoNetParameters::Examples::SimulationType simulation_type("simulation_type", "");
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
  EvoNetParameters::ModelTrainer::BatchSize batch_size("batch_size", 32);
  EvoNetParameters::ModelTrainer::MemorySize memory_size("memory_size", 1);
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
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
  EvoNetParameters::ModelTrainer::ResetModel reset_model("reset_model", false);
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 128);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 128);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 0);
  EvoNetParameters::ModelTrainer::AddGaussian add_gaussian("add_gaussian", true);
  EvoNetParameters::ModelTrainer::AddMixedGaussian add_mixed_gaussian("add_mixed_gaussian", false);
  EvoNetParameters::ModelTrainer::LearningRate learning_rate("learning_rate", 1e-3);
  EvoNetParameters::ModelTrainer::GradientClipping gradient_clipping("gradient_clipping", 10);
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
    device_id, model_name, make_model, load_model_csv, load_model_binary, train_model, evolve_model, evaluate_model,
    model_type, simulation_type,
    population_name, n_generations, n_interpreters, prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling,
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, reset_interpreter,
    n_hidden_0, n_hidden_1, n_hidden_2, add_gaussian, add_mixed_gaussian, learning_rate, gradient_clipping,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error);

  // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = SmartPeak::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  SmartPeak::apply([](auto&& ...args) { main_MNIST(args ...); }, parameters);
  return 0;
}