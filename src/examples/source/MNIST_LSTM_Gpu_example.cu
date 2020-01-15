/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>
#include <SmartPeak/io/ModelFile.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief LSTM classifier

  Pixel by Pixel MNIST.  Examples include the following:
    arXiv:1511.06464: 128 hidden units, alpha = 1e-3, gradient clipping of 1, highest test accuracy of 98.2%
    arXiv:1504.00941: 100 hidden units, alpha = 0.01, forget_gate_bias = 1, gradient clipping of 1, lowest test error rate of 3%
    arXiv:1801.06105: 100 hidden units, alpha = 1e-6, gradient clipping of 1

  @param[in, out] model The network model
  @param[in] n_inputs The number of pixel inputs
  @param[in] n_outputs The number of classifier outputs
  @param[in] n_blocks The number of LSTM blocks to add to the network
  @param[in] n_cells The number of cells in each LSTM block
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeLSTM(Model<TensorT>& model, const int& n_inputs = 784, const int& n_outputs = 10,
    const int& n_blocks_1 = 128, const int& n_cells_1 = 1, const int& n_blocks_2 = 0, const int& n_cells_2 = 1,
    const int& n_hidden = 32, const bool& add_forget_gate = true, const bool& add_feature_norm = true, const bool& specify_layers = true) {
    model.setId(0);
    model.setName("LSTM");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_feature_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation, activation_grad, activation_fc, activation_fc_grad;
    if (add_feature_norm) {
      activation = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
      activation_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());
      activation_fc = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
      activation_fc_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());
    }
    else {
      //activation = std::make_shared<TanHOp<TensorT>>(TanHOp<TensorT>());
      //activation_grad = std::make_shared<TanHGradOp<TensorT>>(TanHGradOp<TensorT>());
      activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
      activation_grad = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
      activation_fc = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
      activation_fc_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    }
    //std::shared_ptr<ActivationOp<TensorT>> activation_norm = std::make_shared<TanHOp<TensorT>>(TanHOp<TensorT>());
    //std::shared_ptr<ActivationOp<TensorT>> activation_norm_grad = std::make_shared<TanHGradOp<TensorT>>(TanHGradOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_norm = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_norm_grad = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_fc_norm = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_fc_norm_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 10));

    // Add the LSTM layer(s)
    std::vector<std::string> node_names = model_builder.addLSTM(model, "LSTM-01", "LSTM-01", node_names_input, n_blocks_1, n_cells_1,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_blocks_1) / 2, 1)),
      solver_op,
      0.0f, 0.0f, true, add_forget_gate, 1, specify_layers);
    if (add_feature_norm) {
      node_names = model_builder.addNormalization(model, "LSTM-01-Norm", "LSTM-01-Norm", node_names, true);
      node_names = model_builder.addSinglyConnected(model, "LSTM-01-Norm-gain", "LSTM-01-Norm-gain", node_names, node_names.size(),
        activation_norm, activation_norm_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        solver_op,
        0.0, 0.0, true, specify_layers);
    }

    if (n_blocks_2 > 0) {
      node_names = model_builder.addLSTM(model, "LSTM-02", "LSTM-02", node_names, n_blocks_2, n_cells_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_blocks_2) / 2, 1)),
        solver_op,
        0.0f, 0.0f, true, add_forget_gate, 1, specify_layers);
    }
    if (add_feature_norm) {
      node_names = model_builder.addNormalization(model, "LSTM-02-Norm", "LSTM-02-Norm", node_names, true);
      node_names = model_builder.addSinglyConnected(model, "LSTM-02-Norm-gain", "LSTM-02-Norm-gain", node_names, node_names.size(),
        activation_norm, activation_norm_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        solver_op,
        0.0, 0.0, true, specify_layers);
    }

    // Add a fully connected layer
    if (n_hidden > 0) {
      node_names = model_builder.addFullyConnected(model, "FC-01", "FC-01", node_names, n_hidden,
        activation_fc, activation_fc_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
    }
    if (add_feature_norm) {
      node_names = model_builder.addNormalization(model, "FC-01-Norm", "FC-01-Norm", node_names, true);
      node_names = model_builder.addSinglyConnected(model, "FC-01-Norm-gain", "FC-01-Norm-gain", node_names, node_names.size(),
        activation_fc_norm, activation_fc_norm_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        solver_op,
        0.0, 0.0, true, specify_layers);
    }

    // Add a final output layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, n_outputs,
      activation_output, activation_output_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief RNN classifier

  References
    arXiv:1504.00941: 100 hidden units, alpha = 10e-8, all weights initialized to 1 and all biases initialized to 0, gradient clipping of 1, lowest test error rate of 3%

  @param[in, out] model The network model
  @param[in] n_inputs The number of pixel inputs
  @param[in] n_outputs The number of classifier outputs
  @param[in] n_blocks The number of LSTM blocks to add to the network
  @param[in] n_cells The number of cells in each LSTM block
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeRNN(Model<TensorT>& model, const int& n_inputs = 784, const int& n_outputs = 10,
    const int& n_hidden_0 = 32, const int& n_hidden_1 = 32, const bool& add_identity = false, const bool& add_feature_norm = true, const bool& specify_layers = true) {
    model.setId(0);
    model.setName("RNN");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_feature_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation, activation_grad;
    if (add_feature_norm) {
      activation = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
      activation_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());
    }
    else {
      activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
      activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    }
    std::shared_ptr<ActivationOp<TensorT>> activation_norm = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_norm_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 10));

    // Add the 1st RNN layer
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      model_builder.addSinglyConnected(model, "EN0-Rec", node_names, node_names,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))),
        solver_op, 0.0f, specify_layers);
      if (add_identity) {
        std::vector<std::string> node_names_tmp = model_builder.addSinglyConnected(model, "EN0-Identity0", "EN0-Identity0", node_names, node_names.size(),
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))),
          std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, false, specify_layers);
        model_builder.addSinglyConnected(model, "EN0-Identity1", node_names_tmp, node_names,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))),
          std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, specify_layers);
      }
      if (add_feature_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
    }

    // Add the 2nd FC layer
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      model_builder.addSinglyConnected(model, "EN1-Rec", node_names, node_names,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))),
        solver_op, 0.0f, specify_layers);
      if (add_identity) {
        std::vector<std::string> node_names_tmp = model_builder.addSinglyConnected(model, "EN1-Identity0", "EN1-Identity0", node_names, node_names.size(),
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))),
          std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, false, specify_layers);
        model_builder.addSinglyConnected(model, "EN1-Identity1", node_names_tmp, node_names,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))),
          std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, specify_layers);
      }
      if (add_feature_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
    }

    // Add a final output layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, n_outputs,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) override {
    //if (n_epochs % 100 == 0 && n_epochs > 100) {
    //  // anneal the learning rate by half on each plateau
    //  TensorT lr_new = this->reduceLROnPlateau(model_errors, 0.5, 100, 10, 0.1);
    //  if (lr_new < 1.0) {
    //    model_interpreter.updateSolverParams(0, lr_new);
    //    std::cout << "The learning rate has been annealed by a factor of " << lr_new << std::endl;
    //  }
    //}
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
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

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  int n_input_nodes_ = 1;
  int memory_size_ = 784;
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    assert(n_output_nodes == 2 * this->training_labels.dimension(1));
    assert(n_metric_output_nodes == this->training_labels.dimension(1));
    assert(n_input_nodes == n_input_nodes_);
    assert(memory_size == memory_size_);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {

      // Assign the final output data (only once)
      for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
        loss_output_data(batch_iter, 0, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
        loss_output_data(batch_iter, 0, nodes_iter + this->training_labels.dimension(1)) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
        metric_output_data(batch_iter, 0, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
      }

      // Assign the input data
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          int iter = memory_size * memory_iter + nodes_iter;
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], iter);
        }
      }
    }
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    assert(n_output_nodes == 2 * this->validation_labels.dimension(1));
    assert(n_metric_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == n_input_nodes_);
    assert(memory_size == memory_size_);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      // Assign the output data
      for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
        loss_output_data(batch_iter, 0, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
        loss_output_data(batch_iter, 0, nodes_iter + this->validation_labels.dimension(1)) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
        metric_output_data(batch_iter, 0, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
      }
      // Assign the input data
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          int iter = memory_size * memory_iter + nodes_iter;
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], iter);
        }
      }
    }
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{};

/**
 @brief Pixel by pixel MNIST example whereby each pixel is
   read into the model one by one and a classification
   is given after reading in all pixels

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
  - classifier (1 hot vector from 0 to 9)
 */
void main_MNIST(const std::string& data_dir, const bool& make_model, const bool& train_model) {

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = 1;

  // define the populatin trainer
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setLogging(false);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // define the data simulator
  const std::size_t input_size = 784;
  const std::size_t n_input_nodes = 28; // per column)
  const std::size_t memory_size = input_size / n_input_nodes;
  const std::size_t n_tbptt = (memory_size > 256) ? 256 : memory_size;
  const std::size_t n_labels = 10;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;
  data_simulator.memory_size_ = memory_size;
  data_simulator.n_input_nodes_ = n_input_nodes;

  // Model architecture config 0
  const std::size_t n_blocks_1 = 128;
  const std::size_t n_cells_1 = 1;
  const std::size_t n_blocks_2 = 0;
  const std::size_t n_cells_2 = 1;
  const bool add_forget_gate = false;
  const std::size_t n_hidden = 0;
  //// Model architecture config 1
  //const std::size_t n_blocks_1 = 128;
  //const std::size_t n_cells_1 = 1;
  //const std::size_t n_blocks_2 = 0;
  //const std::size_t n_cells_2 = 1;
  //const bool add_forget_gate = true;
  //const std::size_t n_hidden = 64;
  //// Model architecture config 2
  //const std::size_t n_blocks_1 = 128;
  //const std::size_t n_cells_1 = 1;
  //const std::size_t n_blocks_2 = 128;
  //const std::size_t n_cells_2 = 1;
  //const bool add_forget_gate = true;
  //const std::size_t n_hidden = 0;

  // read in the training data
  std::string training_data_filename = data_dir + "train-images.idx3-ubyte";
  std::string training_labels_filename = data_dir + "train-labels.idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // read in the validation data
  std::string validation_data_filename = data_dir + "t10k-images.idx3-ubyte";
  std::string validation_labels_filename = data_dir + "t10k-labels.idx1-ubyte";
  data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);
  data_simulator.unitScaleData();

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < data_simulator.mnist_labels.size(); ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(2, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(32);
  model_trainer.setMemorySize(memory_size);
  model_trainer.setNEpochsTraining(100001);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setNTETTSteps(1);
  model_trainer.setNTBPTTSteps(n_tbptt);
  model_trainer.setPreserveOoO(true);
  model_trainer.setFindCycles(true);
  model_trainer.setFastInterpreter(true);
  model_trainer.setLossFunctions({
    std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-24, 0.0)),
    std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-24, 1.0))
    });
  model_trainer.setLossFunctionGrads({
    std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-24, 0.0)),
    std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-24, 1.0))
    });
  model_trainer.setLossOutputNodes({
    output_nodes,
    output_nodes });
  model_trainer.setMetricFunctions({ std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes });
  model_trainer.setMetricNames({ "PrecisionMCMicro" });

  // define the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    //model_trainer.makeRNN(model, input_nodes.size(), output_nodes.size(), 128, 128, true, true, true);
    model_trainer.makeLSTM(model, input_nodes.size(), output_nodes.size(), n_blocks_1, n_cells_1, n_blocks_2, n_cells_2, n_hidden, add_forget_gate, true, true);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "LSTM_model.binary";
    const std::string interpreter_filename = data_dir + "LSTM_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("LSTM1");
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  //std::vector<Model<float>> population = { model };

  if (train_model) {
    // Train the model
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());

    //// Evolve the population
    //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    //PopulationTrainerFile<float> population_trainer_file;
    //population_trainer_file.storeModels(population, "MNIST");
    //population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation);

    //ModelFile<float> data;
    //data.storeModelCsv(population.front().getName() + "_nodes.csv",
    //  population.front().getName() + "_links.csv",
    //  population.front().getName() + "_weights.csv", 
    //  population.front(), true, true, true);
  }
  else {
    //// Evaluate the population
    //population_trainer.evaluateModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
  }
};

int main(int argc, char** argv)
{
  // Parse the user commands
  std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
  //std::string data_dir = "/home/user/data/";
  //std::string data_dir = "C:/Users/domccl/GitHub/mnist/";
  bool make_model = true, train_model = true;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    make_model = (argv[2] == std::string("true")) ? true : false;
  }
  if (argc >= 4) {
    train_model = (argv[3] == std::string("true")) ? true : false;
  }

  // run the application
  main_MNIST(data_dir, make_model, train_model);

  return 0;
}