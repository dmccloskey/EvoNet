/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerExperimentalGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicatorExperimental.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/simulator/AddProbSimulator.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>
#include <SmartPeak/io/Parameters.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

template<typename TensorT>
class DataSimulatorExt : public AddProbSimulator<TensorT>
{
public:
  void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    // sequence length
    const int sequence_length = n_input_nodes / 2;
    assert(sequence_length == this->sequence_length_);

    //// generate a new sequence 
    //Eigen::Tensor<TensorT, 1> random_sequence(this->sequence_length_);
    //Eigen::Tensor<TensorT, 1> mask_sequence(this->sequence_length_);
    //float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);

    // Generate the input and output data for training [BUG FREE]
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

        // generate a new sequence 
        Eigen::Tensor<TensorT, 1> random_sequence(this->sequence_length_);
        Eigen::Tensor<TensorT, 1> mask_sequence(this->sequence_length_);
        float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);

        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          for (int nodes_iter = 0; nodes_iter < n_input_nodes / 2; ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = random_sequence(nodes_iter); // random sequence
            input_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2, epochs_iter) = mask_sequence(nodes_iter); // mask sequence
            //std::cout << "Node: " << nodes_iter << ";Rand: " << input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) << ";Mask: " << input_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2, epochs_iter) << std::endl;
          }
          for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = result;
          }
        }
      }
    }
    //std::cout << "Input data: " << input_data << std::endl; // [TESTS: convert to a test!]
    //std::cout << "Output data: " << output_data << std::endl; // [TESTS: convert to a test!]

    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)override { simulateData(input_data, output_data, time_steps); }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)override { simulateData(input_data, output_data, time_steps); }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) override {};
  void simulateData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    // sequence length
    const int sequence_length = n_input_nodes / 2;
    assert(sequence_length == this->sequence_length_);

    // Generate the input and output data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {

      // generate a new sequence 
      Eigen::Tensor<TensorT, 1> random_sequence(this->sequence_length_);
      Eigen::Tensor<TensorT, 1> mask_sequence(this->sequence_length_);
      float result = this->AddProb(random_sequence, mask_sequence, this->n_mask_);

      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes / 2; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = random_sequence(nodes_iter); // random sequence
          input_data(batch_iter, memory_iter, nodes_iter + n_input_nodes / 2) = mask_sequence(nodes_iter); // mask sequence
        }
        for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
          output_data(batch_iter, memory_iter, nodes_iter) = result;
          metric_data(batch_iter, memory_iter, nodes_iter) = result;
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, output_data, metric_data, time_steps); }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, output_data, metric_data, time_steps); }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_data, Eigen::Tensor<TensorT, 2>& time_steps)override { simulateData(input_data, metric_data, Eigen::Tensor<TensorT, 3>(), time_steps); }
};

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief Minimal network
  */
  void makeModelMinimal(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, int n_hidden_0 = 1, bool specify_layers = false)
  {
    model.setId(0);
    model.setName("AddProbAtt-Min");
    ModelBuilder<TensorT> model_builder;
    // Add the inputs
    std::vector<std::string> node_names_random = model_builder.addInputNodes(model, "Random", "Random", n_inputs, specify_layers);
    std::vector<std::string> node_names_mask = model_builder.addInputNodes(model, "Mask", "Mask", n_inputs, specify_layers);
    // Define the activation 
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());
    // Define the solver
    auto solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-4, 0.9, 10));
    // Add the hidden layer
    std::vector<std::string> node_names = model_builder.addFullyConnected(model, "HiddenR", "HiddenR", node_names_random, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_random.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    model_builder.addFullyConnected(model, "HiddenR", node_names_mask, node_names,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_mask.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, specify_layers);
    // Add the output layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, n_outputs,
      activation_output, activation_output_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::unmodifiable);
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
  @brief Minimal network required to solve the addition problem
  */
  void makeModelSolution(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, bool init_weight_soln = true, bool specify_layers = false)
  {
    model.setId(0);
    model.setName("AddProbAtt-Solution");
    ModelBuilder<TensorT> model_builder;
    // Add the inputs
    std::vector<std::string> node_names_random = model_builder.addInputNodes(model, "Random", "Random", n_inputs, specify_layers);
    std::vector<std::string> node_names_mask = model_builder.addInputNodes(model, "Mask", "Mask", n_inputs, specify_layers);
    // Define the activation 
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_output_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());
    // Define the solver and weight init ops
    std::shared_ptr<SolverOp<TensorT>> solver_op;
    std::shared_ptr<WeightInitOp<TensorT>> weight_init_1, weight_init_2;
    if (init_weight_soln) {
      solver_op = std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>());
      weight_init_1 = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1));
      weight_init_2 = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1));
    }
    else {
      solver_op = std::make_shared<SGDOp<TensorT>>(SGDOp<TensorT>(1e-4, 0.9, 10));
      weight_init_1 = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names_random.size() + n_inputs) / 2, 1));
      weight_init_2 = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(n_inputs, 2));
    }
    // Add the hidden layer
    std::vector<std::string> node_names = model_builder.addSinglyConnected(model, "HiddenR", "HiddenR", node_names_random, n_inputs,
      activation, activation_grad,
      std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),
      std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()),
      std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
      weight_init_1, solver_op, 0.0f, 0.0f, false, specify_layers);
    model_builder.addSinglyConnected(model, "HiddenR", node_names_mask, node_names,
      weight_init_1, solver_op, 0.0f, specify_layers);
    // Add the output layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, n_outputs,
      activation_output, activation_output_grad, integration_op, integration_error_op, integration_weight_grad_op,
      weight_init_2, solver_op, 0.0f, 0.0f, false, specify_layers);
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::unmodifiable);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
  { // Left blank intentionally to prevent writing of files during training
  }
  void validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override
  { // Left blank intentionally to prevent writing of files during validation
  }
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test,
    const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) { // FIXME
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
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
  void evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const Eigen::Tensor<TensorT, 1>& model_metrics) override
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1 == 0) { // FIXME
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_headers;
    std::vector<TensorT> log_values;
    int metric_iter = 0;
    for (const std::string& metric_name : this->getMetricNamesLinearized()) {
      log_headers.push_back(metric_name);
      log_values.push_back(model_metrics(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_headers, {}, log_values, {}, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors)override {
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicatorExperimental<TensorT>
{
public:
  bool set_modification_rate_by_prev_error_ = false;
  bool set_modification_rate_fixed_ = false;
  /*
  @brief Implementation of the `adaptiveReplicatorScheduler`
  */
  void adaptiveReplicatorScheduler(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)override
  {
    // Adjust the models modifications rates
    if (set_modification_rate_by_prev_error_) this->setModificationRateByPrevError(n_generations, models, models_errors_per_generations);
    if (set_modification_rate_fixed_) this->setModificationRateFixed(n_generations, models, models_errors_per_generations);
  }
};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerExperimentalGpu<TensorT>
{
public:
  bool set_population_size_fixed_ = false;
  bool set_population_size_doubling_ = false;
  /*
  @brief Implementation of the `adaptivePopulationScheduler`
  */
  void adaptivePopulationScheduler(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)override
  {
    // Adjust the population size
    if (set_population_size_fixed_) this->setPopulationSizeFixed(n_generations, models, models_errors_per_generations);
    else if (set_population_size_doubling_) this->setPopulationSizeDoubling(n_generations, models, models_errors_per_generations);
  }
  /*
  @brief Implementation of the `trainingPopulationLogger`
  */
  void trainingPopulationLogger(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    PopulationLogger<TensorT>& population_logger,
    const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation) override {
    // Export the selected models
    for (auto& model : models) {
      ModelFile<TensorT> data;
      data.storeModelCsv(model.getName() + "_" + std::to_string(n_generations) + "_nodes.csv",
        model.getName() + "_" + std::to_string(n_generations) + "_links.csv",
        model.getName() + "_" + std::to_string(n_generations) + "_weights.csv", model);
    }
    // Log the population statistics
    population_logger.writeLogs(n_generations, models_validation_errors_per_generation);
  }
};

template<class ...ParameterTypes>
void main_KineticModel(const ParameterTypes& ...args) {
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
  population_trainer.set_population_size_fixed_ = std::get<EvoNetParameters::PopulationTrainer::SetPopulationSizeFixed>(parameters).get();
  population_trainer.set_population_size_doubling_ = std::get<EvoNetParameters::PopulationTrainer::SetPopulationSizeDoubling>(parameters).get();

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the multithreading parameters
  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = (std::get<EvoNetParameters::PopulationTrainer::NInterpreters>(parameters).get() > n_hard_threads) ? n_hard_threads : std::get<EvoNetParameters::PopulationTrainer::NInterpreters>(parameters).get(); // the number of threads

  // define the data simulator
  DataSimulatorExt<float> data_simulator;
  data_simulator.n_mask_ = std::get<EvoNetParameters::Examples::NMask>(parameters).get();
  data_simulator.sequence_length_ = std::get<EvoNetParameters::Examples::SequenceLength>(parameters).get();

  // define the input/output nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < data_simulator.sequence_length_; ++i) {
    char name_char[512];
    sprintf(name_char, "Random_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < data_simulator.sequence_length_; ++i) {
    char name_char[512];
    sprintf(name_char, "Mask_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  std::vector<std::string> output_nodes = { "Output_000000000000" };

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
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
  LossFunctionHelper<float> loss_function_helper2;
  loss_function_helper2.output_nodes_ = output_nodes;
  loss_function_helper2.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-24, 1.0)) };
  loss_function_helper2.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-24, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper2);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Mean")), std::make_shared<EuclideanDistOp<float>>(EuclideanDistOp<float>("Var")) };
  metric_function_helper1.metric_names_ = { "EuclideanDist-Mean", "EuclideanDist-Var" };
  metric_function_helpers.push_back(metric_function_helper1);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
    std::make_pair(std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>())),
    std::make_pair(std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>())),
    std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
    std::make_pair(std::make_shared<TanHOp<float>>(TanHOp<float>()), std::make_shared<TanHGradOp<float>>(TanHGradOp<float>()))//,
    //std::make_pair(std::make_shared<ExponentialOp<float>>(ExponentialOp<float>()), std::make_shared<ExponentialGradOp<float>>(ExponentialGradOp<float>())),
    //std::make_pair(std::make_shared<LogOp<float>>(LogOp<float>()), std::make_shared<LogGradOp<float>>(LogGradOp<float>())),
    //std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
    });
  model_replicator.setNodeIntegrations({ std::make_tuple(std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>())),
    std::make_tuple(std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>())),
    //std::make_tuple(std::make_shared<MeanOp<float>>(MeanOp<float>()), std::make_shared<MeanErrorOp<float>>(MeanErrorOp<float>()), std::make_shared<MeanWeightGradO<float>>(MeanWeightGradOp<float>())),
    //std::make_tuple(std::make_shared<VarModOp<float>>(VarModOp<float>()), std::make_shared<VarModErrorOp<float>>(VarModErrorOp<float>()), std::make_shared<VarModWeightGradOp<float>>(VarModWeightGradOp<float>())),
    //std::make_tuple(std::make_shared<CountOp<float>>(CountOp<float>()), std::make_shared<CountErrorOp<float>>(CountErrorOp<float>()), std::make_shared<CountWeightGradOp<float>>(CountWeightGradOp<float>()))
    });
  model_replicator.set_modification_rate_by_prev_error_ = std::get<EvoNetParameters::ModelReplicator::SetModificationRateByPrevError>(parameters).get();
  model_replicator.set_modification_rate_fixed_ = std::get<EvoNetParameters::ModelReplicator::SetModificationRateFixed>(parameters).get();
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
    if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Minimal") model_trainer.makeModelMinimal(model, input_nodes.size() / 2, output_nodes.size());
    else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Solution") model_trainer.makeModelSolution(model, input_nodes.size() / 2, output_nodes.size(), false, false);
    //else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "DotProdAtt") model_trainer.makeModelDotProdAtt(model, (int)(input_nodes.size() / 2), output_nodes.size(), { 4 }, { 8 }, { 16 }, false, false, false, true);
    model.setId(0);
  }
  else if (std::get<EvoNetParameters::Main::LoadModelBinary>(parameters).get()) {
    // read in the trained model
    std::cout << "Reading in the model from binary..." << std::endl;
    ModelFile<float> model_file;
    model_file.loadModelBinary(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model);
    model.setId(1);
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_interpreter.binary", model_interpreters[0]); // FIX ME!
  }
  else if (std::get<EvoNetParameters::Main::LoadModelCsv>(parameters).get()) {
    // read in the trained model
    std::cout << "Reading in the model from csv..." << std::endl;
    ModelFile<float> model_file;
    model_file.loadModelCsv(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_nodes.csv", std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_links.csv", std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_weights.csv", model, true, true, true);
    model.setId(1);
  }
  model.setName(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory

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
      population, std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get(), //So that all output will be written to a specific directory
      model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    PopulationTrainerFile<float> population_trainer_file;
    population_trainer_file.storeModels(population, std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get());
    population_trainer_file.storeModelValidations(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get() + "Errors.csv", models_validation_errors_per_generation);
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
  EvoNetParameters::Main::DeviceId device_id("device_id", 0);
  EvoNetParameters::Main::ModelName model_name("model_name", "");
  EvoNetParameters::Main::MakeModel make_model("make_model", true);
  EvoNetParameters::Main::LoadModelCsv load_model_csv("load_model_csv", false);
  EvoNetParameters::Main::LoadModelBinary load_model_binary("load_model_binary", false);
  EvoNetParameters::Main::TrainModel train_model("train_model", true);
  EvoNetParameters::Main::EvolveModel evolve_model("evolve_model", false);
  EvoNetParameters::Main::EvaluateModel evaluate_model("evaluate_model", false);
  EvoNetParameters::Examples::NMask n_mask("n_mask", 2);
  EvoNetParameters::Examples::SequenceLength sequence_length("sequence_length", 25);
  EvoNetParameters::Examples::ModelType model_type("model_type", "Solution");
  EvoNetParameters::Examples::SimulationType simulation_type("simulation_type", "");
  EvoNetParameters::Examples::BiochemicalRxnsFilename biochemical_rxns_filename("biochemical_rxns_filename", "iJO1366.csv");
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
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
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
  auto parameters = std::make_tuple(id, data_dir,
    device_id, model_name, make_model, load_model_csv, load_model_binary, train_model, evolve_model, evaluate_model,
    n_mask, sequence_length, model_type, simulation_type, biochemical_rxns_filename,
    population_name, n_generations, n_interpreters, prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling,
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, reset_interpreter,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error);

  // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = SmartPeak::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  SmartPeak::apply([](auto&& ...args) { main_KineticModel(args ...); }, parameters);
  return 0;
}