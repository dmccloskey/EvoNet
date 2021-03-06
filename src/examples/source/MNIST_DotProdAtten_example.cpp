/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief Multi-head self-attention dot product classifier

  @param[in, out] model The network model
  @param[in] n_inputs The number of pixel input nodes
  @param[in] n_outputs The number of classification output nodes
  @param[in] n_heads A vector of the the number of attention heads per attention layer
  @param[in] key_query_values_lengths A vector of the key/query/values lengths per attention layer
  @param[in] model_lengths A vector of the attention model lengths per attention layer
  @param[in] add_FC Optional fully connected layer between attention heads
  @param[in] add_skip Optional skip connections between attention layers
  @param[in] add_feature_norm Optional normalization layer between attention layers
  */
  void makeMultiHeadDotProdAttention(Model<TensorT>& model, const int& n_inputs, const int& n_outputs,
    std::vector<int> n_heads = { 8, 8 },
    std::vector<int> key_query_values_lengths = { 48, 24 },
    std::vector<int> model_lengths = { 48, 24 },
    bool add_FC = true, bool add_skip = true, bool add_feature_norm = false, bool specify_layers = true) {
    model.setId(0);
    model.setName("DotProdAttent");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

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
    std::shared_ptr<ActivationOp<TensorT>> activation_feature_norm = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_feature_norm_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 10));

    // Multi-head attention
    std::vector<std::string> node_names;
    for (size_t i = 0; i < n_heads.size(); ++i) {
      // Add the attention
      std::string name_head1 = "Attention" + std::to_string(i);
      node_names = model_builder.addMultiHeadAttention(model, name_head1, name_head1,
        node_names_input, node_names_input, node_names_input,
        n_heads[i], "DotProd", model_lengths[i], key_query_values_lengths[i], key_query_values_lengths[i],
        activation, activation_grad,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_feature_norm) {
        std::string norm_name = "Norm" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
          activation_feature_norm, activation_feature_norm_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
      }
      node_names_input = node_names;

      // Add the feedforward net
      if (add_FC) {
        std::string norm_name = "FC" + std::to_string(i);
        node_names = model_builder.addFullyConnected(model, norm_name, norm_name, node_names_input, n_inputs,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
      }
      if (add_skip) {
        std::string skip_name = "Skip_FC" + std::to_string(i);
        model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(n_inputs, 2)),
          solver_op, 0.0f);
      }
      if (add_feature_norm) {
        std::string norm_name = "Norm_FC" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
          activation_feature_norm, activation_feature_norm_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
      }
      node_names_input = node_names;
    }

    // Add the FC layer
    node_names = model_builder.addFullyConnected(model, "FC-Out", "FC-Out", node_names, n_outputs,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);

    // Add the actual output nodes
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
  }

  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    if (n_epochs % 1000 == 0 /*&& n_epochs != 0*/) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);

      //// Save to .csv
      //data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);

      // Save to binary
      ModelFile<TensorT> data;
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
  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == this->validation_data.dimension(1));

    // make a vector of sample_indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, n_epochs);

    // Reformat the input data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
          for (int nodes_iter = 0; nodes_iter < this->training_data.dimension(1); ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            //input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
          }
          for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            //output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[0], nodes_iter); // test on only 1 sample
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == this->validation_data.dimension(1));

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, n_epochs);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
          for (int nodes_iter = 0; nodes_iter < this->validation_data.dimension(1); ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
          }
          for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
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
    assert(n_input_nodes == 784);
    assert(memory_size == 1);

    // make the start and end sample indices [BUG FREE]
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training [BUG FREE]
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter + this->training_labels.dimension(1)) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
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
    assert(n_input_nodes == 784);
    assert(memory_size == 1);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter + this->validation_labels.dimension(1)) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
        }
      }
    }
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
  classify the image using a Dot product attention architecture

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
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;

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
  for (int i = 0; i < input_size; ++i) {
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
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(128);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(100001);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setLossFunctions({
    std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-24, 0.0)),
    std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>())
    });
  model_trainer.setLossFunctionGrads({
    std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-24, 0.0)),
    std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>())
    });
  model_trainer.setLossOutputNodes({
    output_nodes,
    output_nodes });
  model_trainer.setMetricFunctions({ std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes });
  model_trainer.setMetricNames({ "PrecisionMCMicro" });

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 2,2 }, { 24,24 }, { 48, 48 }, true, false, true, true); // Test model
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 8, 8 }, { 48, 24 }, { 256, 128 }, false, false, false, true); // Solving model
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 12, 8 }, { 48, 24 }, { 512, 128 }, false, false, false); // Solving model
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "DotProdAtt_model.binary";
    const std::string interpreter_filename = data_dir + "DotProdAtt_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("DotProdAtt1");
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  //std::vector<Model<float>> population = { model };

  if (train_model) {
    // Train the model
    std::cout << "Training the model..." << std::endl;
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());

    //// Evolve the population
    //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    //PopulationTrainerFile<float> population_trainer_file;
    //population_trainer_file.storeModels(population, "MNIST");
    //population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation);
  }
  else {
    //// Evaluate the population
    //population_trainer.evaluateModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
  }
}

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