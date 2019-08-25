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
  @param[in] add_norm Optional normalization layer between attention layers
  */
  void makeMultiHeadDotProdAttention(Model<TensorT>& model, const int& n_inputs, const int& n_outputs,
    std::vector<int> n_heads = { 8, 8 },
    std::vector<int> key_query_values_lengths = { 48, 24 },
    std::vector<int> model_lengths = { 48, 24 },
    bool add_FC = true, bool add_skip = true, bool add_norm = false, bool specify_layers = true) {
    model.setId(0);
    model.setName("DotProdAttent");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Multi-head attention
    std::vector<std::string> node_names;
    for (size_t i = 0; i < n_heads.size(); ++i) {
      // Add the attention
      std::string name_head1 = "Attention" + std::to_string(i);
      node_names = model_builder.addMultiHeadAttention(model, name_head1, name_head1,
        node_names_input, node_names_input, node_names_input,
        n_heads[i], "DotProd", model_lengths[i], key_query_values_lengths[i], key_query_values_lengths[i],
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 1e3)), 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        std::string norm_name = "Norm" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 1e3, 0.0)), 0.0, 0.0, true, specify_layers);
      }
      node_names_input = node_names;

      // Add the feedforward net
      if (add_FC) {
        std::string norm_name = "FC" + std::to_string(i);
        node_names = model_builder.addFullyConnected(model, norm_name, norm_name, node_names_input, n_inputs,
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 1e3)), 0.0f, 0.0f, false, specify_layers);
      }
      if (add_skip) {
        std::string skip_name = "Skip_FC" + std::to_string(i);
        model_builder.addSinglyConnected(model, skip_name, node_names_input, node_names,
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 1e3)), 0.0f);
      }
      if (add_norm) {
        std::string norm_name = "Norm_FC" + std::to_string(i);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
        node_names = model_builder.addSinglyConnected(model, norm_name + "-gain", norm_name + "-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 1e3, 0.0)), 0.0, 0.0, true, specify_layers);
      }
      node_names_input = node_names;
    }

    // Add the FC layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8, 1e3)), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
  }

  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false);
      ModelFile<TensorT> data;
      //data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const TensorT & model_error_train, const TensorT & model_error_test)
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 50 == 0) {
      model_logger.setLogExpectedEpoch(true);
      if (model_logger.getLogExpectedEpoch())
        model_interpreter.getModelResults(model, true, false, false);
      model_logger.writeLogs(model, n_epochs, { "Train_Error" }, { "Test_Error" }, { model_error_train }, { model_error_test }, output_nodes, expected_values);
    }
    else if (n_epochs % 10 == 0) {
      model_logger.setLogExpectedEpoch(false);
      model_logger.writeLogs(model, n_epochs, { "Train_Error" }, { "Test_Error" }, { model_error_train }, { model_error_test }, output_nodes, expected_values);
    }
  }
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == this->validation_data.dimension(1));

    // make a vector of sample_indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < this->training_data.dimension(1); ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
          output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == this->validation_data.dimension(1));

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < this->validation_data.dimension(1); ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
          output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
        }
      }
    }

    time_steps.setConstant(1.0f);
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
void main_MNIST(const bool& make_model, const bool& train_model) {

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = 1;

  // define the populatin trainer
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setLogging(false);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);
  //ModelLogger<float> model_logger(true, true, true, true, true, false, true, true);

  // define the data simulator
  const std::size_t input_size = 784;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;

  // read in the training data
  //const std::string training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
  //const std::string training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
  const std::string training_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-images-idx3-ubyte";
  const std::string training_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-labels-idx1-ubyte";
  //const std::string training_data_filename = "/home/user/data/train-images-idx3-ubyte";
  //const std::string training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // read in the validation data
  //const std::string validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
  //const std::string validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
  const std::string validation_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-images-idx3-ubyte";
  const std::string validation_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-labels-idx1-ubyte";
  //const std::string validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
  //const std::string validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
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
  model_trainer.setBatchSize(64);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(200001);
  model_trainer.setNEpochsValidation(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsLossOp<float>()) });
  model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsLossGradOp<float>()) });
  model_trainer.setLossOutputNodes({ output_nodes });

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 2,2 }, { 24,24 }, { 48, 48 }, false, false, false, true); // Test model
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 8, 8 }, { 48, 24 }, { 256, 128 }, false, false, false, true); // Solving model
    //model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 12, 8 }, { 48, 24 }, { 512, 128 }, false, false, false); // Solving model
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string data_dir = "/home/user/code/build/";
    const std::string model_filename = data_dir + "DotProdAtt_1000_model.binary";
    const std::string interpreter_filename = data_dir + "DotProdAtt_1000_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("DotProdAtt1");
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  std::vector<Model<float>> population = { model };

  if (train_model) {
    // Train the model
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());

    PopulationTrainerFile<float> population_trainer_file;
    population_trainer_file.storeModels(population, "MNIST");

    //// Evolve the population
    //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    //PopulationTrainerFile<float> population_trainer_file;
    //population_trainer_file.storeModels(population, "MNIST");
    //population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation);
  }
  else {
    // Evaluate the population
    population_trainer.evaluateModels(
      population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
  }
}

int main(int argc, char** argv)
{
  // run the application
  main_MNIST(true, true);

  return 0;
}