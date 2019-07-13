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

  @param[in, out] model The network model
  @param[in] n_inputs The number of pixel inputs
  @param[in] n_outputs The number of classifier outputs
  @param[in] n_blocks The number of LSTM blocks to add to the network
  @param[in] n_cells The number of cells in each LSTM block
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeLSTM(Model<TensorT>& model, int n_inputs = 784, int n_outputs = 10, int n_blocks = 512, int n_cells = 1, int n_hidden = 32, bool specify_layers = true) {
    model.setId(0);
    model.setName("LSTM");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Add the LSTM layer(s)
    std::vector<std::string> node_names = model_builder.addLSTM(model, "LSTM-01", "LSTM-01", node_names_input, n_blocks, n_cells,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<float>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<float>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(0.4)),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_blocks) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-3, 10.0)),
      0.0f, 0.0f, false, true, 1, specify_layers);
    node_names = model_builder.addLSTM(model, "LSTM-02", "LSTM-02", node_names, n_blocks, n_cells,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<float>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<float>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(0.4)),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_blocks) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-3, 10.0)),
      0.0f, 0.0f, false, true, 1, specify_layers);

   // Add a fully connected layer
   node_names = model_builder.addFullyConnected(model, "FC-01", "FC-01", node_names, n_hidden,
     std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
     std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
     std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
     std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
     std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
     std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden)/2, 1)),
     std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, false, specify_layers);

    // Add a final output layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(1/(TensorT)node_names.size(), 10/(TensorT)node_names.size())),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
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
      model_interpreter.getModelResults(model, false, true, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const TensorT & model_error_train, const TensorT & model_error_test,
    const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test)
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedPredictedEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      //model_logger.setLogExpectedPredictedEpoch(true);
      model_logger.initLogs(model);
    }

    //// Per n epoch logging
    //if (n_epochs % 10 == 0) {
    //  model_logger.setLogExpectedPredictedEpoch(true);
    //  model_interpreter.getModelResults(model, true, false, false);
    //}

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
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values);
  }
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    assert(n_output_nodes == this->training_labels.dimension(1));
    assert(n_metric_output_nodes == this->training_labels.dimension(1));
    assert(n_input_nodes == 1);
    assert(memory_size == 784);

    // make the start and end sample indices [BUG FREE]
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training [BUG FREE]
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], memory_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
        }
      }
    }
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_metric_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == 1);
    assert(memory_size == 784);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], memory_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
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
  const std::size_t n_labels = 10;
  const std::size_t n_blocks = 128;
  const std::size_t n_hidden = 32;
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
  for (int i = 0; i < 1; ++i) {
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
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(32);
  model_trainer.setMemorySize(input_size);
  model_trainer.setNEpochsTraining(500001);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, true, false);
  model_trainer.setNTETTSteps(1);
  model_trainer.setNTBPTTSteps(256);
  //model_trainer.setNTBPTTSteps(input_size);
  model_trainer.setPreserveOoO(true);
  model_trainer.setFindCycles(true);
  model_trainer.setFastInterpreter(true);
  model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>(1e-6, 1.0)) });
  model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>(1e-6, 1.0)) });
  model_trainer.setLossOutputNodes({ output_nodes });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes });
  model_trainer.setMetricNames({ "PrecisionMCMicro" });

  // define the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    model_trainer.makeLSTM(model, input_nodes.size(), output_nodes.size(), n_blocks, 1, n_hidden, true);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "LSTM_9000_model.binary";
    const std::string interpreter_filename = data_dir + "LSTM_9000_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("LSTM1");
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  std::vector<Model<float>> population = { model };

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
    // Evaluate the population
    population_trainer.evaluateModels(
      population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
  }
};

int main(int argc, char** argv)
{
  // define the data directory
  //std::string data_dir = "/home/user/data/";
  std::string data_dir = "C:/Users/domccl/GitHub/mnist/";
  //std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";

  // run the application
  main_MNIST(data_dir, true, true);

  return 0;
}