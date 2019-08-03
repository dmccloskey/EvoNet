/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/io/ModelFile.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Extended 
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief AutoEncoder that encodes the labels using a concrete distribution
    and style using a gaussian distribution

  References:
  Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey. "Adversarial Autoencoders" 2015.  arXiv:1511.05644
  https://github.com/musyoku/adversarial-autoencoder/blob/master/run/semi-supervised/regularize_z/model.py

  @param[in, out] model The network model
  @param[in] n_inputs The number of input pixels
  @param[in] n_categorical The length of the categorical layer
  @param[in] n_encodings The length of the encodings layer
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation

  */
  void makeCVAE(Model<TensorT>& model, int n_inputs = 784, int n_categorical = 10, int n_encodings = 64, int n_hidden_0 = 512, bool specify_layer = true) {
    model.setId(0);
    model.setName("CVAE");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layer);

    // Add the Endocer FC layers
    std::vector<std::string> node_names, node_names_mu, node_names_logvar, node_names_logalpha;
    node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + node_names.size()) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layer);
    node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names.size() + node_names.size()) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layer);
    node_names_mu = model_builder.addFullyConnected(model, "Mu", "Mu", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layer);
    node_names_logvar = model_builder.addFullyConnected(model, "LogVar", "LogVar", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layer);
    node_names_logalpha = model_builder.addFullyConnected(model, "LogAlpha", "LogAlpha", node_names, n_categorical,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layer);

    // Add the Encoding layers
    std::vector<std::string> node_names_Gencoder = model_builder.addGaussianEncoding(model, "Gaussian_encoding", "Gaussian_encoding", node_names_mu, node_names_logvar, true);
    std::vector<std::string> node_names_Cencoder = model_builder.addCategoricalEncoding(model, "Categorical_encoding", "Categorical_encoding", node_names_logalpha, true);

    // Add the Decoder FC layers
    node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names_Gencoder, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_0) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layer);
    model_builder.addFullyConnected(model, "DE0", node_names_Cencoder, node_names,
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_0) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, specify_layer);
    node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layer);
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_inputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>(1.0)),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>(1.0)),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
  }
  Model<TensorT> makeModel() { return Model<TensorT>(); }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
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
  int n_encodings_;
  int n_categorical_;
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_ + 2*n_categorical_); // mu, logvar, logalpha, XEntropy
    assert(n_metric_output_nodes == n_categorical_ + n_input_pixels);
    assert(n_input_nodes == n_input_pixels + n_encodings_ + 2 * n_categorical_); // Guassian sampler, Gumbel sampler, inverse tau

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // Gaussian Sampler
        Eigen::Tensor<TensorT, 2> gaussian_samples = GaussianSampler<TensorT>(1, n_encodings_);

        // Concrete Sampler
        Eigen::Tensor<TensorT, 2> categorical_samples = GumbelSampler<TensorT>(1, n_categorical_);
        TensorT inverse_tau = 1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

        // Assign the input/output values
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = gaussian_samples(0, nodes_iter); // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
          if (nodes_iter < n_categorical_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = categorical_samples(0, nodes_iter); // sample from gumbel distribution
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_ + n_categorical_) = inverse_tau; // inverse tau
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_) = 0; // Dummy data for the KL divergence cat
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_ + n_categorical_) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter); // Expected label
            metric_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter); // Expected label
          }
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
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_ + 2 * n_categorical_); // mu, logvar, logalpha, XEntropy
    assert(n_metric_output_nodes == n_categorical_ + n_input_pixels);
    assert(n_input_nodes == n_input_pixels + n_encodings_ + 2 * n_categorical_); // Guassian sampler, Gumbel sampler, inverse tau

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // Gaussian Sampler
        Eigen::Tensor<TensorT, 2> gaussian_samples = GaussianSampler<TensorT>(1, n_encodings_);

        // Concrete Sampler
        Eigen::Tensor<TensorT, 2> categorical_samples = GumbelSampler<TensorT>(1, n_categorical_);
        TensorT inverse_tau = 1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

        // Assign the input/output values
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = gaussian_samples(0, nodes_iter); // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
          if (nodes_iter < n_categorical_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = categorical_samples(0, nodes_iter); // sample from gumbel distribution
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_ + n_categorical_) = inverse_tau; // inverse tau
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_) = 0; // Dummy data for KL divergence cat
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_ + n_categorical_) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter); // Expected label
            metric_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter); // Expected label
          }
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
 @brief Pixel reconstruction MNIST example whereby all pixels are
  linearized and read into the model.  The model then attempts to
  reconstruction the pixels using an Auto Encoder network where
  the labels of the images are disentangled from the style of the images
  using a concrete distribution and gaussian distribution, respectively

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
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
  const std::size_t encoding_size = 64;
  const std::size_t categorical_size = 10;
  const std::size_t n_hidden = 128;
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
  data_simulator.n_encodings_ = encoding_size;
  data_simulator.n_categorical_ = categorical_size;

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the encoding nodes and add them to the input
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Gaussian_encoding_%012d-Sampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding_%012d-GumbelSampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding_%012d-InverseTau", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // Make the mu nodes
  std::vector<std::string> encoding_nodes_mu;
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Mu_%012d", i);
    std::string name(name_char);
    encoding_nodes_mu.push_back(name);
  }

  // Make the var nodes
  std::vector<std::string> encoding_nodes_logvar;
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "LogVar_%012d", i);
    std::string name(name_char);
    encoding_nodes_logvar.push_back(name);
  }

  // Make the alpha nodes
  std::vector<std::string> encoding_nodes_logalpha;
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "LogAlpha_%012d", i);
    std::string name(name_char);
    encoding_nodes_logalpha.push_back(name);
  }

  // Make the categorical output nodes
  std::vector<std::string> categorical_nodes_output;
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding-SoftMax-Out_%012d", i);
    std::string name(name_char);
    categorical_nodes_output.push_back(name);
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  //model_trainer.setBatchSize(1); // evaluation only
  model_trainer.setBatchSize(64);
  model_trainer.setNEpochsTraining(100001);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNEpochsEvaluation(0);
  model_trainer.setMemorySize(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, true);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setLossFunctions({
    //std::shared_ptr<LossFunctionOp<float>>(new MSELossOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceMuOp<float>(1e-6, 0.5)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceLogVarOp<float>(1e-6, 0.5)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceCatOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>(1e-6, 0.1)) });
  model_trainer.setLossFunctionGrads({
    //std::shared_ptr<LossFunctionGradOp<float>>(new MSELossGradOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceMuGradOp<float>(1e-6, 0.5)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceLogVarGradOp<float>(1e-6, 0.5)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceCatGradOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>(1e-6, 0.1)) });
  model_trainer.setLossOutputNodes({ output_nodes, encoding_nodes_mu, encoding_nodes_logvar, encoding_nodes_logalpha, categorical_nodes_output });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new MAEOp<float>()), std::shared_ptr<MetricFunctionOp<float>>(new PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes, categorical_nodes_output });
  model_trainer.setMetricNames({ "MAE", "PrecisionMCMicro" });

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    model_trainer.makeCVAE(model, input_size, categorical_size, encoding_size, n_hidden);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "CVAE_9000_model.binary";
    const std::string interpreter_filename = data_dir + "CVAE_9000_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("CVAE1");
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  std::vector<Model<float>> population = { model };

  if (train_model) {
    // Train the model
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());

    //// Evolve the population
    //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    //	population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

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
  // define the data directory
  //std::string data_dir = "/home/user/data/";
  std::string data_dir = "C:/Users/domccl/GitHub/mnist/";
  //std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";

  // run the application
  main_MNIST(data_dir, true, true);
}