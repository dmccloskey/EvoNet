/**TODO:  Add copyright*/

#include <EvoNet/ml/PopulationTrainerDefaultDevice.h>
#include <EvoNet/io/ModelInterpreterFileDefaultDevice.h>
#include <EvoNet/models/CVAEFullyConnDefaultDevice.h>
#include <EvoNet/simulator/MNISTSimulator.h>
#include <EvoNet/simulator/DataSimulator.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace EvoNet;
using namespace EvoNetParameters;

// Extended 
template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  int n_encodings_;
  int n_categorical_;
  int encodings_traversal_iter_ = 0;
  int categorical_traversal_iter_ = 0;
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_ + n_categorical_); // mu, logvar, logalpha
    assert(n_metric_output_nodes == n_input_pixels);
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
        TensorT inverse_tau = 3.0 / 2.0; //1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

        // Assign the input/output values
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices(batch_iter), nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices(batch_iter), nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices(batch_iter), nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = gaussian_samples(0, nodes_iter); // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
          if (nodes_iter < n_categorical_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = categorical_samples(0, nodes_iter); // sample from gumbel distribution
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_ + n_categorical_) = inverse_tau; // inverse tau
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_) = 0; // Dummy data for the KL divergence cat
          }
        }
      }
    }
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_ + n_categorical_); // mu, logvar, logalpha
    assert(n_metric_output_nodes == n_input_pixels);
    assert(n_input_nodes == n_input_pixels + n_encodings_ + 2 * n_categorical_); // Guassian sampler, Gumbel sampler, inverse tau

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // Gaussian Sampler
        Eigen::Tensor<TensorT, 2> gaussian_samples = GaussianSampler<TensorT>(1, n_encodings_);

        // Concrete Sampler
        Eigen::Tensor<TensorT, 2> categorical_samples = GumbelSampler<TensorT>(1, n_categorical_);
        TensorT inverse_tau = 3.0 / 2.0; //1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

        // Assign the input/output values
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices(batch_iter), nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices(batch_iter), nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices(batch_iter), nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = gaussian_samples(0, nodes_iter); // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
          if (nodes_iter < n_categorical_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = categorical_samples(0, nodes_iter); // sample from gumbel distribution
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_ + n_categorical_) = inverse_tau; // inverse tau
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_) = 0; // Dummy data for KL divergence cat
          }
        }
      }
    }
  }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);

    assert(n_input_nodes == n_encodings_ + n_categorical_); // Guassian encoding, Gumbel categorical

    // Initialize the gaussian encodings to random and all categorical encodings to 0
    input_data = input_data.constant(TensorT(0)); // initialize the input to 0;
    Eigen::array<Eigen::Index, 3> offsets = { 0, 0, 0 };
    Eigen::array<Eigen::Index, 3> extents = { batch_size, memory_size, n_encodings_ };
    input_data.slice(offsets, extents) = input_data.slice(offsets, extents).random();

    // Assign the encoding values by sampling the 95% confidence limits of the inverse normal distribution
    const TensorT step_size = (0.95 - 0.05) / batch_size;
    input_data.chip(encodings_traversal_iter_, 2) = (input_data.chip(encodings_traversal_iter_, 2).constant(step_size).cumsum(0) +
      input_data.chip(encodings_traversal_iter_, 2).constant(TensorT(0.05))).ndtri();

    // Assign the categorical values
    input_data.chip(n_encodings_ + categorical_traversal_iter_, 2) = input_data.chip(n_encodings_ + categorical_traversal_iter_, 2).constant(TensorT(1));

    // Increase the traversal iterators
    encodings_traversal_iter_ += 1;
    if (encodings_traversal_iter_ >= n_encodings_) {
      encodings_traversal_iter_ = 0;
      categorical_traversal_iter_ += 1;
    }
    if (categorical_traversal_iter_ >= n_categorical_) {
      categorical_traversal_iter_ = 0;
    }
  }
};

/**
 @brief Pixel reconstruction MNIST example whereby all pixels are
  linearized and read into the model.  The model then attempts to
  reconstruction the pixels using an Auto Encoder network where
  the labels of the images are disentangled from the style of the images
  using a concrete distribution and gaussian distribution, respectively

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
 */
template<class ...ParameterTypes>
void main_(const ParameterTypes& ...args) {
  auto parameters = std::make_tuple(args...);

  // define the data simulator
  const std::size_t n_pixels = 784;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;

  // read in the training data
  std::string training_data_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "train-images.idx3-ubyte";
  std::string training_labels_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "train-labels.idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, n_pixels);

  // read in the validation data
  std::string validation_data_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "t10k-images.idx3-ubyte";
  std::string validation_labels_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "t10k-labels.idx1-ubyte";
  data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, n_pixels);
  data_simulator.unitScaleData();
  data_simulator.n_encodings_ = std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get();
  data_simulator.n_categorical_ = std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get();

  // Make the input nodes
  std::vector<std::string> input_nodes;
  makeInputNodes(input_nodes, n_pixels);

  // Make the encoding nodes and add them to the input
  makeGaussianEncodingSamplerNodes(input_nodes, args...);
  makeCategoricalEncodingSamplerNodes(input_nodes, args...);
  makeCategoricalEncodingTauNodes(input_nodes, args...);

  // Make the output nodes
  std::vector<std::string> output_nodes = makeOutputNodes(n_pixels);
  std::vector<std::string> encoding_nodes_mu = makeMuEncodingNodes(args...);
  std::vector<std::string> encoding_nodes_logvar = makeLogVarEncodingNodes(args...);
  std::vector<std::string> encoding_nodes_logalpha = makeAlphaEncodingNodes(args...);
  std::vector<std::string> categorical_softmax_nodes = makeCategoricalSoftmaxNodes(args...);

  // define the model trainers and resources for the trainers
  CVAEFullyConnDefaultDevice<float> model_trainer;
  setModelTrainerParameters(model_trainer, args...);
  model_trainer.KL_divergence_warmup_ = std::get<EvoNetParameters::ModelTrainer::KLDivergenceWarmup>(parameters).get();
  model_trainer.beta_ = std::get<EvoNetParameters::ModelTrainer::Beta>(parameters).get();
  model_trainer.capacity_c_ = std::get<EvoNetParameters::ModelTrainer::CapacityC>(parameters).get();
  model_trainer.capacity_d_ = std::get<EvoNetParameters::ModelTrainer::CapacityD>(parameters).get();
  model_trainer.learning_rate_ = std::get<EvoNetParameters::ModelTrainer::LearningRate>(parameters).get();
  model_trainer.gradient_clipping_ = std::get<EvoNetParameters::ModelTrainer::GradientClipping>(parameters).get();
  model_trainer.classification_loss_weight_ = std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get();
  model_trainer.supervision_warmup_ = std::get<EvoNetParameters::Examples::SupervisionWarmup>(parameters).get();
  model_trainer.supervision_percent_ = std::get<EvoNetParameters::Examples::SupervisionPercent>(parameters).get();

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3, loss_function_helper4;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0)) };
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
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
  metric_function_helper1.metric_names_ = { "MAE" };
  metric_function_helpers.push_back(metric_function_helper1);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the model and resources
  Model<float> model;
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
  makeModelAndInterpreters(model, model_trainer, model_interpreters, model_interpreter_file, n_pixels, args...);

  // define the model logger
  ModelLogger<float> model_logger(true, true, true, false, false, true, false, true);

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
    input_nodes, model_logger, model_interpreters.front());
}

//void traverseLatentSpace(const std::string& data_dir, const bool& make_model) {
//
//  // define the model logger
//  ModelLogger<float> model_logger(true, true, false, false, false, true, false, true);
//
//  // define the data simulator
//  const std::size_t n_pixels = 784;
//  const std::size_t encoding_size = 8;
//  const std::size_t categorical_size = 10;
//  const std::size_t n_hidden = 512;
//  DataSimulatorExt<float> data_simulator;
//  data_simulator.n_encodings_ = encoding_size;
//  data_simulator.n_categorical_ = categorical_size;
//
//  // Make the input nodes
//  std::vector<std::string> input_nodes;
//
//  // Make the encoding nodes and add them to the input
//  for (int i = 0; i < encoding_size; ++i) {
//    char name_char[512];
//    sprintf(name_char, "Gaussian_encoding_%012d", i);
//    std::string name(name_char);
//    input_nodes.push_back(name);
//  }
//  for (int i = 0; i < categorical_size; ++i) {
//    char name_char[512];
//    sprintf(name_char, "Categorical_encoding-SoftMax-Out_%012d", i);
//    std::string name(name_char);
//    input_nodes.push_back(name);
//  }
//
//  // Make the output nodes
//  std::vector<std::string> output_nodes;
//  for (int i = 0; i < n_pixels; ++i) {
//    char name_char[512];
//    sprintf(name_char, "Output_%012d", i);
//    std::string name(name_char);
//    output_nodes.push_back(name);
//  }
//
//  // define the model trainers and resources for the trainers
//  ModelResources model_resources = { ModelDevice(0, 1) };
//  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
//
//  ModelTrainerExt<float> model_trainer;
//  model_trainer.setBatchSize(8); // determines the number of samples across the latent dimension
//  model_trainer.setNEpochsEvaluation(encoding_size * categorical_size); // determined by the number of latent dimensions to traverse
//  model_trainer.setMemorySize(1);
//  model_trainer.setVerbosityLevel(1);
//  model_trainer.setLogging(false, false, true);
//  model_trainer.setFindCycles(false);
//  model_trainer.setFastInterpreter(true);
//
//  std::vector<LossFunctionHelper<float>> loss_function_helpers;
//  LossFunctionHelper<float> loss_function_helper1;
//  loss_function_helper1.output_nodes_ = output_nodes;
//  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
//  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0 )) };
//  loss_function_helpers.push_back(loss_function_helper1);
//  model_trainer.setLossFunctionHelpers(loss_function_helpers);
//
//  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
//  MetricFunctionHelper<float> metric_function_helper1;
//  metric_function_helper1.output_nodes_ = output_nodes;
//  metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
//  metric_function_helper1.metric_names_ = { "MAE" };
//  metric_function_helpers.push_back(metric_function_helper1);
//  model_trainer.setMetricFunctionHelpers(metric_function_helpers);
//
//  // build the decoder and update the weights from the trained model
//  Model<float> model;
//  if (make_model) {
//    std::cout << "Making the model..." << std::endl;
//    model_trainer.makeCVAEDecoder(model, n_pixels, categorical_size, encoding_size, n_hidden);
//    std::cout << "Reading in the trained model weights..." << std::endl;
//    const std::string model_filename = data_dir + "CVAE_model.binary";
//    ModelFile<float> model_file;
//    model_file.loadWeightValuesBinary(model_filename, model.weights_);
//
//    // check that all weights were read in correctly
//    for (auto& weight_map : model.getWeightsMap()) {
//      if (weight_map.second->getInitWeight()) {
//        std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
//      }
//    }
//  }
//  else {
//    // read in the trained model
//    std::cout << "Reading in the model..." << std::endl;
//    const std::string model_filename = data_dir + "CVAEDecoder_model.binary";
//    const std::string interpreter_filename = data_dir + "CVAEDecoder_interpreter.binary";
//    ModelFile<float> model_file;
//    model_file.loadModelBinary(model_filename, model);
//    model.setId(1);
//    model.setName("CVAEDecoder1");
//    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
//    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreter);
//  }
//
//  // traverse the latent space (evaluation)
//  Eigen::Tensor<float, 4> values = model_trainer.evaluateModel(model, data_simulator, input_nodes, model_logger, model_interpreter);
//}

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
  EvoNetParameters::Examples::SupervisionWarmup supervision_warmup("supervision_warmup", false);
  EvoNetParameters::Examples::SupervisionPercent supervision_percent("supervision_percent", 0);
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
    model_type, simulation_type, supervision_warmup, supervision_percent,
    population_name, n_generations, n_interpreters, /*prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling, set_training_steps_by_model_size,*/
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, n_hidden_0, n_hidden_1, n_hidden_2, loss_fnc_weight_0, loss_fnc_weight_1, loss_fnc_weight_2, learning_rate, gradient_clipping, reset_interpreter, KL_divergence_warmup, n_encodings_continuous, n_encodings_categorical, beta, capacity_c, capacity_d/*,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error*/);

    // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = EvoNet::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  EvoNet::apply([](auto&& ...args) { main_(args ...); }, parameters);
  return 0;
}