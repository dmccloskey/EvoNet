/**TODO:  Add copyright*/

#include <EvoNet/ml/PopulationTrainerDefaultDevice.h>
#include <EvoNet/ml/ModelReplicator.h>
#include <EvoNet/ml/ModelBuilder.h>
#include <EvoNet/io/PopulationTrainerFile.h>
#include <EvoNet/simulator/MetabolomicsReconstructionDataSimulator.h>
#include <EvoNet/models/CVAEFullyConnDefaultDevice.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace EvoNet;

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
    std::cout << "The number of labels " << n_labels_training << " does not match the number of discrete encodings " << std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get() << std::endl;
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
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  setModelInterpreterParameters(model_interpreters, args...);

  // define the model trainer
  CVAEFullyConnDefaultDevice<float> model_trainer;
  setModelTrainerParameters(model_trainer, args...);
  model_trainer.setNEpochsTraining(std::get<EvoNetParameters::ModelTrainer::NEpochsTraining>(parameters).get() * 10 + 1); // iterate through the cache 10x
  model_trainer.KL_divergence_warmup_ = std::get<EvoNetParameters::ModelTrainer::KLDivergenceWarmup>(parameters).get();
  model_trainer.beta_ = std::get<EvoNetParameters::ModelTrainer::Beta>(parameters).get();
  model_trainer.capacity_c_ = std::get<EvoNetParameters::ModelTrainer::CapacityC>(parameters).get();
  model_trainer.capacity_d_ = std::get<EvoNetParameters::ModelTrainer::CapacityD>(parameters).get();
  model_trainer.learning_rate_ = std::get<EvoNetParameters::ModelTrainer::LearningRate>(parameters).get();
  model_trainer.gradient_clipping_ = std::get<EvoNetParameters::ModelTrainer::GradientClipping>(parameters).get();

  std::shared_ptr<LossFunctionOp<float>> loss_function_op;
  std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad_op;
  if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MSE")) {
    loss_function_op = std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    loss_function_grad_op = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MAE")) {
    loss_function_op = std::make_shared<MAELossOp<float>>(MAELossOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    loss_function_grad_op = std::make_shared<MAELossGradOp<float>>(MAELossGradOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MLE")) {
    loss_function_op = std::make_shared<MLELossOp<float>>(MLELossOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    loss_function_grad_op = std::make_shared<MLELossGradOp<float>>(MLELossGradOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MAPE")) {
    loss_function_op = std::make_shared<MAPELossOp<float>>(MAPELossOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    loss_function_grad_op = std::make_shared<MAPELossGradOp<float>>(MAPELossGradOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
  }
  else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("BCEWithLogits")) {
    loss_function_op = std::make_shared<BCEWithLogitsLossOp<float>>(BCEWithLogitsLossOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    loss_function_grad_op = std::make_shared<BCEWithLogitsLossGradOp<float>>(BCEWithLogitsLossGradOp<float>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
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
    if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "EncDec") {
      model_trainer.makeCVAE(model, n_input_nodes,
        std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
    }
    else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Enc") {
      // make the encoder only
      model_trainer.makeCVAEEncoder(model, n_input_nodes,
        std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);

      // read in the weights
      ModelFile<float> model_file;
      model_file.loadWeightValuesBinary(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model.weights_);

      // check that all weights were read in correctly
      for (auto& weight_map : model.getWeightsMap()) {
        if (weight_map.second->getInitWeight()) {
          std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
        }
      }
    }
    else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Dec") {
      // make the decoder only
      model_trainer.makeCVAEDecoder(model, n_input_nodes,
        std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
        std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);

      // read in the weights
      ModelFile<float> model_file;
      model_file.loadWeightValuesBinary(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model.weights_);

      // check that all weights were read in correctly
      for (auto& weight_map : model.getWeightsMap()) {
        if (weight_map.second->getInitWeight()) {
          std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
        }
      }
    }
  }
  else {
    ModelFile<float> model_file;
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    loadModelFromParameters(model, model_interpreters.at(0), model_file, model_interpreter_file, args...);
  }
  model.setName(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
    input_nodes, model_logger, model_interpreters.front());
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
  EvoNetParameters::Examples::ModelType model_type("model_type", "EncDec"); // Options include EncDec, Enc, Dec
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
  EvoNetParameters::ModelTrainer::LossFncWeight0 loss_fnc_weight_0("loss_fnc_weight_0", 1); // Classification loss
  EvoNetParameters::ModelTrainer::LossFncWeight1 loss_fnc_weight_1("loss_fnc_weight_1", 1); // Reconstruction loss
  EvoNetParameters::ModelTrainer::LossFncWeight2 loss_fnc_weight_2("loss_fnc_weight_2", 0);
  EvoNetParameters::ModelTrainer::LearningRate learning_rate("learning_rate", 1e-5);
  EvoNetParameters::ModelTrainer::GradientClipping gradient_clipping("gradient_clipping", 10);
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
  EvoNetParameters::ModelTrainer::LossFunction loss_function("loss_function", "MSE");
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
    model_type, simulation_type, biochemical_rxns_filename, metabo_data_train_filename, metabo_data_test_filename, meta_data_train_filename, meta_data_test_filename, use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base, offline_linear_scale_input, offline_log_transform_input, offline_standardize_input, online_linear_scale_input, online_log_transform_input, online_standardize_input,
    population_name, n_generations, n_interpreters, /*prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling, set_training_steps_by_model_size,*/
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, n_hidden_0, n_hidden_1, n_hidden_2, loss_fnc_weight_0, loss_fnc_weight_1, loss_fnc_weight_2, learning_rate, gradient_clipping, reset_interpreter, loss_function, KL_divergence_warmup, n_encodings_continuous, n_encodings_categorical, beta, capacity_c, capacity_d/*,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error*/);

    // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = EvoNet::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  EvoNet::apply([](auto&& ...args) { main_(args ...); }, parameters);
  return 0;
}