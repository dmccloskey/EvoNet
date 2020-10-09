/**TODO:  Add copyright*/

#ifndef EVONET_METABOLOMICSCVAE_H
#define EVONET_METABOLOMICSCVAE_H

// .h
#include <EvoNet/simulator/BiochemicalDataSimulator.h>
#include <EvoNet/models/CVAEFullyConn.h>

using namespace EvoNet;

namespace EvoNetMetabolomics
{
  static void makeInputNodes(std::vector<std::string>& input_nodes, const int& n_features) {
    for (int i = 0; i < n_features; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeGaussianEncodingSamplerNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Gaussian_encoding_%012d-Sampler", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeCategoricalEncodingSamplerNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Categorical_encoding_%012d-GumbelSampler", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeCategoricalEncodingTauNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Categorical_encoding_%012d-InverseTau", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeLogAlphaEncodingNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "LogAlpha_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeMuEncodingNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Gaussian_encoding_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  static std::vector<std::string> makeOutputNodes(const int& n_features) {
    std::vector<std::string> output_nodes;
    for (int i = 0; i < n_features; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeMuEncodingNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Mu_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeLogVarEncodingNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "LogVar_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeLogAlphaEncodingNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "LogAlpha_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeCategoricalSoftmaxNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Categorical_encoding-SoftMax-Out_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<typename TensorT, typename TrainerT, typename InterpreterT, typename InterpreterFileT, class ...ParameterTypes>
  static void makeModelAndInterpreters(Model<TensorT>& model, TrainerT& model_trainer, std::vector<InterpreterT>& model_interpreters, InterpreterFileT& model_interpreter_file, const int& n_features, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);

    // define the model interpreters
    setModelInterpreterParameters(model_interpreters, args...);

    // define the model
    if (std::get<EvoNetParameters::Main::MakeModel>(parameters).get()) {
      std::cout << "Making the model..." << std::endl;
      if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "EncDec") {
        model_trainer.makeCVAE(model, n_features,
          std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
      }
      else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Enc") {
        // make the encoder only
        model_trainer.makeCVAEEncoder(model, n_features,
          std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);

        // read in the weights
        ModelFile<TensorT> model_file;
        model_file.loadWeightValuesBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model.weights_);

        // check that all weights were read in correctly
        for (auto& weight_map : model.getWeightsMap()) {
          if (weight_map.second->getInitWeight()) {
            std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
          }
        }
      }
      else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Dec") {
        // make the decoder only
        model_trainer.makeCVAEDecoder(model, n_features,
          std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);

        // read in the weights
        ModelFile<TensorT> model_file;
        model_file.loadWeightValuesBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model.weights_);

        // check that all weights were read in correctly
        for (auto& weight_map : model.getWeightsMap()) {
          if (weight_map.second->getInitWeight()) {
            std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
          }
        }
      }
    }
    else {
      ModelFile<TensorT> model_file;
      loadModelFromParameters(model, model_interpreters.at(0), model_file, model_interpreter_file, args...);
    }
    model.setName(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory
  }
  template<typename TensorT, typename InterpreterT, class ...ParameterTypes>
  static void makeModelTrainer(CVAEFullyConn<TensorT, InterpreterT>& model_trainer, std::vector<std::string>& output_nodes, std::vector<std::string>& encoding_nodes_mu, std::vector<std::string>& encoding_nodes_logvar, std::vector<std::string>& encoding_nodes_logalpha, std::vector<std::string>& categorical_softmax_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    setModelTrainerParameters(model_trainer, args...);

    // CVAE specific parameters and adjustments
    if (std::get<EvoNetParameters::Examples::SimulationType>(parameters).get() == "Train10x") {
      model_trainer.setNEpochsTraining(std::get<EvoNetParameters::ModelTrainer::NEpochsTraining>(parameters).get() * 10 + 1); // iterate through the cache 10x
      model_trainer.setLogging(true, false, false);
    }
    else if (std::get<EvoNetParameters::Examples::SimulationType>(parameters).get() == "LatentTraversal" || std::get<EvoNetParameters::Examples::SimulationType>(parameters).get() == "LatentUnsClass") {
      model_trainer.setLogging(false, true, false);
    }
    model_trainer.KL_divergence_warmup_ = std::get<EvoNetParameters::ModelTrainer::KLDivergenceWarmup>(parameters).get();
    model_trainer.beta_ = std::get<EvoNetParameters::ModelTrainer::Beta>(parameters).get();
    model_trainer.capacity_c_ = std::get<EvoNetParameters::ModelTrainer::CapacityC>(parameters).get();
    model_trainer.capacity_d_ = std::get<EvoNetParameters::ModelTrainer::CapacityD>(parameters).get();
    model_trainer.learning_rate_ = std::get<EvoNetParameters::ModelTrainer::LearningRate>(parameters).get();
    model_trainer.gradient_clipping_ = std::get<EvoNetParameters::ModelTrainer::GradientClipping>(parameters).get();

    // Decide on the reconstruction loss function to use
    std::shared_ptr<LossFunctionOp<TensorT>> loss_function_op;
    std::shared_ptr<LossFunctionGradOp<TensorT>> loss_function_grad_op;
    if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MSE")) {
      loss_function_op = std::make_shared<MSELossOp<TensorT>>(MSELossOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
      loss_function_grad_op = std::make_shared<MSELossGradOp<TensorT>>(MSELossGradOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    }
    else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MAE")) {
      loss_function_op = std::make_shared<MAELossOp<TensorT>>(MAELossOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
      loss_function_grad_op = std::make_shared<MAELossGradOp<TensorT>>(MAELossGradOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    }
    else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MLE")) {
      loss_function_op = std::make_shared<MLELossOp<TensorT>>(MLELossOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
      loss_function_grad_op = std::make_shared<MLELossGradOp<TensorT>>(MLELossGradOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    }
    else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("MAPE")) {
      loss_function_op = std::make_shared<MAPELossOp<TensorT>>(MAPELossOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
      loss_function_grad_op = std::make_shared<MAPELossGradOp<TensorT>>(MAPELossGradOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    }
    else if (std::get<EvoNetParameters::ModelTrainer::LossFunction>(parameters).get() == std::string("BCEWithLogits")) {
      loss_function_op = std::make_shared<BCEWithLogitsLossOp<TensorT>>(BCEWithLogitsLossOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
      loss_function_grad_op = std::make_shared<BCEWithLogitsLossGradOp<TensorT>>(BCEWithLogitsLossGradOp<TensorT>(1e-6, std::get<EvoNetParameters::ModelTrainer::LossFncWeight1>(parameters).get()));
    }

    // Set the loss functions
    std::vector<LossFunctionHelper<TensorT>> loss_function_helpers;
    LossFunctionHelper<TensorT> loss_function_helper1, loss_function_helper2, loss_function_helper3, loss_function_helper4, loss_function_helper5;
    if (output_nodes.size()) {
      loss_function_helper1.output_nodes_ = output_nodes;
      loss_function_helper1.loss_functions_ = { loss_function_op };
      loss_function_helper1.loss_function_grads_ = { loss_function_grad_op };
      loss_function_helpers.push_back(loss_function_helper1);
    }
    if (encoding_nodes_mu.size()) {
      loss_function_helper2.output_nodes_ = encoding_nodes_mu;
      loss_function_helper2.loss_functions_ = { std::make_shared<KLDivergenceMuLossOp<TensorT>>(KLDivergenceMuLossOp<TensorT>(1e-6, 0.0, 0.0)) };
      loss_function_helper2.loss_function_grads_ = { std::make_shared<KLDivergenceMuLossGradOp<TensorT>>(KLDivergenceMuLossGradOp<TensorT>(1e-6, 0.0, 0.0)) };
      loss_function_helpers.push_back(loss_function_helper2);
    }
    if (encoding_nodes_logvar.size()) {
      loss_function_helper3.output_nodes_ = encoding_nodes_logvar;
      loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<TensorT>>(KLDivergenceLogVarLossOp<TensorT>(1e-6, 0.0, 0.0)) };
      loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<TensorT>>(KLDivergenceLogVarLossGradOp<TensorT>(1e-6, 0.0, 0.0)) };
      loss_function_helpers.push_back(loss_function_helper3);
    }
    if (encoding_nodes_logalpha.size()) {
      loss_function_helper4.output_nodes_ = encoding_nodes_logalpha;
      loss_function_helper4.loss_functions_ = { std::make_shared<KLDivergenceCatLossOp<TensorT>>(KLDivergenceCatLossOp<TensorT>(1e-6, 0.0, 0.0)) };
      loss_function_helper4.loss_function_grads_ = { std::make_shared<KLDivergenceCatLossGradOp<TensorT>>(KLDivergenceCatLossGradOp<TensorT>(1e-6, 0.0, 0.0)) };
      loss_function_helpers.push_back(loss_function_helper4);
    }
    if (std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get() > 0 && categorical_softmax_nodes.size()) {
      loss_function_helper5.output_nodes_ = categorical_softmax_nodes;
      loss_function_helper5.loss_functions_ = { std::make_shared<CrossEntropyWithLogitsLossOp<TensorT>>(CrossEntropyWithLogitsLossOp<TensorT>(1e-8, std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get())) };
      loss_function_helper5.loss_function_grads_ = { std::make_shared<CrossEntropyWithLogitsLossGradOp<TensorT>>(CrossEntropyWithLogitsLossGradOp<TensorT>(1e-8, std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get())) };
      loss_function_helpers.push_back(loss_function_helper5);
    }
    model_trainer.setLossFunctionHelpers(loss_function_helpers);

    // Set the metric functions
    std::vector<MetricFunctionHelper<TensorT>> metric_function_helpers;
    MetricFunctionHelper<TensorT> metric_function_helper1, metric_function_helper2;
    if (output_nodes.size()) {
      metric_function_helper1.output_nodes_ = output_nodes;
      metric_function_helper1.metric_functions_ = { 
        //std::make_shared<CosineSimilarityOp<TensorT>>(CosineSimilarityOp<TensorT>("Mean")), std::make_shared<CosineSimilarityOp<TensorT>>(CosineSimilarityOp<TensorT>("Var")),
        //std::make_shared<PearsonROp<TensorT>>(PearsonROp<TensorT>("Mean")), std::make_shared<PearsonROp<TensorT>>(PearsonROp<TensorT>("Var")),
        std::make_shared<EuclideanDistOp<TensorT>>(EuclideanDistOp<TensorT>("Mean")), std::make_shared<EuclideanDistOp<TensorT>>(EuclideanDistOp<TensorT>("Var")),
        //std::make_shared<ManhattanDistOp<TensorT>>(ManhattanDistOp<TensorT>("Mean")), std::make_shared<ManhattanDistOp<TensorT>>(ManhattanDistOp<TensorT>("Var")),
        //std::make_shared<JeffreysAndMatusitaDistOp<TensorT>>(JeffreysAndMatusitaDistOp<TensorT>("Mean")), std::make_shared<JeffreysAndMatusitaDistOp<TensorT>>(JeffreysAndMatusitaDistOp<TensorT>("Var")),
        //std::make_shared<LogarithmicDistOp<TensorT>>(LogarithmicDistOp<TensorT>("Mean")), std::make_shared<LogarithmicDistOp<TensorT>>(LogarithmicDistOp<TensorT>("Var")),
        std::make_shared<PercentDifferenceOp<TensorT>>(PercentDifferenceOp<TensorT>("Mean")), std::make_shared<PercentDifferenceOp<TensorT>>(PercentDifferenceOp<TensorT>("Var")) };
      metric_function_helper1.metric_names_ = { 
        //"CosineSimilarity-Mean", "CosineSimilarity-Var",
        //"PearsonR-Mean", "PearsonR-Var",
        "EuclideanDist-Mean", "EuclideanDist-Var",
        //"ManhattanDist-Mean", "ManhattanDist-Var",
        //"JeffreysAndMatusitaDist-Mean", "JeffreysAndMatusitaDist-Var",
        //"LogarithmicDist-Mean", "LogarithmicDist-Var",
        "PercentDifference-Mean", "PercentDifference-Var" };
      metric_function_helpers.push_back(metric_function_helper1);
    }
    if (std::get<EvoNetParameters::ModelTrainer::LossFncWeight0>(parameters).get() > 0 && categorical_softmax_nodes.size()) {
      metric_function_helper2.output_nodes_ = categorical_softmax_nodes;
      metric_function_helper2.metric_functions_ = { std::make_shared<AccuracyMCMicroOp<TensorT>>(AccuracyMCMicroOp<TensorT>()), std::make_shared<PrecisionMCMicroOp<TensorT>>(PrecisionMCMicroOp<TensorT>()) };
      metric_function_helper2.metric_names_ = { "AccuracyMCMicro", "PrecisionMCMicro" };
      metric_function_helpers.push_back(metric_function_helper2);
    }
    model_trainer.setMetricFunctionHelpers(metric_function_helpers);
  }
  template<typename TensorT, class ...ParameterTypes>
  static int makeDataSimulator(BiochemicalDataSimulator<TensorT>& data_simulator, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);

    // define the data simulator
    data_simulator.n_encodings_continuous_ = std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get();
    data_simulator.n_encodings_discrete_ = std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get();
    int n_reaction_ids_training, n_labels_training, n_component_group_names_training;
    int n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation;
    int n_reps_per_sample = -1;
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

    // Warn about a mismatch in the number of labels and categorical encoding nodes
    if (n_labels_training != std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get()) {
      std::cout << "The number of labels " << n_labels_training << " does not match the number of discrete encodings " << std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get() << std::endl;
      std::cout << "Ensure that classification losses and metric weights are set to 0. " << std::endl;
    }

    // define the model input/output nodes
    int n_features;
    if (std::get<EvoNetParameters::Examples::UseMARs>(parameters).get()) n_features = n_reaction_ids_training;
    else n_features = n_component_group_names_training;
    return n_features;
  }
}
#endif //EVONET_METABOLOMICSCVAE_H