/**TODO:  Add copyright*/

#include <EvoNet/ml/PopulationTrainerDefaultDevice.h>
#include <EvoNet/ml/ModelTrainerDefaultDevice.h>
#include <EvoNet/ml/ModelReplicator.h>
#include <EvoNet/ml/ModelBuilder.h>
#include <EvoNet/io/PopulationTrainerFile.h>
#include <EvoNet/io/ModelInterpreterFileDefaultDevice.h>
#include <EvoNet/simulator/BiochemicalReaction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <EvoNet/io/ModelFile.h>

using namespace EvoNet;

// Other extended classes

template<typename TensorT>
class LatentArithmetic {
public:
  LatentArithmetic(const int& encoding_size, const bool& simulate_MARs, const bool& sample_concs, const bool& use_fold_change, std::string& ref_fold_change) :
    encoding_size_(encoding_size), simulate_MARs_(simulate_MARs), sample_concs_(sample_concs), use_fold_change_(use_fold_change), ref_fold_change_(ref_fold_change) {};
  ~LatentArithmetic() = default;
  /*
  @brief Read in and create the metabolomics data

  @param[in] biochem_rxns_filename
  @param[in] metabo_data_filename_train
  @param[in] meta_data_filename_train
  @param[in] metabo_data_filename_test
  @param[in] meta_data_filename_test
  */
  void setMetabolomicsData(const std::string& biochem_rxns_filename,
    const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
    const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test) {
    // Training data
    reaction_model_.clear();
    reaction_model_.readBiochemicalReactions(biochem_rxns_filename, true);
    reaction_model_.readMetabolomicsData(metabo_data_filename_train);
    reaction_model_.readMetaData(meta_data_filename_train);
    reaction_model_.findComponentGroupNames();
    if (simulate_MARs_) {
      reaction_model_.findMARs();
      reaction_model_.findMARs(true, false);
      reaction_model_.findMARs(false, true);
      reaction_model_.removeRedundantMARs();
    }
    reaction_model_.findLabels();
    metabolomics_data_.model_training_ = reaction_model_;

    // Validation data
    reaction_model_.clear();
    reaction_model_.readBiochemicalReactions(biochem_rxns_filename, true);
    reaction_model_.readMetabolomicsData(metabo_data_filename_test);
    reaction_model_.readMetaData(meta_data_filename_test);
    reaction_model_.findComponentGroupNames();
    if (simulate_MARs_) {
      reaction_model_.findMARs();
      reaction_model_.findMARs(true, false);
      reaction_model_.findMARs(false, true);
      reaction_model_.removeRedundantMARs();
    }
    reaction_model_.findLabels();
    metabolomics_data_.model_validation_ = reaction_model_;
    metabolomics_data_.simulate_MARs_ = simulate_MARs_;
    metabolomics_data_.sample_concs_ = sample_concs_;
    metabolomics_data_.use_train_ = true;
    metabolomics_data_.use_fold_change_ = this->use_fold_change_;
    metabolomics_data_.ref_fold_change_ = this->ref_fold_change_;

    // Checks for the training and validation data
    assert(metabolomics_data_.model_validation_.reaction_ids_.size() == metabolomics_data_.model_training_.reaction_ids_.size());
    assert(metabolomics_data_.model_validation_.labels_.size() == metabolomics_data_.model_training_.labels_.size());
    assert(metabolomics_data_.model_validation_.component_group_names_.size() == metabolomics_data_.model_training_.component_group_names_.size());

    // Set the encoding size and define the input/output sizes
    metabolomics_data_.n_encodings_ = encoding_size_;
    if (simulate_MARs_) n_input_nodes_ = reaction_model_.reaction_ids_.size();
    else n_input_nodes_ = reaction_model_.component_group_names_.size();
    n_output_nodes_ = reaction_model_.labels_.size();
  };

  /*
  @brief Make the encoder/decoder models

  @param[in] model_encoder_weights_filename
  @param[in] model_decoder_weights_filename
  */
  void setEncDecModels(ModelTrainerExt<TensorT>& model_trainer, const std::string& model_encoder_weights_filename, const std::string& model_decoder_weights_filename,
    const int& n_en_hidden_0 = 64, const int& n_en_hidden_1 = 64, const int& n_en_hidden_2 = 0,
    const int& n_de_hidden_0 = 64, const int& n_de_hidden_1 = 64, const int& n_de_hidden_2 = 0) {
    // initialize the models
    model_encoder_.clear();
    model_decoder_.clear();

    // define the encoder and decoders
    model_trainer.makeModelFCVAE_Encoder(model_encoder_, n_input_nodes_, encoding_size_, true, false, false, false,
      n_en_hidden_0, n_en_hidden_1, n_en_hidden_2, this->use_fold_change_); // normalization type 1
    model_trainer.makeModelFCVAE_Decoder(model_decoder_, n_input_nodes_, encoding_size_, false,
      n_de_hidden_0, n_de_hidden_1, n_de_hidden_2, this->use_fold_change_);

    // read in the encoder and decoder weights
    WeightFile<TensorT> data;
    data.loadWeightValuesCsv(model_encoder_weights_filename, model_encoder_.getWeightsMap());
    data.loadWeightValuesCsv(model_decoder_weights_filename, model_decoder_.getWeightsMap());

    // check that all weights were read in correctly
    for (auto& weight_map : model_encoder_.getWeightsMap()) {
      if (weight_map.second->getInitWeight()) {
        std::cout << "Model " << model_encoder_.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
      }
    }
    for (auto& weight_map : model_decoder_.getWeightsMap()) {
      if (weight_map.second->getInitWeight()) {
        std::cout << "Model " << model_decoder_.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
      }
    }
  };

  /*
  @brief Set the encoder and decoder model interpreters

  @param[in] model_interpreter_encoder_
  @param[in] model_interpreter_decoder_
  */
  void setEncDecModelInterpreters(const ModelInterpreterDefaultDevice<TensorT>& model_interpreter_encoder, const ModelInterpreterDefaultDevice<TensorT>& model_interpreter_decoder) {
    model_interpreter_encoder_ = model_interpreter_encoder;
    model_interpreter_decoder_ = model_interpreter_decoder;
  };

  /*
  @brief Generate an encoded latent space

  @param[in] sample_group_name

  @returns 4D Tensor of the encoded latent space
  */
  Eigen::Tensor<TensorT, 4> generateEncoding(
    const std::string& sample_group_name, 
    ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger)
  {
    // Make the input nodes
    std::vector<std::string> input_nodes;
    for (int i = 0; i < this->n_input_nodes_; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }

    // Make the mu nodes
    std::vector<std::string> encoding_nodes_mu;
    for (int i = 0; i < this->encoding_size_; ++i) {
      char name_char[512];
      sprintf(name_char, "Mu_%012d", i);
      std::string name(name_char);
      encoding_nodes_mu.push_back(name);
    }

    // generate the input for condition_1 and condition_2
    Eigen::Tensor<TensorT, 4> condition_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    this->metabolomics_data_.sample_group_name_ = sample_group_name;
    this->metabolomics_data_.simulateEvaluationData(condition_1_input, time_steps_1_input);;

    // evaluate the encoder for condition_1 and condition_2
    model_trainer.setLossOutputNodes({ encoding_nodes_mu });
    Eigen::Tensor<TensorT, 4> condition_1_output = model_trainer.evaluateModel(
      this->model_encoder_, condition_1_input, time_steps_1_input, input_nodes, model_logger, this->model_interpreter_encoder_);

    return condition_1_output;
  }

  /*
  @brief Generate a reconstruction of the latent space

  @param[in] encoding_output 4D Tensor of the encoded latent space

  @returns 4D Tensor of the reconstruction
  */
  Eigen::Tensor<TensorT, 4> generateReconstruction(
    Eigen::Tensor<TensorT, 4>& encoding_output,
    ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger)
  {
    // Make the reconstruction nodes
    std::vector<std::string> output_nodes_reconstruction;
    for (int i = 0; i < this->n_input_nodes_; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes_reconstruction.push_back(name);
    }

    // Make the encoding nodes
    std::vector<std::string> encoding_nodes;
    for (int i = 0; i < this->encoding_size_; ++i) {
      char name_char[512];
      sprintf(name_char, "Encoding_%012d", i);
      std::string name(name_char);
      encoding_nodes.push_back(name);
    }

    // evaluate the decoder
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    model_trainer.setLossOutputNodes({ output_nodes_reconstruction });
    Eigen::Tensor<TensorT, 4> reconstructed_output = model_trainer.evaluateModel(
      this->model_decoder_, encoding_output, time_steps_1_input, encoding_nodes, model_logger, this->model_interpreter_decoder_);

    return reconstructed_output;
  }

  /*
  @brief Score a reconstruction or input using a similarity metric function

  @param[in] sample_group_name_expected
  */
  std::pair<TensorT,TensorT> scoreReconstructionSimilarity(const std::string& sample_group_name_expected, const Eigen::Tensor<TensorT, 4>& reconstructed_output,
    MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function, ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger) {
    // Make the input nodes
    std::vector<std::string> input_nodes;
    for (int i = 0; i < this->n_input_nodes_; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }

    // Make the classification nodes
    std::vector<std::string> output_nodes_normalization;
    for (int i = 0; i < this->n_input_nodes_; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes_normalization.push_back(name);
    }

    // generate the input for the expected
    Eigen::Tensor<TensorT, 4> condition_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    this->metabolomics_data_.sample_group_name_ = sample_group_name_expected;
    this->metabolomics_data_.simulateEvaluationData(condition_1_input, time_steps_1_input);

    // score the decoded data using the classification model
    // TODO: currently just ignoring the `is_neg` nodes in the score
    Eigen::Tensor<TensorT, 3> expected = condition_1_input.chip(0, 1);
    Eigen::Tensor<TensorT, 2> predicted = reconstructed_output.chip(0, 3).chip(0, 1);
    //Eigen::Tensor<TensorT, 2> predicted = normalization_reconstruction.chip(0, 3).chip(0, 1);
    Eigen::Tensor<TensorT, 2> score_mean(1, 1); score_mean.setZero();
    Eigen::Tensor<TensorT, 2> score_var(1, 1); score_var.setZero();
    Eigen::DefaultDevice device;
    metric_function.setReductionFunc(std::string("Mean"));
    metric_function(predicted.data(), expected.data(), score_mean.data(), model_trainer.getBatchSize(), model_trainer.getMemorySize(), this->n_input_nodes_, 1, 0, 0, device);
    metric_function.setReductionFunc(std::string("Var"));
    metric_function(predicted.data(), expected.data(), score_var.data(), model_trainer.getBatchSize(), model_trainer.getMemorySize(), this->n_input_nodes_, 1, 0, 0, device);
    return std::make_pair(score_mean(0, 0), score_var(0, 0));
  };

  /*
  @brief Score the similarity between data sets using a similarity metric function

  TODO: refactor to allow for using the GPU

  @param[in] sample_group_name_expected
  */
  std::pair<TensorT,TensorT> scoreDataSimilarity(const std::string& sample_group_name_expected, const std::string& sample_group_name_predicted,
    MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function, ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger) {
    // Make the input nodes
    std::vector<std::string> input_nodes;
    for (int i = 0; i < this->n_input_nodes_; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }

    // Make the classification nodes
    std::vector<std::string> output_nodes_normalization;
    for (int i = 0; i < this->n_input_nodes_; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes_normalization.push_back(name);
    }

    // generate the input for the expected
    Eigen::Tensor<TensorT, 4> condition_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    this->metabolomics_data_.sample_group_name_ = sample_group_name_expected;
    this->metabolomics_data_.simulateEvaluationData(condition_1_input, time_steps_1_input);

    // normalize the expected
    model_trainer.setLossOutputNodes({ output_nodes_normalization });
    Eigen::Tensor<TensorT, 4> normalization_output_1 = model_trainer.evaluateModel(
      this->model_normalization_, condition_1_input, time_steps_1_input, input_nodes, model_logger, this->model_interpreter_normalization_);

    // generate the input for the predicted
    Eigen::Tensor<TensorT, 4> condition_2_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_2_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    this->metabolomics_data_.sample_group_name_ = sample_group_name_predicted;
    this->metabolomics_data_.simulateEvaluationData(condition_2_input, time_steps_2_input);

    // normalize the expected
    model_trainer.setLossOutputNodes({ output_nodes_normalization });
    Eigen::Tensor<TensorT, 4> normalization_output_2 = model_trainer.evaluateModel(
      this->model_normalization_, condition_2_input, time_steps_2_input, input_nodes, model_logger, this->model_interpreter_normalization_);

    // score the decoded data using the classification model
    // TODO: currently just ignoring the `is_neg` nodes in the score
    Eigen::Tensor<TensorT, 3> expected = normalization_output_1.chip(0, 1);
    Eigen::Tensor<TensorT, 2> predicted = normalization_output_2.chip(0, 3).chip(0, 1);
    Eigen::Tensor<TensorT, 2> score_mean(1, 1); score_mean.setZero();
    Eigen::Tensor<TensorT, 2> score_var(1, 1); score_var.setZero();
    Eigen::DefaultDevice device;
    metric_function.setReductionFunc(std::string("Mean"));
    metric_function(predicted.data(), expected.data(), score_mean.data(), model_trainer.getBatchSize(), model_trainer.getMemorySize(), this->n_input_nodes_, 1, 0, 0, device);
    metric_function.setReductionFunc(std::string("Var"));
    metric_function(predicted.data(), expected.data(), score_var.data(), model_trainer.getBatchSize(), model_trainer.getMemorySize(), this->n_input_nodes_, 1, 0, 0, device);
    return std::make_pair(score_mean(0,0), score_var(0,0));
  };

  int getNInputNodes() const { return n_input_nodes_; }
  int getNOutputNodes() const { return n_output_nodes_; }

  int encoding_size_ = 16;
  bool simulate_MARs_ = false;
  bool sample_concs_ = true;
  bool use_fold_change_ = false;
  std::string ref_fold_change_ = "";

protected:
  /// Defined in setMetabolomicsData
  MetDataSimLatentArithmetic<TensorT> metabolomics_data_;
  BiochemicalReactionModel<TensorT> reaction_model_;
  int n_input_nodes_ = -1;
  int n_output_nodes_ = -1; 

  /// Defined in setEncDecModels
  Model<TensorT> model_decoder_;
  Model<TensorT> model_encoder_;
  ModelInterpreterDefaultDevice<TensorT> model_interpreter_decoder_;
  ModelInterpreterDefaultDevice<TensorT> model_interpreter_encoder_;
};

/*
@brief Compute the similarity between data sets

TODO: refactor to allow for running on the GPU
*/
template<typename TensorT>
void computeDataSimilarity(const std::vector<std::string>& predicted, const std::vector<std::string>& expected, LatentArithmetic<TensorT>& latentArithmetic, MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function, ModelTrainerExt<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger,
  const bool& init_interpreter) {
  assert(predicted.size() == expected.size());
  for (int case_iter = 0; case_iter < predicted.size(); ++case_iter) {
    //// Determine when to initialize the model interpreter
    //if (case_iter == 0 && init_interpreter) {
    //  model_trainer.setInterpretModel(true);
    //}
    //else {
    //  model_trainer.setInterpretModel(false);
    //}
    //model_trainer.setResetModel(false);
    //model_trainer.setResetInterpreter(false);

    // Calculate the similarity
    std::pair<TensorT, TensorT> score = latentArithmetic.scoreDataSimilarity(expected.at(case_iter), predicted.at(case_iter), metric_function, model_trainer, model_logger);
    std::cout << expected.at(case_iter) << " -> " << predicted.at(case_iter) << ": " << score.first << " +/- " << score.second << std::endl;
  }
}

/*
@brief Compute the similarity between the data and the generated data

TODO: refactor to allow for running on the GPU
*/
template<typename TensorT>
void computeGenerationSimilarity(const std::vector<std::string>& predicted, const std::vector<std::string>& expected, LatentArithmetic<TensorT>& latentArithmetic, MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function, ModelTrainerExt<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger,
  const bool& init_interpreter) {
  assert(predicted.size() == expected.size());

  //// define the reconstruction output
  //Eigen::Tensor<TensorT, 4> reconstruction_output(model_trainer.getBatchSize(), model_trainer.getMemorySize(), latentArithmetic.getNInputNodes(), model_trainer.getNEpochsEvaluation());

  for (int case_iter = 0; case_iter < predicted.size(); ++case_iter) {
    //// Determine when to initialize the model interpreter
    //if (case_iter == 0 && init_interpreter) {
    //  model_trainer.setInterpretModel(true);
    //}
    //else {
    //  model_trainer.setInterpretModel(false);
    //}
    //model_trainer.setResetModel(false);
    //model_trainer.setResetInterpreter(false);

    // Generate the encoding and decoding and score the result
    auto encoding_output = latentArithmetic.generateEncoding(predicted.at(case_iter), model_trainer, model_logger);
    auto reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger);
    std::pair<TensorT, TensorT> score = latentArithmetic.scoreReconstructionSimilarity(expected.at(case_iter), reconstruction_output, metric_function, model_trainer, model_logger);
    std::cout << predicted.at(case_iter) << " -> " << expected.at(case_iter) << ": " << score.first << " +/- " << score.second << std::endl;
  }
}

/*
@brief Compute the similarity between the data and the generated data after a latent arithmetic operation

TODO: refactor to allow for running on the GPU
*/
template<typename TensorT>
void computeLatentArithmeticSimilarity(const std::vector<std::string>& condition_1, const std::vector<std::string>& condition_2, 
  const std::vector<std::string>& expected, LatentArithmetic<TensorT>& latentArithmetic,
  MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function, ModelTrainerExt<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger,
  const bool& init_interpreter, const std::string& latent_operation) {
  assert(condition_1.size() == condition_2.size());
  assert(expected.size() == condition_2.size());
  assert(condition_1.size() == expected.size());

  for (int case_iter = 0; case_iter < condition_1.size(); ++case_iter) {
    //// Determine when to initialize the model interpreter
    //if (case_iter == 0) {
    //  model_trainer.setInterpretModel(true);
    //}
    //else {
    //  model_trainer.setInterpretModel(false);
    //}
    //model_trainer.setResetModel(false);
    //model_trainer.setResetInterpreter(false);

    // define the reconstruction output
    Eigen::Tensor<TensorT, 4> reconstruction_output(model_trainer.getBatchSize(), model_trainer.getMemorySize(), latentArithmetic.getNInputNodes(), model_trainer.getNEpochsEvaluation());

    // Calculate the latent arithmetic
    auto encoding_output_1 = latentArithmetic.generateEncoding(condition_1.at(case_iter), model_trainer, model_logger);
    auto encoding_output_2 = latentArithmetic.generateEncoding(condition_2.at(case_iter), model_trainer, model_logger);
    if (latent_operation == "-") {
      Eigen::Tensor<TensorT, 4> encoding_output = encoding_output_1 - encoding_output_2;
      reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger);
    }
    else if (latent_operation == "+") {
      Eigen::Tensor<TensorT, 4> encoding_output = encoding_output_1 + encoding_output_2;
      reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger);
    }

    // Score the reconstruction similarity to the expected
    std::pair<TensorT, TensorT> score = latentArithmetic.scoreReconstructionSimilarity(expected.at(case_iter), reconstruction_output, metric_function, model_trainer, model_logger);
    std::cout << condition_1.at(case_iter) << " " << latent_operation << " " << condition_2.at(case_iter) << " -> " << expected.at(case_iter) << ": " << score.first << " +/- " << score.second << std::endl;
  }
}

/*
@brief Compute the similarity between the data and the generated data after a latent interpolation

TODO: refactor to allow for running on the GPU
*/
template<typename TensorT>
void computeLatentInterpolationSimilarity(const std::vector<std::string>& condition_1, const std::vector<std::string>& condition_2,
  const std::vector<std::vector<std::string>>& expected, LatentArithmetic<TensorT>& latentArithmetic,
  MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function, ModelTrainerExt<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger,
  const bool& init_interpreter, const bool& interp_q1, const bool& interp_median, const bool& interp_q3) {
  assert(condition_1.size() == condition_2.size());
  assert(expected.size() == condition_2.size());
  assert(condition_1.size() == expected.size());

  //// define the reconstruction output (is this leading to data corruption?)
  //Eigen::Tensor<TensorT, 4> reconstruction_output(model_trainer.getBatchSize(), model_trainer.getMemorySize(), latentArithmetic.getNInputNodes(), model_trainer.getNEpochsEvaluation());

  for (int case_iter = 0; case_iter < condition_1.size(); ++case_iter) {

    // Determine when to initialize the model interpreter
    //if (case_iter == 0) {
    //  model_trainer.setInterpretModel(true);
    //}
    //else {
    //  model_trainer.setInterpretModel(false);
    //}    
    //model_trainer.setResetModel(false);
    //model_trainer.setResetInterpreter(false);

    // Generate the encodings
    auto encoding_output_1 = latentArithmetic.generateEncoding(condition_1.at(case_iter), model_trainer, model_logger);
    auto encoding_output_2 = latentArithmetic.generateEncoding(condition_2.at(case_iter), model_trainer, model_logger);

    // Interpolations
    if (interp_q1) {
      Eigen::Tensor<TensorT, 4> encoding_output = encoding_output_1 * encoding_output_1.constant(0.25) + encoding_output_2 * encoding_output_2.constant(0.75);
      auto reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger);
      for (int trial_iter = 0; trial_iter < expected.at(case_iter).size(); ++trial_iter) {
        std::pair<TensorT, TensorT> score = latentArithmetic.scoreReconstructionSimilarity(expected.at(case_iter).at(trial_iter), reconstruction_output, metric_function, model_trainer, model_logger);
        std::cout << "0.25 * " << condition_1.at(case_iter) << " + 0.75 * " << condition_2.at(case_iter) << " -> " << expected.at(case_iter).at(trial_iter) << ": " << score.first << " +/- " << score.second << std::endl;
      }
    }
    if (interp_median) {
      Eigen::Tensor<TensorT, 4> encoding_output = encoding_output_1 * encoding_output_1.constant(0.5) + encoding_output_2 * encoding_output_2.constant(0.5);
      auto reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger);
      for (int trial_iter = 0; trial_iter < expected.at(case_iter).size(); ++trial_iter) {
        std::pair<TensorT, TensorT> score = latentArithmetic.scoreReconstructionSimilarity(expected.at(case_iter).at(trial_iter), reconstruction_output, metric_function, model_trainer, model_logger);
        std::cout << "0.5 * " << condition_1.at(case_iter) << " + 0.5 * " << condition_2.at(case_iter) << " -> " << expected.at(case_iter).at(trial_iter) << ": " << score.first << " +/- " << score.second << std::endl;
      }
    }
    if (interp_q3) {
      Eigen::Tensor<TensorT, 4> encoding_output = encoding_output_1 * encoding_output_1.constant(0.75) + encoding_output_2 * encoding_output_2.constant(0.25);
      auto reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger);
      for (int trial_iter = 0; trial_iter < expected.at(case_iter).size(); ++trial_iter) {
        std::pair<TensorT, TensorT> score = latentArithmetic.scoreReconstructionSimilarity(expected.at(case_iter).at(trial_iter), reconstruction_output, metric_function, model_trainer, model_logger);
        std::cout << "0.75 * " << condition_1.at(case_iter) << " + 0.25 * " << condition_2.at(case_iter) << " -> " << expected.at(case_iter).at(trial_iter) << ": " << score.first << " +/- " << score.second << std::endl;
      }
    }
  }
}

/// KALE Latent arithmetic and interpolation script
template<typename TensorT>
void main_KALE(ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelTrainerExt<TensorT>& model_trainer,
  ModelLogger<TensorT>& model_logger, LatentArithmetic<TensorT>& latentArithmetic, MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function,
  const bool& compute_data_similarities,  const bool& compute_generation_similarity,
  const bool& compute_latent_arithmetic,  const bool& compute_latent_interpolation) {

  // NOTE: similarity metric of Manhattan distance used as per 10.1109/TCBB.2016.2586065
  //  that found the following similarity metrics to work well for metabolomic prfile data:
  //    Minkowski distance, Euclidean distance, Manhattan distance, Jeffreys & Matusita distance, Dice’s coefficient, Jaccard similarity coefficient
  //  and the following similarity metrics to be unsuitable for metabolomic profile data:
  //    Canberra distance, relative distance, and cosine of angle
  std::vector<std::string> condition_1, condition_2, predicted, expected;

  if (compute_data_similarities) {
    // Reference similarity metrics
    //predicted = { "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
    //  "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
    //expected = { "Evo04", "Evo04", "Evo04gnd", "Evo04gnd", "Evo04pgi", "Evo04pgi",
    //  "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04sdhCB", "Evo04sdhCB", "Evo04tpiA", "Evo04tpiA" };
    //computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
    //  true);

    //predicted = std::vector<std::string>({ "Evo04gnd", "Evo04pgi", "Evo04ptsHIcrr", "Evo04sdhCB", "Evo04tpiA" });
    //expected = std::vector<std::string>({ "Evo04", "Evo04", "Evo04", "Evo04", "Evo04" });
    //computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
    //  true);

    //predicted = std::vector<std::string>({ "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
    //  "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" });
    //expected = std::vector<std::string>({ "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04",
    //  "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04" });
    //computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
    //  true);

    predicted = std::vector<std::string>({ "Evo04", "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" });
    expected = std::vector<std::string>({ "Evo04", "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" });
    computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true);
  }

  if (compute_generation_similarity) {
    predicted = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
    expected = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
    computeGenerationSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true);
  }

  if (compute_latent_arithmetic) {
    // 1. EPi - KO -> Ref
    condition_1 = { "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
    condition_2 = { "Evo04gnd", "Evo04gnd", "Evo04pgi", "Evo04pgi",
      "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04sdhCB", "Evo04sdhCB", "Evo04tpiA", "Evo04tpiA" };
    expected = { "Evo04", "Evo04", "Evo04", "Evo04",
     "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "-");

    // 2. EPi - Ref -> KO
    condition_1 = { "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
    condition_2 = { "Evo04", "Evo04", "Evo04", "Evo04",
      "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04" };
    expected = { "Evo04gnd", "Evo04gnd", "Evo04pgi", "Evo04pgi",
      "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04sdhCB", "Evo04sdhCB", "Evo04tpiA", "Evo04tpiA" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "-");

    // 3. KOi + Ref -> EPi
    condition_1 = { "Evo04gnd", "Evo04gnd", "Evo04pgi", "Evo04pgi",
      "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04sdhCB", "Evo04sdhCB", "Evo04tpiA", "Evo04tpiA" };
    condition_2 = { "Evo04", "Evo04", "Evo04", "Evo04",
      "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04" };
    expected = { "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "+");
  }

  if (compute_latent_interpolation) {
    const std::vector<std::string> condition_1 = { 
      //"Evo04gnd", "Evo04gnd", 
      "Evo04pgi", "Evo04pgi",
      "Evo04ptsHIcrr", "Evo04ptsHIcrr", 
      //"Evo04sdhCB", "Evo04sdhCB", 
      "Evo04tpiA", "Evo04tpiA" };
    const std::vector<std::string> condition_2 = { 
      //"Evo04gndEvo01EP", "Evo04gndEvo02EP", 
      "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
      "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", 
      //"Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", 
      "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
    const std::vector<std::vector<std::string>> expected = {
      //{"Evo04gnd", "Evo04gndEvo01EP"}, {"Evo04gnd", "Evo04gndEvo02EP"},
      // Pgi: J01 and J02 for Evo01, J01, J02, and J03 for all others
      {"Evo04pgi", "Evo04pgiEvo01J01", "Evo04pgiEvo01J02", "Evo04pgiEvo01EP"}, {"Evo04pgi", "Evo04pgiEvo02J01", "Evo04pgiEvo02J02", "Evo04pgiEvo02J03", "Evo04pgiEvo02EP"},
      // PtsHIcrr: J01 and J03 for Evo01 and EVo02, J01, J03, J04 for Evo03 and Ev04
      {"Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01J01", "Evo04ptsHIcrrEvo01J03", "Evo04ptsHIcrrEvo01EP"}, {"Evo04ptsHIcrr", "Evo04ptsHIcrrEvo02J01", "Evo04ptsHIcrrEvo02J03", "Evo04ptsHIcrrEvo02EP"},
      //{"Evo04sdhCB", "Evo04sdhCBEvo01EP"}, {"Evo04sdhCB", "Evo04sdhCBEvo02EP"},
      // TpiA: J01 and J03 for all
      {"Evo04tpiA", "Evo04tpiAEvo01J01", "Evo04tpiAEvo01J03", "Evo04tpiAEvo01EP"}, {"Evo04tpiA", "Evo04tpiAEvo02J01", "Evo04tpiAEvo02J03", "Evo04tpiAEvo02EP"} };
    computeLatentInterpolationSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, true, true, true);
  }
}

/// Industrial strain reconstruction script
template<typename TensorT>
void main_IndustrialStrains(ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelTrainerExt<TensorT>& model_trainer,
  ModelLogger<TensorT>& model_logger, LatentArithmetic<TensorT>& latentArithmetic, MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function,
  const bool& compute_data_similarities, const bool& compute_generation_similarity) {

  // NOTE: similarity metric of Manhattan distance used as per 10.1109/TCBB.2016.2586065
  //  that found the following similarity metrics to work well for metabolomic prfile data:
  //    Minkowski distance, Euclidean distance, Manhattan distance, Jeffreys & Matusita distance, Dice’s coefficient, Jaccard similarity coefficient
  //  and the following similarity metrics to be unsuitable for metabolomic profile data:
  //    Canberra distance, relative distance, and cosine of angle
  std::vector<std::string> condition_1, condition_2, predicted, expected;

  if (compute_data_similarities) {
    // Reference similarity metrics
    predicted = { "EColi_BL21","EColi_C","EColi_Crooks","EColi_DH5a","EColi_MG1655","EColi_W","EColi_W3110" };
    expected = { "EColi_BL21","EColi_C","EColi_Crooks","EColi_DH5a","EColi_MG1655","EColi_W","EColi_W3110" };
    computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true);
  }

  if (compute_generation_similarity) {
    predicted = { "EColi_BL21","EColi_C","EColi_Crooks","EColi_DH5a","EColi_MG1655","EColi_W","EColi_W3110" };
    expected = { "EColi_BL21","EColi_C","EColi_Crooks","EColi_DH5a","EColi_MG1655","EColi_W","EColi_W3110" };
    computeGenerationSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true);
  }
}

/// PLT time-course Latent arithmetic and interpolation script
template<typename TensorT>
void main_PLT(ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelTrainerExt<TensorT>& model_trainer,
  ModelLogger<TensorT>& model_logger, LatentArithmetic<TensorT>& latentArithmetic, MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function,
  const bool& compute_data_similarities, const bool& compute_generation_similarity,
  const bool& compute_latent_arithmetic, const bool& compute_latent_interpolation) {

  // NOTE: similarity metric of Manhattan distance used as per 10.1109/TCBB.2016.2586065
  //  that found the following similarity metrics to work well for metabolomic prfile data:
  //    Minkowski distance, Euclidean distance, Manhattan distance, Jeffreys & Matusita distance, Dice’s coefficient, Jaccard similarity coefficient
  //  and the following similarity metrics to be unsuitable for metabolomic profile data:
  //    Canberra distance, relative distance, and cosine of angle
  std::vector<std::string> condition_1, condition_2, predicted, expected;

  if (compute_data_similarities) {
    // Reference similarity metrics
    predicted = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_0hr",
                  "S01_D02_PLT_37C_22hr", "S01_D02_PLT_25C_22hr", "S01_D02_PLT_25C_6.5hr", "S01_D02_PLT_25C_0hr",
                  "S01_D05_PLT_37C_22hr", "S01_D05_PLT_25C_22hr", "S01_D05_PLT_25C_6.5hr", "S01_D05_PLT_25C_0hr" };
    expected = { "S01_D01_PLT_25C_0hr", "S01_D01_PLT_25C_0hr", "S01_D01_PLT_25C_0hr", "S01_D01_PLT_25C_0hr", 
      "S01_D02_PLT_25C_0hr", "S01_D02_PLT_25C_0hr","S01_D02_PLT_25C_0hr","S01_D02_PLT_25C_0hr",
      "S01_D05_PLT_25C_0hr", "S01_D05_PLT_25C_0hr", "S01_D05_PLT_25C_0hr", "S01_D05_PLT_25C_0hr" };
    computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true);

    predicted = { "S01_D02_PLT_37C_22hr", "S01_D02_PLT_25C_22hr", "S01_D02_PLT_25C_6.5hr", "S01_D02_PLT_25C_0hr",
                  "S01_D05_PLT_37C_22hr", "S01_D05_PLT_25C_22hr", "S01_D05_PLT_25C_6.5hr", "S01_D05_PLT_25C_0hr" };
    expected = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_0hr",
                  "S01_D01_PLT_37C_22hr", "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_0hr" };
    computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      false);

    predicted = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_0hr",
                  "S01_D02_PLT_37C_22hr", "S01_D02_PLT_25C_22hr", "S01_D02_PLT_25C_6.5hr", "S01_D02_PLT_25C_0hr",
                  "S01_D05_PLT_37C_22hr", "S01_D05_PLT_25C_22hr", "S01_D05_PLT_25C_6.5hr", "S01_D05_PLT_25C_0hr" };
    expected = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_0hr",
                  "S01_D02_PLT_37C_22hr", "S01_D02_PLT_25C_22hr", "S01_D02_PLT_25C_6.5hr", "S01_D02_PLT_25C_0hr",
                  "S01_D05_PLT_37C_22hr", "S01_D05_PLT_25C_22hr", "S01_D05_PLT_25C_6.5hr", "S01_D05_PLT_25C_0hr" };
    computeDataSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      false);
  }

  if (compute_generation_similarity) {
    predicted = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_0hr",
                  "S01_D02_PLT_37C_22hr", "S01_D02_PLT_25C_22hr", "S01_D02_PLT_25C_6.5hr", "S01_D02_PLT_25C_0hr",
                  "S01_D05_PLT_37C_22hr", "S01_D05_PLT_25C_22hr", "S01_D05_PLT_25C_6.5hr", "S01_D05_PLT_25C_0hr" };
    expected = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_0hr",
                  "S01_D02_PLT_37C_22hr", "S01_D02_PLT_25C_22hr", "S01_D02_PLT_25C_6.5hr", "S01_D02_PLT_25C_0hr",
                  "S01_D05_PLT_37C_22hr", "S01_D05_PLT_25C_22hr", "S01_D05_PLT_25C_6.5hr", "S01_D05_PLT_25C_0hr" };
    computeGenerationSimilarity(predicted, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true);
  }

  if (compute_latent_arithmetic) {
    // 1. drug + degradation -> drug & degradation 
    condition_1 = { "S01_D02_PLT_25C_0hr", "S01_D05_PLT_25C_0hr" };
    condition_2 = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_37C_22hr" };
    expected = { "S01_D02_PLT_37C_22hr", "S01_D05_PLT_37C_22hr" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "+");

    // 2. drug + metabolic -> drug & metabolic
    condition_1 = { "S01_D02_PLT_25C_0hr", "S01_D05_PLT_25C_0hr" };
    condition_2 = { "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_22hr" };
    expected = { "S01_D02_PLT_25C_22hr", "S01_D05_PLT_25C_22hr" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "+");

    // 3. drug & metabolic - metabolic -> drug
    condition_1 = { "S01_D02_PLT_25C_22hr", "S01_D05_PLT_25C_22hr" };
    condition_2 = { "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_22hr" };
    expected = { "S01_D02_PLT_25C_0hr", "S01_D05_PLT_25C_0hr" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "-");

    // 4. drug & metabolic - drug -> metabolic
    condition_1 = { "S01_D02_PLT_25C_22hr", "S01_D05_PLT_25C_22hr" };
    condition_2 = { "S01_D02_PLT_25C_0hr", "S01_D05_PLT_25C_0hr" };
    expected = { "S01_D01_PLT_25C_22hr", "S01_D01_PLT_25C_22hr" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "-");

    // 5. drug & degradation - degradation -> drug
    condition_1 = { "S01_D02_PLT_37C_22hr", "S01_D05_PLT_37C_22hr" };
    condition_2 = { "S01_D01_PLT_37C_22hr", "S01_D01_PLT_37C_22hr" };
    expected = { "S01_D02_PLT_25C_0hr", "S01_D05_PLT_25C_0hr" };
    computeLatentArithmeticSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, "-");
  }

  if (compute_latent_interpolation) {
    condition_1 = { "S01_D01_PLT_25C_0hr", "S01_D02_PLT_25C_0hr", "S01_D05_PLT_25C_0hr" };
    condition_2 = { "S01_D01_PLT_25C_22hr", "S01_D02_PLT_25C_22hr", "S01_D05_PLT_25C_22hr" };
    const std::vector<std::vector<std::string>> expected = {
      {"S01_D01_PLT_25C_0hr", "S01_D01_PLT_25C_2hr", "S01_D01_PLT_25C_6.5hr", "S01_D01_PLT_25C_22hr"},
      {"S01_D02_PLT_25C_0hr", "S01_D02_PLT_25C_2hr", "S01_D02_PLT_25C_6.5hr", "S01_D02_PLT_25C_22hr"},
      {"S01_D05_PLT_25C_0hr", "S01_D05_PLT_25C_2hr", "S01_D05_PLT_25C_6.5hr", "S01_D05_PLT_25C_22hr"},
    };
    computeLatentInterpolationSimilarity(condition_1, condition_2, expected, latentArithmetic, metric_function, model_trainer, model_logger,
      true, true, true, true);
  }
}

// Main
int main(int argc, char** argv)
{
  /// KALE and Industrial strains

  //// Set the data directories
  //const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "/home/user/Data/";

  // Set the biochemical reaction filenames
  const std::string biochem_rxns_filename = data_dir + "iJO1366.csv";

  // Set the model filenames
  const std::string model_encoder_weights_filename = data_dir + "TrainTestData/SampledArithmeticMath/VAE_weights.csv";
  const std::string model_decoder_weights_filename = data_dir + "TrainTestData/SampledArithmeticMath/VAE_weights.csv";
  // NOTE: be sure to re-name the Input_000000000000-LinearScale_to_... weights to Input_000000000000_to_...
  //       using regex "-LinearScale_to_FC0" with "_to_FC0"
  const std::string model_classifier_weights_filename = data_dir + "TrainTestData/SampledArithmeticMath/Classifier_5000_weights.csv";

  // ALEsKOs01
  const std::string metabo_data_filename_train = data_dir + "ALEsKOs01_Metabolomics_train.csv";
  const std::string meta_data_filename_train = data_dir + "ALEsKOs01_MetaData_train.csv";
  const std::string metabo_data_filename_test = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  const std::string meta_data_filename_test = data_dir + "ALEsKOs01_MetaData_test.csv";

  //// IndustrialStrains0103
  //const std::string metabo_data_filename_train = data_dir + "IndustrialStrains0103_Metabolomics_train.csv";
  //const std::string meta_data_filename_train = data_dir + "IndustrialStrains0103_MetaData_train.csv";
  //const std::string metabo_data_filename_test = data_dir + "IndustrialStrains0103_Metabolomics_test.csv";
  //const std::string meta_data_filename_test = data_dir + "IndustrialStrains0103_MetaData_test.csv";

  /// PLTs

  //// Set the data directories
  ////const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
  //const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
  ////const std::string data_dir = "/home/user/Data/";

  //// Set the biochemical reaction filenames
  //const std::string biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
  ////const std::string biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";

  //// Set the model filenames
  //const std::string model_encoder_weights_filename = data_dir + "VAE_weights.csv";
  //const std::string model_decoder_weights_filename = data_dir + "VAE_weights.csv";

  //// Platelets
  //const std::string metabo_data_filename_train = data_dir + "PLT_timeCourse_Metabolomics_train.csv";
  //const std::string meta_data_filename_train = data_dir + "PLT_timeCourse_MetaData_train.csv";
  //const std::string metabo_data_filename_test = data_dir + "PLT_timeCourse_Metabolomics_test.csv";
  //const std::string meta_data_filename_test = data_dir + "PLT_timeCourse_MetaData_test.csv";

  // Define the model trainers and resources for the trainers
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(512);
  //model_trainer.setBatchSize(1); // Logging only
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsEvaluation(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(false, false, false);
  //model_trainer.setLogging(false, false, true); // Logging only
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);

  // Define the model logger
  ModelLogger<float> model_logger(false, false, false, false, false, true, false, true);

  // Read in the metabolomics data and models
  LatentArithmetic<float> latentArithmetic(16, false, true, true, std::string("Evo04"));
  latentArithmetic.setMetabolomicsData(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test);
  latentArithmetic.setEncDecModels(model_trainer, model_encoder_weights_filename, model_decoder_weights_filename,
    64, 64, 0, 64, 64, 0);
  latentArithmetic.setEncDecModelInterpreters(model_interpreter, model_interpreter);

  // Run the script
  //main_KALE(model_interpreter, model_trainer, model_logger, latentArithmetic, PercentDifferenceTensorOp<float, Eigen::DefaultDevice>(), false, true, false, false);
  //main_KALE(model_interpreter, model_trainer, model_logger, latentArithmetic, EuclideanDistTensorOp<float, Eigen::DefaultDevice>(), false, true, false, false);
  //main_KALE(model_interpreter, model_trainer, model_logger, latentArithmetic, PearsonRTensorOp<float, Eigen::DefaultDevice>(), false, true, false, false);
  main_KALE(model_interpreter, model_trainer, model_logger, latentArithmetic, ManhattanDistTensorOp<float, Eigen::DefaultDevice>(), false, true, false, false);
  //main_KALE(model_interpreter, model_trainer, model_logger, latentArithmetic, LogarithmicDistTensorOp<float, Eigen::DefaultDevice>(), false, true, false, false);
  //main_KALE(model_interpreter, model_trainer, model_logger, latentArithmetic, JeffreysAndMatusitaDistTensorOp<float, Eigen::DefaultDevice>(), false, true, false, false);
  //main_IndustrialStrains(model_interpreter, model_trainer, model_logger, latentArithmetic, true, false);
  //main_PLT(model_interpreter, model_trainer, model_logger, latentArithmetic, false, true, true, true);

  return 0;
}