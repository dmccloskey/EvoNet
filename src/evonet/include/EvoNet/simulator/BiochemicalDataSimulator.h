/**TODO:  Add copyright*/

#ifndef EVONET_BIOCHEMICALDATASIMULATOR_H
#define EVONET_BIOCHEMICALDATASIMULATOR_H

// .h
#include <EvoNet/simulator/BiochemicalReaction.h>
#include <EvoNet/simulator/DataSimulator.h>
#include <EvoNet/io/Parameters.h>

namespace EvoNet
{
  /// List of all available parameters and their types
  namespace EvoNetParameters {
    namespace Examples {
      struct MetaboDataTrainFilename : Parameter<std::string> { using Parameter::Parameter; };
      struct MetaboDataTestFilename : Parameter<std::string> { using Parameter::Parameter; };
      struct MetaDataTrainFilename : Parameter<std::string> { using Parameter::Parameter; };
      struct MetaDataTestFilename : Parameter<std::string> { using Parameter::Parameter; };
      struct UseConcentrations : Parameter<bool> { using Parameter::Parameter; };
      struct UseMARs : Parameter<bool> { using Parameter::Parameter; };
      struct SampleValues : Parameter<bool> { using Parameter::Parameter; };
      struct IterValues : Parameter<bool> { using Parameter::Parameter; };
      struct FillSampling : Parameter<bool> { using Parameter::Parameter; };
      struct FillMean : Parameter<bool> { using Parameter::Parameter; };
      struct FillZero : Parameter<bool> { using Parameter::Parameter; };
      struct ApplyFoldChange : Parameter<bool> { using Parameter::Parameter; };
      struct FoldChangeRef : Parameter<std::string> { using Parameter::Parameter; };
      struct FoldChangeLogBase : Parameter<float> { using Parameter::Parameter; };
      struct OfflineLinearScaleInput : Parameter<bool> { using Parameter::Parameter; };
      struct OfflineLogTransformInput : Parameter<bool> { using Parameter::Parameter; };
      struct OfflineStandardizeInput : Parameter<bool> { using Parameter::Parameter; };
      struct OnlineLinearScaleInput : Parameter<bool> { using Parameter::Parameter; };
      struct OnlineLogTransformInput : Parameter<bool> { using Parameter::Parameter; };
      struct OnlineStandardizeInput : Parameter<bool> { using Parameter::Parameter; };
    }
  }
  /**
    @brief A class to generate -omics data
  */
  template <typename TensorT>
  class BiochemicalDataSimulator : public DataSimulator<TensorT>
  {
  public:
    BiochemicalDataSimulator() = default; ///< Default constructor
    ~BiochemicalDataSimulator() = default; ///< Default destructor

    /*
    @brief Simulate the evaluation data for the next batch
    */
    void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override;

    /*
    @brief Simulate the training data for the next batch
    */
    void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override;

    /*
    @brief Simulate the validation data for the next batch
    */
    void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override;

    /*
    @brief Transform the training and validation data.
    Member variables for min, max, mean, and var values can be optionally specified by the user when these values are known ahead of time (e.g., MARs min/max = 1e-3/1e3)
    transformation will be applied to the entire training data set and the parameters from the training data set will be applied to the validation data set

    @param[in, out] data_training training data set where dim 0 = features and dim 1 = samples
    @param[in, out] data_validation validation data set where dim 0 = features and dim 1 = samples
    @param[in] linear_scale
    @param[in] log_transform
    @param[in] standardize
    @param[in] min_value_training linear scale min value for offline normalization
    @param[in] max_value_training linear scale max value for offline normalization
    @param[in] mean_value_training standardize mean value for offline normalization
    @param[in] var_value_training standardize var value for offline normalization
    */
    void transformTrainingAndValidationDataOffline(Eigen::Tensor<TensorT, 2>& data_training, Eigen::Tensor<TensorT, 2>& data_validation, 
      const bool& linear_scale, const bool & log_transform, const bool& standardize,
      const bool& use_min_max_linear_scale = false, const int& min_value_training = -1, const int& max_value_training = -1,
      const bool& use_mean_var_standardize = false, const int& mean_value_training = -1, const int& var_value_training = -1);

    /*
    @brief Transform the training and validation data.
    Transformation will be applied sample by sample to the training and validation data

    @param[in, out] data_training training data set where dim 0 = features and dim 1 = samples
    @param[in, out] data_validation validation data set where dim 0 = features and dim 1 = samples
    @param[in] linear_scale
    @param[in] log_transform
    @param[in] standardize
    */
    void transformTrainingAndValidationDataOnline(Eigen::Tensor<TensorT, 2>& data_training, Eigen::Tensor<TensorT, 2>& data_validation, const bool& linear_scale, const bool & log_transform, const bool& standardize);


    /* Read and process the training and validation data for metabolomics analysis

    @param[out] n_reaction_ids_training
    @param[out] n_labels_training
    @param[out] n_component_group_names_training
    @param[out] n_reaction_ids_validation
    @param[out] n_labels_validation
    @param[out] n_component_group_names_validation
    @param[out] features_training The list of training features (features)
    @param[out] data_training The training data matrix (features x samples)
    @param[out] features_validation The list of validation features (features)
    @param[out] labels_training The training labels vector (samples)
    @param[out] data_validation The validation data matrix (features x samples)
    @param[out] labels_validation The validation labels vector (samples)
    @param[out] n_component_group_names_validation
    @param[in] biochem_rxns_filename
    @param[in] metabo_data_filename_train
    @param[in] meta_data_filename_train
    @param[in] metabo_data_filename_test
    @param[in] meta_data_filename_test
    @param[in] use_concentrations
    @param[in] use_MARs
    @param[in] sample_values
    @param[in] iter_values
    @param[in] fill_sampling
    @param[in] fill_mean
    @param[in] fill_zero
    @param[in] apply_fold_change
    @param[in] fold_change_ref
    @param[in] online_standardize_input
    @param[in,out] n_reps_per_sample The number of replicates per sample to use for sample values. If -1, the number of reps will be split evenly.
    @param[in] randomize_sample_group_names Whether to randomize the order of sample groups as the data matrix is made
    @param[in] n_epochs
    @param[in] batch_size
    @param[in] memory_size
    */
    void readAndMakeMetabolomicsTrainingAndValidationDataMatrices(int& n_reaction_ids_training, int& n_labels_training, int& n_component_group_names_training,
      int& n_reaction_ids_validation, int& n_labels_validation, int& n_component_group_names_validation,
      std::vector<std::string>& features_training, Eigen::Tensor<TensorT, 2>& data_training, std::vector<std::string>& labels_training,
      std::vector<std::string>& features_validation, Eigen::Tensor<TensorT, 2>& data_validation, std::vector<std::string>& labels_validation,
      const std::string& biochem_rxns_filename,
      const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
      const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
      const bool& use_concentrations, const bool& use_MARs,
      const bool& sample_values, const bool& iter_values,
      const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero,
      const bool& apply_fold_change, const std::string& fold_change_ref, const TensorT& fold_change_log_base,
      int& n_reps_per_sample, const bool& randomize_sample_group_names,
      const int& n_epochs, const int& batch_size, const int& memory_size);

    /*
    @brief Make the training data cache from the training data.  The classification and reconstruction version of this method will be different,
    and it is intended for these methods to be overridden when a classification or reconstruction derived class is made.

    @param[in] features The data set features along dim 0
    @param[in] data_training training data set where dim 0 = features and dim 1 = samples
    @param[in] labels_training The data set labels along dim 1
    @param[in] n_epochs
    @param[in] batch_size
    @param[in] memory_size
    @param[in] n_input_nodes
    @param[in] n_loss_output_nodes
    @param[in] n_metric_output_nodes
    @param[in] shuffle_data_and_labels If true, will shuffle the expanded data and label tensors prior to initializing the training/validation data caches
    */
    virtual void makeTrainingDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_training, const std::vector<std::string>& labels_training,
      const int& n_epochs, const int& batch_size, const int& memory_size,
      const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes, const bool& shuffle_data_and_labels) = 0;

    /*
    @brief Make the validation data cache from the validation data.  The classification and reconstruction version of this method will be different,
    and it is intended for these methods to be overridden when a classification or reconstruction derived class is made.

    @param[in] features The data set features along dim 0
    @param[in] data_validation validation data set where dim 0 = features and dim 1 = samples
    @param[in] labels_validation The data set labels along dim 1
    @param[in] n_epochs
    @param[in] batch_size
    @param[in] memory_size
    @param[in] n_input_nodes
    @param[in] n_loss_output_nodes
    @param[in] n_metric_output_nodes
    @param[in] shuffle_data_and_labels If true, will shuffle the expanded data and label tensors prior to initializing the training/validation data caches
    */
    virtual void makeValidationDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_validation, const std::vector<std::string>& labels_validation, 
      const int& n_epochs, const int& batch_size, const int& memory_size,
      const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes, const bool& shuffle_data_and_labels) = 0;

    /* Read and process the training and validation data for metabolomics analysis

    @param[out] n_reaction_ids_training
    @param[out] n_labels_training
    @param[out] n_component_group_names_training
    @param[out] n_reaction_ids_validation
    @param[out] n_labels_validation
    @param[out] n_component_group_names_validation
    @param[in] biochem_rxns_filename
    @param[in] metabo_data_filename_train
    @param[in] meta_data_filename_train
    @param[in] metabo_data_filename_test
    @param[in] meta_data_filename_test
    @param[in] use_concentrations
    @param[in] use_MARs
    @param[in] sample_values
    @param[in] iter_values
    @param[in] fill_sampling
    @param[in] fill_mean
    @param[in] fill_zero
    @param[in] apply_fold_change
    @param[in] fold_change_ref
    @param[in] offline_linear_scale_input
    @param[in] offline_log_transform_input
    @param[in] offline_standardize_input
    @param[in] online_linear_scale_input
    @param[in] online_log_transform_input
    @param[in] online_standardize_input
    @param[in,out] n_reps_per_sample The number of replicates per sample to use for sample values. If -1, the number of reps will be split evenly.
    @param[in] randomize_sample_group_names Whether to randomize the order of sample groups as the data matrix is made
    @param[in] shuffle_data_and_labels Whether to shuffle the data and labels tensors after the data tensors have been expanded/replicated to fill the requested batch_size and n_epochs parameters
    @param[in] n_epochs
    @param[in] batch_size
    @param[in] memory_size
    */
    virtual void readAndProcessMetabolomicsTrainingAndValidationData(int& n_reaction_ids_training, int& n_labels_training, int& n_component_group_names_training,
      int& n_reaction_ids_validation, int& n_labels_validation, int& n_component_group_names_validation,
      const std::string& biochem_rxns_filename,
      const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
      const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
      const bool& use_concentrations, const bool& use_MARs,
      const bool& sample_values, const bool& iter_values,
      const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero,
      const bool& apply_fold_change, const std::string& fold_change_ref, const TensorT& fold_change_log_base,
      const bool& offline_linear_scale_input, const bool& offline_log_transform_input, const bool& offline_standardize_input,
      const bool& online_linear_scale_input, const bool& online_log_transform_input, const bool& online_standardize_input,
      int& n_reps_per_sample, const bool& randomize_sample_group_names, const bool& shuffle_data_and_labels,
      const int& n_epochs, const int& batch_size, const int& memory_size) = 0;

    bool use_train_for_eval_ = true;    
    Eigen::Tensor<TensorT, 4> input_data_training_;
    Eigen::Tensor<TensorT, 4> loss_output_data_training_;
    Eigen::Tensor<TensorT, 4> metric_output_data_training_;
    Eigen::Tensor<TensorT, 3> time_steps_training_;
    Eigen::Tensor<TensorT, 4> input_data_validation_;
    Eigen::Tensor<TensorT, 4> loss_output_data_validation_;
    Eigen::Tensor<TensorT, 4> metric_output_data_validation_;
    Eigen::Tensor<TensorT, 3> time_steps_validation_;
    int n_epochs_training_ = 0;
    int n_epochs_validation_ = 0;
    std::vector<std::string> labels_training_;
    std::vector<std::string> labels_validation_;
  protected:
    void getTrainingDataFromCache_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps);
    void getValidationDataFromCache_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps);
	};
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    // TODO: add logic to regenerate the cache when needed
    if (this->use_train_for_eval_) this->getTrainingDataFromCache_(input_data, Eigen::Tensor<TensorT, 3>(), Eigen::Tensor<TensorT, 3>(), time_steps);
    else this->getValidationDataFromCache_(input_data, Eigen::Tensor<TensorT, 3>(), Eigen::Tensor<TensorT, 3>(), time_steps);
  }
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    // TODO: add logic to regenerate the cache when needed
    this->getTrainingDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
  }
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    // TODO: add logic to regenerate the cache when needed
    this->getValidationDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
  }
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::transformTrainingAndValidationDataOffline(Eigen::Tensor<TensorT, 2>& data_training, Eigen::Tensor<TensorT, 2>& data_validation, 
    const bool & linear_scale, const bool & log_transform, const bool & standardize,
    const bool& use_min_max_linear_scale, const int& min_value_training, const int& max_value_training,
    const bool& use_mean_var_standardize, const int& mean_value_training, const int& var_value_training)
  {
    // Estimate the parameters from the training data and apply to the training data
    // Apply the training data paremeters to the validation data
    if (log_transform) {
      data_training = data_training.log();
      data_validation = data_validation.log();
    }
    if (standardize) {
      Standardize<TensorT, 2> standardizeTrans;
      if (use_mean_var_standardize) standardizeTrans.setMeanAndVar(mean_value_training, var_value_training);
      else standardizeTrans.setMeanAndVar(data_training);
      data_training = standardizeTrans(data_training);
      data_validation = standardizeTrans(data_validation);
    }
    if (linear_scale) {
      LinearScale<TensorT, 2> linearScaleTrans(0, 1);
      if (use_min_max_linear_scale) linearScaleTrans.setDomain(min_value_training, max_value_training);
      else linearScaleTrans.setDomain(data_training);
      data_training = linearScaleTrans(data_training);
      data_validation = linearScaleTrans(data_validation);
    }
  }
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::transformTrainingAndValidationDataOnline(Eigen::Tensor<TensorT, 2>& data_training, Eigen::Tensor<TensorT, 2>& data_validation, const bool & linear_scale, const bool & log_transform, const bool & standardize)
  {
    // Apply the transformation to both training and test set on a per sample basis
    if (log_transform) {
      data_training = data_training.log();
      data_validation = data_validation.log();
    }
    if (standardize) {
      for (int sample_iter = 0; sample_iter < data_training.dimension(1); ++sample_iter) {
        Eigen::array<Eigen::Index, 2> offset = {0, sample_iter};
        Eigen::array<Eigen::Index, 2> span = { data_training.dimension(0), 1 };
        Eigen::Tensor<TensorT, 2> data_slice = data_training.slice(offset, span);
        Standardize<TensorT, 2> standardizeTrans(data_slice);
        data_training.slice(offset, span) = standardizeTrans(data_slice);
      }
      for (int sample_iter = 0; sample_iter < data_validation.dimension(1); ++sample_iter) {
        Eigen::array<Eigen::Index, 2> offset = { 0, sample_iter };
        Eigen::array<Eigen::Index, 2> span = { data_validation.dimension(0), 1 };
        Eigen::Tensor<TensorT, 2> data_slice = data_validation.slice(offset, span);
        Standardize<TensorT, 2> standardizeTrans(data_slice);
        data_validation.slice(offset, span) = standardizeTrans(data_slice);
      }
    }
    if (linear_scale) {
      for (int sample_iter = 0; sample_iter < data_training.dimension(1); ++sample_iter) {
        Eigen::array<Eigen::Index, 2> offset = {0, sample_iter};
        Eigen::array<Eigen::Index, 2> span = { data_training.dimension(0), 1 };
        Eigen::Tensor<TensorT, 2> data_slice = data_training.slice(offset, span);
        LinearScale<TensorT, 2> linearScaleTrans(data_slice, 0, 1);
        data_training.slice(offset, span) = linearScaleTrans(data_slice);
      }
      for (int sample_iter = 0; sample_iter < data_validation.dimension(1); ++sample_iter) {
        Eigen::array<Eigen::Index, 2> offset = { 0, sample_iter };
        Eigen::array<Eigen::Index, 2> span = { data_validation.dimension(0), 1 };
        Eigen::Tensor<TensorT, 2> data_slice = data_validation.slice(offset, span);
        LinearScale<TensorT, 2> linearScaleTrans(data_slice, 0, 1);
        data_validation.slice(offset, span) = linearScaleTrans(data_slice);
      }
    }
  }
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::readAndMakeMetabolomicsTrainingAndValidationDataMatrices(int& n_reaction_ids_training, int& n_labels_training, int& n_component_group_names_training, int& n_reaction_ids_validation, int& n_labels_validation, int& n_component_group_names_validation, std::vector<std::string>& features_training, Eigen::Tensor<TensorT, 2>& data_training, std::vector<std::string>& labels_training, std::vector<std::string>& features_validation, Eigen::Tensor<TensorT, 2>& data_validation, std::vector<std::string>& labels_validation, const std::string& biochem_rxns_filename, const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train, const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test, const bool& use_concentrations, const bool& use_MARs, const bool& sample_values, const bool& iter_values, const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero, const bool& apply_fold_change, const std::string& fold_change_ref, const TensorT& fold_change_log_base, int& n_reps_per_sample, const bool& randomize_sample_group_names, const int& n_epochs, const int& batch_size, const int& memory_size)
  {
    // define the data simulator
    BiochemicalReactionModel<TensorT> reaction_model;

    // clear the input data
    n_reaction_ids_training = -1;
    n_labels_training = -1;
    n_component_group_names_training = -1;
    n_reaction_ids_validation = -1;
    n_labels_validation = -1;
    n_component_group_names_validation = -1;
    this->labels_training_.clear();
    this->labels_validation_.clear();

    // Read in the training data
    reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
    reaction_model.readMetabolomicsData(metabo_data_filename_train);
    reaction_model.readMetaData(meta_data_filename_train);
    reaction_model.findComponentGroupNames();
    if (use_MARs) {
      reaction_model.findMARs();
      reaction_model.findMARs(true, false);
      reaction_model.findMARs(false, true);
      reaction_model.removeRedundantMARs();
    }
    reaction_model.findLabels();
    n_reaction_ids_training = reaction_model.reaction_ids_.size();
    n_labels_training = reaction_model.labels_.size();
    n_component_group_names_training = reaction_model.component_group_names_.size();
    this->labels_training_ = reaction_model.labels_;

    // define the n_reps_per_sample if not defined previously
    if (n_reps_per_sample <= 0)
      n_reps_per_sample = batch_size * n_epochs / reaction_model.sample_group_names_.size();

    // Make the training data
    labels_training.clear();
    features_training.clear();
    std::map<std::string, int> sample_group_name_to_reps;
    std::pair<int, int> max_reps_n_labels = reaction_model.getMaxReplicatesAndNLabels(sample_group_name_to_reps, reaction_model.sample_group_names_, reaction_model.component_group_names_);
    if (use_concentrations) {
      // Adjust the number of replicates per sample group
      if (sample_values) for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;

      // Initialize the input labels and data
      if (sample_values) data_training.resize(int(reaction_model.component_group_names_.size()), n_reps_per_sample * int(sample_group_name_to_reps.size()));
      else data_training.resize(int(reaction_model.component_group_names_.size()), max_reps_n_labels.second);
      features_training = reaction_model.component_group_names_;

      // Create the data matrix
      reaction_model.getMetDataAsTensors(data_training, labels_training,
        reaction_model.sample_group_names_, reaction_model.component_group_names_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
        use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base, randomize_sample_group_names);
    }
    else if (use_MARs) {
      // Adjust the number of replicates per sample group
      for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;

      // Initialize the input labels and data
      labels_training.reserve(n_reps_per_sample * sample_group_name_to_reps.size());
      data_training.resize(int(reaction_model.reaction_ids_.size()), n_reps_per_sample * int(sample_group_name_to_reps.size()));
      features_training = reaction_model.reaction_ids_;

      // Create the data matrix
      reaction_model.getMetDataAsTensors(data_training, labels_training,
        reaction_model.sample_group_names_, reaction_model.reaction_ids_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
        use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base, randomize_sample_group_names);
    }

    // Read in the validation data
    reaction_model.clear();
    reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
    reaction_model.readMetabolomicsData(metabo_data_filename_test);
    reaction_model.readMetaData(meta_data_filename_test);
    reaction_model.findComponentGroupNames();
    if (use_MARs) {
      reaction_model.findMARs();
      reaction_model.findMARs(true, false);
      reaction_model.findMARs(false, true);
      reaction_model.removeRedundantMARs();
    }
    reaction_model.findLabels();
    n_reaction_ids_validation = reaction_model.reaction_ids_.size();
    n_labels_validation = reaction_model.labels_.size();
    n_component_group_names_validation = reaction_model.component_group_names_.size();
    this->labels_validation_ = reaction_model.labels_;

    // Make the validation data caches
    labels_validation.clear();
    features_validation.clear();
    sample_group_name_to_reps.clear();
    max_reps_n_labels = reaction_model.getMaxReplicatesAndNLabels(sample_group_name_to_reps, reaction_model.sample_group_names_, reaction_model.component_group_names_);
    if (use_concentrations) {
      // Adjust the number of replicates per sample group
      if (sample_values) for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;

      // Initialize the input labels and data
      if (sample_values) data_validation.resize(int(reaction_model.component_group_names_.size()), n_reps_per_sample * int(sample_group_name_to_reps.size()));
      else data_validation.resize(int(reaction_model.component_group_names_.size()), max_reps_n_labels.second);
      features_validation = reaction_model.component_group_names_;

      // Create the data matrix
      reaction_model.getMetDataAsTensors(data_validation, labels_validation,
        reaction_model.sample_group_names_, reaction_model.component_group_names_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
        use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base, randomize_sample_group_names);
    }
    else if (use_MARs) {
      // Adjust the number of replicates per sample group
      for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;

      // Initialize the input labels and data
      labels_validation.reserve(n_reps_per_sample * sample_group_name_to_reps.size());
      data_validation.resize(int(reaction_model.reaction_ids_.size()), n_reps_per_sample * int(sample_group_name_to_reps.size()));
      features_validation = reaction_model.reaction_ids_;

      // Create the data matrix
      reaction_model.getMetDataAsTensors(data_validation, labels_validation,
        reaction_model.sample_group_names_, reaction_model.reaction_ids_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
        use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base, randomize_sample_group_names);
    }

    // Checks for the training and validation data
    assert(n_reaction_ids_training == n_reaction_ids_validation);
    assert(n_labels_training == n_labels_validation);
    assert(n_component_group_names_training == n_component_group_names_validation);
  }
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::getTrainingDataFromCache_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    // Check that we have not exceeded the number of cached training data
    if (this->n_epochs_training_ >= this->input_data_training_.dimension(3))
      this->n_epochs_training_ = 0;

    // Copy over the training data
    input_data = this->input_data_training_.chip(this->n_epochs_training_, 3).slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }),
      Eigen::array<Eigen::Index, 3>({ input_data.dimension(0), input_data.dimension(1), input_data.dimension(2) }));
    loss_output_data = this->loss_output_data_training_.chip(this->n_epochs_training_, 3).slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }),
      Eigen::array<Eigen::Index, 3>({ loss_output_data.dimension(0), loss_output_data.dimension(1), loss_output_data.dimension(2) }));
    metric_output_data = this->metric_output_data_training_.chip(this->n_epochs_training_, 3).slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }),
      Eigen::array<Eigen::Index, 3>({ metric_output_data.dimension(0), metric_output_data.dimension(1), metric_output_data.dimension(2) }));
    //time_steps = this->time_steps_training_.chip(this->n_epochs_training_, 2);

    // Increment the iterator
    this->n_epochs_training_++;
  }
  template<typename TensorT>
  inline void BiochemicalDataSimulator<TensorT>::getValidationDataFromCache_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    // Check that we have not exceeded the number of cached validation data
    if (this->n_epochs_validation_ >= this->input_data_validation_.dimension(3))
      this->n_epochs_validation_ = 0;

    // Copy over the validation data
    input_data = this->input_data_validation_.chip(this->n_epochs_validation_, 3).slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }),
      Eigen::array<Eigen::Index, 3>({ input_data.dimension(0), input_data.dimension(1), input_data.dimension(2) }));
    loss_output_data = this->loss_output_data_validation_.chip(this->n_epochs_validation_, 3).slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }),
      Eigen::array<Eigen::Index, 3>({ loss_output_data.dimension(0), loss_output_data.dimension(1), loss_output_data.dimension(2) }));
    metric_output_data = this->metric_output_data_validation_.chip(this->n_epochs_validation_, 3).slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }),
      Eigen::array<Eigen::Index, 3>({ metric_output_data.dimension(0), metric_output_data.dimension(1), metric_output_data.dimension(2) }));
    //time_steps = this->time_steps_validation_.chip(this->n_epochs_validation_, 2);

    // Increment the iterator
    this->n_epochs_validation_++;
  }
}

#endif //EVONET_BIOCHEMICALDATASIMULATOR_H