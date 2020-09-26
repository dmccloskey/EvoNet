/**TODO:  Add copyright*/

#ifndef EVONET_METABOLOMICSLATENTTRAVERSALDATASIMULATOR_H
#define EVONET_METABOLOMICSLATENTTRAVERSALDATASIMULATOR_H

// .h
#include <EvoNet/simulator/BiochemicalDataSimulator.h>

namespace EvoNet
{
  template<typename TensorT>
  class MetabolomicsLatentTraversalDataSimulator : public BiochemicalDataSimulator<TensorT>
  {
  public:
    int n_encodings_continuous_ = 0;
    int n_encodings_discrete_ = 0;
    void makeTrainingDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_training, const std::vector<std::string>& labels_training,
      const int& n_epochs, const int& batch_size, const int& memory_size,
      const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes, const bool& shuffle_data_and_labels) override;
    void makeValidationDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_validation, const std::vector<std::string>& labels_validation,
      const int& n_epochs, const int& batch_size, const int& memory_size,
      const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes, const bool& shuffle_data_and_labels) override;
    void readAndProcessMetabolomicsTrainingAndValidationData(int& n_reaction_ids_training, int& n_labels_training, int& n_component_group_names_training,
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
      const int& n_epochs, const int& batch_size, const int& memory_size) override;
  };
  template<typename TensorT>
  inline void MetabolomicsLatentTraversalDataSimulator<TensorT>::makeTrainingDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_training, const std::vector<std::string>& labels_training, const int & n_epochs, const int & batch_size, const int & memory_size, const int & n_input_nodes, const int & n_loss_output_nodes, const int & n_metric_output_nodes, const bool& shuffle_data_and_labels)
  {
    // infer the input sizes
    assert(n_input_nodes == this->n_encodings_continuous_ + this->n_encodings_discrete_);
    assert(n_loss_output_nodes == data_training.dimension(0));
    assert(n_metric_output_nodes == data_training.dimension(0));
    assert(data_training.dimension(0) == features.size());
    assert(data_training.dimension(1) == labels_training.size());
    assert(this->n_encodings_continuous_ > 0);
    assert(this->n_encodings_discrete_ > 0);
    assert(batch_size > 0);
    assert(memory_size == 1);
    assert(n_epochs == (this->n_encodings_continuous_* this->n_encodings_discrete_) * this->labels_training_.size());

    // Gaussian sampler traversal:
    const TensorT step_size = (0.95 - 0.05) / (batch_size - 1);
    // Assign the encoding values by sampling the 95% confidence limits of the inverse normal distribution
    Eigen::Tensor<TensorT, 4> gaussian_samples(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);
    gaussian_samples.setZero();

    // Concrete Sampler
    Eigen::Tensor<TensorT, 4> categorical_samples(batch_size, memory_size, this->n_encodings_discrete_, n_epochs);
    categorical_samples.setZero();

    int encodings_continuous_iter = 0;
    int encodings_discrete_iter = 0;
    for (int e = 0; e < n_epochs; ++e) {
      // for each epoch, sample the confidence intervals of the next encoding node...
      gaussian_samples.chip(e, 3).chip(encodings_continuous_iter, 2) = (gaussian_samples.chip(e, 3).chip(encodings_continuous_iter, 2).constant(step_size).cumsum(0) + 
        gaussian_samples.chip(e, 3).chip(encodings_continuous_iter, 2).constant(TensorT(0.05) - step_size)).ndtri();
      // for each epoch, iterate the next label of the next categorical node...
      categorical_samples.chip(e, 3).chip(encodings_discrete_iter, 2) = categorical_samples.chip(e, 3).chip(encodings_discrete_iter, 2).constant(TensorT(1));
      ++encodings_continuous_iter;
      if (encodings_continuous_iter >= this->n_encodings_continuous_) {
        encodings_continuous_iter = 0;
        ++encodings_discrete_iter;
      }
      if (encodings_discrete_iter >= this->n_encodings_discrete_) {
        encodings_continuous_iter = 0;
        encodings_discrete_iter = 0;
      }
    }

    // initialize the Tensors
    this->input_data_training_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_training_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_training_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_training_.resize(batch_size, memory_size, n_epochs);

    // expand the training data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_training.dimension(1))), 1);
    assert(expansion_factor == 1);
    const int over_expanded = data_training.dimension(1)*expansion_factor - batch_size * n_epochs;
    assert(over_expanded == 0);
    assert(batch_size * memory_size * n_epochs == data_training.dimension(1)*expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_training_expanded(data_training.dimension(0), data_training.dimension(1)*expansion_factor);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = { 0, i*data_training.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_training.dimension(0), data_training.dimension(1) };
      data_training_expanded.slice(offset1, span1) = data_training;
    }

    // assign the input tensors
    auto data_training_expanded_4d = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), data_training.dimension(1)*expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_training.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 1,2,0,3 }));
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = gaussian_samples;
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = categorical_samples;

    // assign the loss tensors
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_loss_output_nodes, n_epochs })) = data_training_expanded_4d;

    // assign the metric tensors
    this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_metric_output_nodes, n_epochs })) = data_training_expanded_4d;
  }
  template<typename TensorT>
  inline void MetabolomicsLatentTraversalDataSimulator<TensorT>::makeValidationDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_validation, const std::vector<std::string>& labels_validation, const int& n_epochs, const int& batch_size, const int& memory_size, const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes, const bool& shuffle_data_and_labels)
  {
    // infer the input sizes
    assert(n_input_nodes == this->n_encodings_continuous_ + this->n_encodings_discrete_);
    assert(n_loss_output_nodes == data_validation.dimension(0));
    assert(n_metric_output_nodes == data_validation.dimension(0));
    assert(data_validation.dimension(0) == features.size());
    assert(data_validation.dimension(1) == labels_validation.size());
    assert(this->n_encodings_continuous_ > 0);
    assert(this->n_encodings_discrete_ > 0);
    assert(batch_size > 0);
    assert(memory_size == 1);
    assert(n_epochs == (this->n_encodings_continuous_*this->n_encodings_discrete_) * this->labels_validation_.size());

    // Gaussian sampler traversal:
    const TensorT step_size = (0.95 - 0.05) / (batch_size - 1);
    // Assign the encoding values by sampling the 95% confidence limits of the inverse normal distribution
    Eigen::Tensor<TensorT, 4> gaussian_samples(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);
    gaussian_samples.setZero();

    // Concrete Sampler
    Eigen::Tensor<TensorT, 4> categorical_samples(batch_size, memory_size, this->n_encodings_discrete_, n_epochs);
    categorical_samples.setZero();

    int encodings_continuous_iter = 0;
    int encodings_discrete_iter = 0;
    for (int e = 0; e < n_epochs; ++e) {
      // for each epoch, sample the confidence intervals of the next encoding node...
      gaussian_samples.chip(e, 3).chip(encodings_continuous_iter, 2) = (gaussian_samples.chip(e, 3).chip(encodings_continuous_iter, 2).constant(step_size).cumsum(0) +
        gaussian_samples.chip(e, 3).chip(encodings_continuous_iter, 2).constant(TensorT(0.05) - step_size)).ndtri();
      // for each epoch, iterate the next label of the next categorical node...
      categorical_samples.chip(e, 3).chip(encodings_discrete_iter, 2) = categorical_samples.chip(e, 3).chip(encodings_discrete_iter, 2).constant(TensorT(1));
      ++encodings_continuous_iter;
      if (encodings_continuous_iter >= this->n_encodings_continuous_) {
        encodings_continuous_iter = 0;
        ++encodings_discrete_iter;
      }
      if (encodings_discrete_iter >= this->n_encodings_discrete_) {
        encodings_continuous_iter = 0;
        encodings_discrete_iter = 0;
      }
    }

    // initialize the Tensors
    this->input_data_validation_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_validation_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_validation_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_validation_.resize(batch_size, memory_size, n_epochs);

    // expand the validation data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_validation.dimension(1))), 1);
    assert(expansion_factor == 1);
    const int over_expanded = data_validation.dimension(1) * expansion_factor - batch_size * n_epochs;
    assert(over_expanded == 0);
    assert(batch_size * memory_size * n_epochs == data_validation.dimension(1) * expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_validation_expanded(data_validation.dimension(0), data_validation.dimension(1) * expansion_factor);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = { 0, i * data_validation.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_validation.dimension(0), data_validation.dimension(1) };
      data_validation_expanded.slice(offset1, span1) = data_validation;
    }

    // assign the input tensors
    auto data_validation_expanded_4d = data_validation_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_validation.dimension(0), data_validation.dimension(1) * expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_validation.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 1,2,0,3 }));
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = gaussian_samples;
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = categorical_samples;

    // assign the loss tensors
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_loss_output_nodes, n_epochs })) = data_validation_expanded_4d;

    // assign the metric tensors
    this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_metric_output_nodes, n_epochs })) = data_validation_expanded_4d;
  }
  template<typename TensorT>
  inline void MetabolomicsLatentTraversalDataSimulator<TensorT>::readAndProcessMetabolomicsTrainingAndValidationData(int& n_reaction_ids_training, int& n_labels_training, int& n_component_group_names_training, int& n_reaction_ids_validation, int& n_labels_validation, int& n_component_group_names_validation, const std::string& biochem_rxns_filename, const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train, const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
    const bool& use_concentrations, const bool& use_MARs, const bool& sample_values, const bool& iter_values, const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero, const bool& apply_fold_change, const std::string& fold_change_ref, const TensorT& fold_change_log_base, const bool& offline_linear_scale_input, const bool& offline_log_transform_input, const bool& offline_standardize_input, const bool& online_linear_scale_input, const bool& online_log_transform_input, const bool& online_standardize_input,
    int& n_reps_per_sample, const bool& randomize_sample_group_names, const bool& shuffle_data_and_labels, const int& n_epochs, const int& batch_size, const int& memory_size)
  {
    // Read in the data and make the data matrices
    std::vector<std::string> labels_training;
    std::vector<std::string> features_training;
    Eigen::Tensor<TensorT, 2> data_training;
    std::vector<std::string> labels_validation;
    std::vector<std::string> features_validation;
    Eigen::Tensor<TensorT, 2> data_validation;
    this->readAndMakeMetabolomicsTrainingAndValidationDataMatrices(n_reaction_ids_training, n_labels_training, n_component_group_names_training,
      n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
      features_training, data_training, labels_training,
      features_validation, data_validation, labels_validation,
      biochem_rxns_filename,
      metabo_data_filename_train, meta_data_filename_train,
      metabo_data_filename_test, meta_data_filename_test,
      use_concentrations, use_MARs,
      sample_values, iter_values,
      fill_sampling, fill_mean, fill_zero,
      apply_fold_change, fold_change_ref, fold_change_log_base,
      n_reps_per_sample, randomize_sample_group_names,
      n_epochs, batch_size, memory_size);

    // Make the training and validation data caches after an optional transformation step
    if (use_concentrations) {
      // Apply offline transformations
      this->transformTrainingAndValidationDataOffline(data_training, data_validation,
        offline_linear_scale_input, offline_log_transform_input, offline_standardize_input, false, -1, -1, false, -1, -1);

      // Apply online transformations
      this->transformTrainingAndValidationDataOnline(data_training, data_validation,
        online_linear_scale_input, online_log_transform_input, online_standardize_input);

      // Make the training data cache
      this->makeTrainingDataForCache(features_training, data_training, labels_training, n_epochs, batch_size, memory_size,
        this->n_encodings_continuous_ + this->n_encodings_discrete_, n_component_group_names_training, n_component_group_names_training, shuffle_data_and_labels);
      this->makeValidationDataForCache(features_validation, data_validation, labels_validation, n_epochs, batch_size, memory_size,
        this->n_encodings_continuous_ + this->n_encodings_discrete_, n_component_group_names_training, n_component_group_names_training, shuffle_data_and_labels);
    }
    else if (use_MARs) {
      // Apply offline transformations
      TensorT min_value = 1e-3;
      TensorT max_value = 1e3;
      if (offline_log_transform_input) {
        min_value = std::log(min_value);
        max_value = std::log(max_value);
      }
      this->transformTrainingAndValidationDataOffline(data_training, data_validation,
        offline_linear_scale_input, offline_log_transform_input, offline_standardize_input, true, min_value, max_value, false, -1, -1);

      // Apply online transformations
      this->transformTrainingAndValidationDataOnline(data_training, data_validation,
        online_linear_scale_input, online_log_transform_input, online_standardize_input);

      // Make the training data cache
      this->makeTrainingDataForCache(features_training, data_training, labels_training, n_epochs, batch_size, memory_size,
        this->n_encodings_continuous_ + this->n_encodings_discrete_, n_reaction_ids_validation, n_reaction_ids_validation, shuffle_data_and_labels);
      this->makeValidationDataForCache(features_validation, data_validation, labels_validation, n_epochs, batch_size, memory_size,
        this->n_encodings_continuous_ + this->n_encodings_discrete_, n_reaction_ids_validation, n_reaction_ids_validation, shuffle_data_and_labels);
    }
  }
}
#endif //EVONET_METABOLOMICSLATENTTRAVERSALDATASIMULATOR_H