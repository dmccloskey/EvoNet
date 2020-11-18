/**TODO:  Add copyright*/

#ifndef EVONET_METABOLOMICSLATENTUNSCLASSDATASIMULATOR_H
#define EVONET_METABOLOMICSLATENTUNSCLASSDATASIMULATOR_H

// .h
#include <EvoNet/simulator/BiochemicalDataSimulator.h>

namespace EvoNet
{
  template<typename TensorT>
  class MetabolomicsLatentUnsClassDataSimulator : public BiochemicalDataSimulator<TensorT>
  {
  public:
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
  inline void MetabolomicsLatentUnsClassDataSimulator<TensorT>::makeTrainingDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_training, const std::vector<std::string>& labels_training, const int& n_epochs, const int& batch_size, const int& memory_size, const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes, const bool& shuffle_data_and_labels)
  {
    // infer the input sizes
    assert(n_input_nodes == data_training.dimension(0));
    assert(n_loss_output_nodes == /*2*this->n_encodings_continuous_ + */this->n_encodings_discrete_);
    assert(n_metric_output_nodes == /*2*this->n_encodings_continuous_ + */this->n_encodings_discrete_);
    assert(data_training.dimension(0) == features.size());
    assert(data_training.dimension(1) == labels_training.size());
    assert(this->n_encodings_continuous_ > 0);
    assert(this->n_encodings_discrete_  == this->labels_training_.size());
    assert(batch_size > 0);
    assert(memory_size == 1);
    assert(n_epochs == this->n_encodings_discrete_ * this->labels_training_.size());

    // Dummy data for the KL divergence losses
    Eigen::Tensor<TensorT, 4> KL_losses_continuous(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);
    KL_losses_continuous.setZero();

    // initialize the Tensors
    this->input_data_training_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_training_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_training_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_training_.resize(batch_size, memory_size, n_epochs);

    // expand the training data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_training.dimension(1))), 1);
    assert(expansion_factor == 1);
    const int over_expanded = data_training.dimension(1) * expansion_factor - batch_size * n_epochs;
    assert(over_expanded == 0);
    assert(batch_size * memory_size * n_epochs == data_training.dimension(1) * expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_training_expanded(data_training.dimension(0), data_training.dimension(1) * expansion_factor);
    Eigen::Tensor<std::string, 2> labels_training_expanded(data_training.dimension(1) * expansion_factor, 1);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = { 0, i * data_training.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_training.dimension(0), data_training.dimension(1) };
      data_training_expanded.slice(offset1, span1) = data_training;

      // Slices for the labels with a reorder to partition a unique label into each batch
      int step = 0, stride = labels_training.size()/this->labels_training_.size(), batch_iter = 0, iter = 0;
      for (int j = 0; j < data_training.dimension(1); ++j) {
        labels_training_expanded(i * data_training.dimension(1) + j, 0) = labels_training.at(iter);
        ++batch_iter;
        ++iter;
        if (batch_iter >= batch_size) {
          batch_iter = 0;
          iter -= batch_size; // subtract out the iterations along the batch
          iter += stride; // and jump to the next set of labels
        }
        if (iter >= data_training.dimension(1)) {
          ++step;
          iter = step;
        }
      }
    }

    // make the one-hot encodings       
    Eigen::Tensor<TensorT, 2> one_hot_vec = OneHotEncoder<std::string, TensorT>(labels_training_expanded, this->labels_training_);

    // assign the input tensors
    auto data_training_expanded_4d = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), data_training.dimension(1) * expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_training.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 1,2,0,3 }));
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_input_nodes, n_epochs })) = data_training_expanded_4d;

    // assign the loss tensors
    auto one_hot_vec_4d = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(1) * expansion_factor - over_expanded, one_hot_vec.dimension(1) })
    ).reshape(Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_epochs, int(labels_training_.size()) })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 0,1,3,2 }));
    //this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 2 * this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;

    // assign the metric tensors
    //this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 2 * this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;
    this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;
  }
  template<typename TensorT>
  inline void MetabolomicsLatentUnsClassDataSimulator<TensorT>::makeValidationDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_validation, const std::vector<std::string>& labels_validation, const int& n_epochs, const int& batch_size, const int& memory_size, const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes, const bool& shuffle_data_and_labels)
  {
    // infer the input sizes
    assert(n_input_nodes == data_validation.dimension(0));
    assert(n_loss_output_nodes == /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_);
    assert(n_metric_output_nodes == /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_);
    assert(data_validation.dimension(0) == features.size());
    assert(data_validation.dimension(1) == labels_validation.size());
    assert(this->n_encodings_continuous_ > 0);
    assert(this->n_encodings_discrete_ == this->labels_validation_.size());
    assert(batch_size > 0);
    assert(memory_size == 1);
    assert(n_epochs == this->n_encodings_discrete_ * this->labels_validation_.size());

    // Dummy data for the KL divergence losses
    Eigen::Tensor<TensorT, 4> KL_losses_continuous(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);
    KL_losses_continuous.setZero();

    // initialize the Tensors
    this->input_data_validation_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_validation_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_validation_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_validation_.resize(batch_size, memory_size, n_epochs);

    // expand the validation data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_validation.dimension(1))), 1);
    if (expansion_factor != 1) {
      std::cout << "validation expansion_factor = " << expansion_factor << "." << std::endl;
    };
    const int over_expanded = data_validation.dimension(1) * expansion_factor - batch_size * n_epochs;
    if (over_expanded != 0) {
      std::cout << "validation over_expanded = " << over_expanded << "." << std::endl;
    }
    assert(batch_size * memory_size * n_epochs == data_validation.dimension(1) * expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_validation_expanded(data_validation.dimension(0), data_validation.dimension(1) * expansion_factor);
    Eigen::Tensor<std::string, 2> labels_validation_expanded(data_validation.dimension(1) * expansion_factor, 1);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = { 0, i * data_validation.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_validation.dimension(0), data_validation.dimension(1) };
      data_validation_expanded.slice(offset1, span1) = data_validation;

      // Slices for the labels with a reorder
      int step = 0, stride = labels_validation.size()/this->labels_validation_.size(), iter = 0;
      for (int j = 0; j < data_validation.dimension(1); ++j) {
        labels_validation_expanded(i * data_validation.dimension(1) + j, 0) = labels_validation.at(iter);
        iter += stride;
        if (iter >= data_validation.dimension(1)) {
          ++step;
          iter = step;
        }
      }
    }

    // make the one-hot encodings       
    Eigen::Tensor<TensorT, 2> one_hot_vec = OneHotEncoder<std::string, TensorT>(labels_validation_expanded, this->labels_validation_);

    // assign the input tensors
    auto data_validation_expanded_4d = data_validation_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_validation.dimension(0), data_validation.dimension(1) * expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_validation.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 1,2,0,3 }));
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_input_nodes, n_epochs })) = data_validation_expanded_4d;

    // assign the loss tensors
    auto one_hot_vec_4d = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_validation.dimension(1) * expansion_factor - over_expanded, one_hot_vec.dimension(1) })
    ).reshape(Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_epochs, int(labels_validation_.size()) })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 0,1,3,2 }));
    //this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 2 * this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;

    // assign the metric tensors
    //this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    //this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 2 * this->n_encodings_continuous_, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;
    this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = one_hot_vec_4d;
  }
  template<typename TensorT>
  inline void MetabolomicsLatentUnsClassDataSimulator<TensorT>::readAndProcessMetabolomicsTrainingAndValidationData(int& n_reaction_ids_training, int& n_labels_training, int& n_component_group_names_training, int& n_reaction_ids_validation, int& n_labels_validation, int& n_component_group_names_validation, const std::string& biochem_rxns_filename, const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train, const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
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
      n_reps_per_sample, false, //randomize_sample_group_names,
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
        n_component_group_names_training, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, shuffle_data_and_labels);
      this->makeValidationDataForCache(features_validation, data_validation, labels_validation, n_epochs, batch_size, memory_size,
        n_component_group_names_training, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, shuffle_data_and_labels);
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
        n_reaction_ids_validation, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, shuffle_data_and_labels);
      this->makeValidationDataForCache(features_validation, data_validation, labels_validation, n_epochs, batch_size, memory_size,
        n_reaction_ids_validation, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, /*2 * this->n_encodings_continuous_ + */this->n_encodings_discrete_, shuffle_data_and_labels);
    }
  }
}

#endif //EVONET_METABOLOMICSLATENTUNSCLASSDATASIMULATOR_H