/**TODO:  Add copyright*/

#ifndef EVONET_METABOLOMICSRECONSTRUCTIONDATASIMULATOR_H
#define EVONET_METABOLOMICSRECONSTRUCTIONDATASIMULATOR_H

// .h
#include <EvoNet/simulator/BiochemicalDataSimulator.h>

namespace EvoNet
{
  template<typename TensorT>
  class MetabolomicsReconstructionDataSimulator : public BiochemicalDataSimulator<TensorT>
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

    /* Get the non randomized training data from the cache corresponding to a single label
    
    Assumes that the data has not been randomized nor shuffled.
      i.e., randomize_sample_group_names = false and shuffle_data_and_labels = false
    Assumes that the expansion factor was 0
      i.e., n_reps_per_sample = -1

    @param[in] label The label to get data for
    */
    void getNonRandomizedEncoderTrainingInputFromCacheByLabel(const std::string& label, const int& n_features, Eigen::Tensor<TensorT, 4>& input_data);

    /* Get the non randomized training data from the cache corresponding to a single label
    
    Assumes that the data has not been randomized nor shuffled.
      i.e., randomize_sample_group_names = false and shuffle_data_and_labels = false
    Assumes that the expansion factor was 0
      i.e., n_reps_per_sample = -1

    @param[in] label The label to get data for
    */
    void getNonRandomizedDecoderTrainingOutputFromCacheByLabel(const std::string& label, const int& n_features, Eigen::Tensor<TensorT, 4>& output_data);
  };
  template<typename TensorT>
  inline void MetabolomicsReconstructionDataSimulator<TensorT>::makeTrainingDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_training, const std::vector<std::string>& labels_training, const int & n_epochs, const int & batch_size, const int & memory_size, const int & n_input_nodes, const int & n_loss_output_nodes, const int & n_metric_output_nodes, const bool& shuffle_data_and_labels)
  {
    // infer the input sizes
    const int input_nodes = data_training.dimension(0);
    assert(n_input_nodes == input_nodes + this->n_encodings_continuous_ + 2*this->n_encodings_discrete_);
    assert(n_loss_output_nodes == input_nodes + 2*this->n_encodings_continuous_ + this->n_encodings_discrete_ + this->labels_training_.size());
    assert(n_metric_output_nodes == input_nodes + this->labels_training_.size());
    assert(data_training.dimension(0) == features.size());
    assert(data_training.dimension(1) == labels_training.size());
    assert(this->n_encodings_continuous_ > 0);
    assert(this->n_encodings_discrete_ > 0);

    // Gaussian Sampler
    Eigen::Tensor<TensorT, 4> gaussian_samples = GaussianSampler<TensorT>(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);

    // Concrete Sampler
    Eigen::Tensor<TensorT, 4> categorical_samples = GumbelSampler<TensorT>(batch_size, memory_size, this->n_encodings_discrete_, n_epochs);
    TensorT inverse_tau = 3.0 / 2.0; //1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

    // Dummy data for the KL divergence losses
    Eigen::Tensor<TensorT, 4> KL_losses_continuous(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);
    KL_losses_continuous.setZero();
    Eigen::Tensor<TensorT, 4> KL_losses_discrete(batch_size, memory_size, this->n_encodings_discrete_, n_epochs);
    KL_losses_discrete.setZero();

    // initialize the Tensors
    this->input_data_training_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_training_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_training_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_training_.resize(batch_size, memory_size, n_epochs);

    // expand the training data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_training.dimension(1))), 1);
    const int over_expanded = data_training.dimension(1)*expansion_factor - batch_size * n_epochs;
    assert(batch_size * memory_size * n_epochs == data_training.dimension(1)*expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_training_expanded(data_training.dimension(0), data_training.dimension(1)*expansion_factor);
    Eigen::Tensor<std::string, 2> labels_training_expanded(data_training.dimension(1)*expansion_factor, 1);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = { 0, i*data_training.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_training.dimension(0), data_training.dimension(1) };
      data_training_expanded.slice(offset1, span1) = data_training;

      // Slices for the labels
      for (int j = 0; j < data_training.dimension(1); ++j) {
        labels_training_expanded(i*data_training.dimension(1) + j, 0) = labels_training.at(j);
      }
      //Eigen::array<Eigen::Index, 2> offset2 = { i*data_training.dimension(1), 0 };
      //Eigen::array<Eigen::Index, 2> span2 = { data_training.dimension(1), 1 };
      //Eigen::TensorMap<Eigen::Tensor<std::string, 2>> labels_2d(labels_training.data(), data_training.dimension(1), 1);
      //labels_training_expanded.slice(offset2, span2) = labels_2d;
    }

    // make the one-hot encodings       
    Eigen::Tensor<TensorT, 2> one_hot_vec = OneHotEncoder<std::string, TensorT>(labels_training_expanded, this->labels_training_);
    //Eigen::Tensor<TensorT, 2> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

    // optionally shuffle the data and labels
    if (shuffle_data_and_labels) {
      MakeShuffleMatrix<TensorT> shuffleMatrix(data_training.dimension(1) * expansion_factor, true);
      shuffleMatrix(data_training_expanded, true);
      shuffleMatrix.setShuffleMatrix(false); // re-orient for column with the same random indices
      shuffleMatrix(one_hot_vec, false);
    }

    // assign the input tensors
    auto data_training_expanded_4d = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), data_training.dimension(1)*expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_training.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 1,2,0,3 }));
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = data_training_expanded_4d;
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = gaussian_samples;
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = categorical_samples;
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_continuous_ + this->n_encodings_discrete_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = categorical_samples.constant(inverse_tau);

    //// Check that values of the data and input tensors are correctly aligned
    //Eigen::Tensor<TensorT, 1> data_training_head = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
    //  Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    //Eigen::Tensor<TensorT, 1> data_training_tail = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
    //  Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), data_training.dimension(1)*expansion_factor - over_expanded })
    //).slice(Eigen::array<Eigen::Index, 2>({ 0, batch_size * memory_size * n_epochs - 1 }),
    //  Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    //Eigen::Tensor<TensorT, 1> input_training_head = this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ 1, 1, data_training.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    //Eigen::Tensor<TensorT, 1> input_training_tail = this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    //  Eigen::array<Eigen::Index, 4>({ 1, 1, data_training.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    //std::cout << "data_training_head\n" << data_training_head << std::endl;
    //std::cout << "data_training_tail\n" << data_training_tail << std::endl;
    //for (int i = 0; i < data_training.dimension(0); ++i) {
    //  assert(data_training_head(i) == input_training_head(i));
    //  assert(data_training_tail(i) == input_training_tail(i));
    //}

    // assign the loss tensors
    auto one_hot_vec_4d = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(1) * expansion_factor - over_expanded, one_hot_vec.dimension(1) })
    ).reshape(Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_epochs, int(labels_training_.size()) })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 0,1,3,2 }));
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = data_training_expanded_4d;
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + 2 * this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = KL_losses_discrete;
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + 2 * this->n_encodings_continuous_ + this->n_encodings_discrete_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_training_.size()), n_epochs })) = one_hot_vec_4d;

    // assign the metric tensors
    this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = data_training_expanded_4d;
    this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_training_.size()), n_epochs })) = one_hot_vec_4d;
  }
  template<typename TensorT>
  inline void MetabolomicsReconstructionDataSimulator<TensorT>::makeValidationDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_validation, const std::vector<std::string>& labels_validation, const int & n_epochs, const int & batch_size, const int & memory_size, const int & n_input_nodes, const int & n_loss_output_nodes, const int & n_metric_output_nodes, const bool& shuffle_data_and_labels)
  {
    // infer the input sizes
    const int input_nodes = data_validation.dimension(0);
    assert(n_input_nodes == input_nodes + this->n_encodings_continuous_ + 2 * this->n_encodings_discrete_);
    assert(n_loss_output_nodes == input_nodes + 2 * this->n_encodings_continuous_ + this->n_encodings_discrete_ + this->labels_validation_.size());
    assert(n_metric_output_nodes == input_nodes + this->labels_validation_.size());
    assert(data_validation.dimension(0) == features.size());
    assert(data_validation.dimension(1) == labels_validation.size());
    assert(this->n_encodings_continuous_ > 0);
    assert(this->n_encodings_discrete_ > 0);

    // Gaussian Sampler
    Eigen::Tensor<TensorT, 4> gaussian_samples = GaussianSampler<TensorT>(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);

    // Concrete Sampler
    Eigen::Tensor<TensorT, 4> categorical_samples = GumbelSampler<TensorT>(batch_size, memory_size, this->n_encodings_discrete_, n_epochs);
    TensorT inverse_tau = 3.0 / 2.0; //1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

    // Dummy data for the KL divergence losses
    Eigen::Tensor<TensorT, 4> KL_losses_continuous(batch_size, memory_size, this->n_encodings_continuous_, n_epochs);
    KL_losses_continuous.setZero();
    Eigen::Tensor<TensorT, 4> KL_losses_discrete(batch_size, memory_size, this->n_encodings_discrete_, n_epochs);
    KL_losses_discrete.setZero();

    // initialize the Tensors
    this->input_data_validation_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_validation_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_validation_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_validation_.resize(batch_size, memory_size, n_epochs);

    // expand the validation data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_validation.dimension(1))), 1);
    const int over_expanded = data_validation.dimension(1) * expansion_factor - batch_size * n_epochs;
    assert(batch_size * memory_size * n_epochs == data_validation.dimension(1) * expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_validation_expanded(data_validation.dimension(0), data_validation.dimension(1) * expansion_factor);
    Eigen::Tensor<std::string, 2> labels_validation_expanded(data_validation.dimension(1) * expansion_factor, 1);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = { 0, i * data_validation.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_validation.dimension(0), data_validation.dimension(1) };
      data_validation_expanded.slice(offset1, span1) = data_validation;

      // Slices for the labels
      for (int j = 0; j < data_validation.dimension(1); ++j) {
        labels_validation_expanded(i * data_validation.dimension(1) + j, 0) = labels_validation.at(j);
      }
      //Eigen::array<Eigen::Index, 2> offset2 = { i*data_validation.dimension(1), 0 };
      //Eigen::array<Eigen::Index, 2> span2 = { data_validation.dimension(1), 1 };
      //Eigen::TensorMap<Eigen::Tensor<std::string, 2>> labels_2d(labels_validation.data(), data_validation.dimension(1), 1);
      //labels_validation_expanded.slice(offset2, span2) = labels_2d;
    }

    // make the one-hot encodings       
    Eigen::Tensor<TensorT, 2> one_hot_vec = OneHotEncoder<std::string, TensorT>(labels_validation_expanded, this->labels_validation_);
    //Eigen::Tensor<TensorT, 2> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

    // optionally shuffle the data and labels
    if (shuffle_data_and_labels) {
      MakeShuffleMatrix<TensorT> shuffleMatrix(data_validation.dimension(1) * expansion_factor, true);
      shuffleMatrix(data_validation_expanded, true);
      shuffleMatrix.setShuffleMatrix(false); // re-orient for column with the same random indices
      shuffleMatrix(one_hot_vec, false);
    }

    // assign the input tensors
    auto data_validation_expanded_4d = data_validation_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_validation.dimension(0), data_validation.dimension(1) * expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_validation.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 1,2,0,3 }));
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = data_validation_expanded_4d;
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = gaussian_samples;
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = categorical_samples;
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_continuous_ + this->n_encodings_discrete_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = categorical_samples.constant(inverse_tau);

    //// Check that values of the data and input tensors are correctly aligned
    //Eigen::Tensor<TensorT, 1> data_validation_head = data_validation_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
    //  Eigen::array<Eigen::Index, 2>({ data_validation.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_validation.dimension(0) }));
    //Eigen::Tensor<TensorT, 1> data_validation_tail = data_validation_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
    //  Eigen::array<Eigen::Index, 2>({ data_validation.dimension(0), data_validation.dimension(1)*expansion_factor - over_expanded })
    //).slice(Eigen::array<Eigen::Index, 2>({ 0, batch_size * memory_size * n_epochs - 1 }),
    //  Eigen::array<Eigen::Index, 2>({ data_validation.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_validation.dimension(0) }));
    //Eigen::Tensor<TensorT, 1> input_validation_head = this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
    //  Eigen::array<Eigen::Index, 4>({ 1, 1, data_validation.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_validation.dimension(0) }));
    //Eigen::Tensor<TensorT, 1> input_validation_tail = this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
    //  Eigen::array<Eigen::Index, 4>({ 1, 1, data_validation.dimension(0), 1 })
    //).reshape(Eigen::array<Eigen::Index, 1>({ data_validation.dimension(0) }));
    //std::cout << "data_validation_head\n" << data_validation_head << std::endl;
    //std::cout << "data_validation_tail\n" << data_validation_tail << std::endl;
    //for (int i = 0; i < data_validation.dimension(0); ++i) {
    //  assert(data_validation_head(i) == input_validation_head(i));
    //  assert(data_validation_tail(i) == input_validation_tail(i));
    //}

    // assign the loss tensors
    auto one_hot_vec_4d = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_validation.dimension(1) * expansion_factor - over_expanded, one_hot_vec.dimension(1) })
    ).reshape(Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_epochs, int(labels_validation_.size()) })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 0,1,3,2 }));
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = data_validation_expanded_4d;
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_continuous_, n_epochs })) = KL_losses_continuous;
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + 2 * this->n_encodings_continuous_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_discrete_, n_epochs })) = KL_losses_discrete;
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + 2 * this->n_encodings_continuous_ + this->n_encodings_discrete_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_validation_.size()), n_epochs })) = one_hot_vec_4d;

    // assign the metric tensors
    this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = data_validation_expanded_4d;
    this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_validation_.size()), n_epochs })) = one_hot_vec_4d;
  }
  template<typename TensorT>
  inline void MetabolomicsReconstructionDataSimulator<TensorT>::readAndProcessMetabolomicsTrainingAndValidationData(int & n_reaction_ids_training, int & n_labels_training, int & n_component_group_names_training, int & n_reaction_ids_validation, int & n_labels_validation, int & n_component_group_names_validation, const std::string & biochem_rxns_filename, const std::string & metabo_data_filename_train, const std::string & meta_data_filename_train, const std::string & metabo_data_filename_test, const std::string & meta_data_filename_test, 
    const bool & use_concentrations, const bool & use_MARs, const bool & sample_values, const bool & iter_values, const bool & fill_sampling, const bool & fill_mean, const bool & fill_zero, const bool & apply_fold_change, const std::string & fold_change_ref, const TensorT & fold_change_log_base, const bool & offline_linear_scale_input, const bool & offline_log_transform_input, const bool & offline_standardize_input, const bool & online_linear_scale_input, const bool & online_log_transform_input, const bool & online_standardize_input, 
    int & n_reps_per_sample, const bool& randomize_sample_group_names, const bool& shuffle_data_and_labels, const int & n_epochs, const int & batch_size, const int & memory_size)
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
        n_component_group_names_training + this->n_encodings_continuous_ + 2*this->n_encodings_discrete_, n_component_group_names_training + 2 * this->n_encodings_continuous_ + this->n_encodings_discrete_ + this->labels_training_.size(), n_component_group_names_training + this->labels_training_.size(), shuffle_data_and_labels);
      this->makeValidationDataForCache(features_validation, data_validation, labels_validation, n_epochs, batch_size, memory_size,
        n_component_group_names_training + this->n_encodings_continuous_ + 2*this->n_encodings_discrete_, n_component_group_names_training + 2 * this->n_encodings_continuous_ + this->n_encodings_discrete_ + this->labels_validation_.size(), n_component_group_names_training + this->labels_validation_.size(), shuffle_data_and_labels);
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
        n_reaction_ids_validation + this->n_encodings_continuous_ + 2 * this->n_encodings_discrete_, n_reaction_ids_validation + 2 * this->n_encodings_continuous_ + this->n_encodings_discrete_ + this->labels_training_.size(), n_reaction_ids_validation + this->labels_training_.size(), shuffle_data_and_labels);
      this->makeValidationDataForCache(features_validation, data_validation, labels_validation, n_epochs, batch_size, memory_size,
        n_reaction_ids_validation + this->n_encodings_continuous_ + 2 * this->n_encodings_discrete_, n_reaction_ids_validation + 2*this->n_encodings_continuous_ + this->n_encodings_discrete_ + this->labels_validation_.size(), n_reaction_ids_validation + this->labels_validation_.size(), shuffle_data_and_labels);
    }
  }
  template<typename TensorT>
  inline void MetabolomicsReconstructionDataSimulator<TensorT>::getNonRandomizedEncoderTrainingInputFromCacheByLabel(const std::string& label, const int& n_features, Eigen::Tensor<TensorT, 4>& input_data)
  {
    // Determine the offset of the label
    auto l = std::find(this->labels_training_.begin(), this->labels_training_.end(), label);

    // Assign the output data based on the offset
    if (l != std::end(this->labels_training_)) {
      int index = std::distance(this->labels_training_.begin(), l);
      int n_reps_per_sample = this->input_data_training_.dimension(0) * this->input_data_training_.dimension(3) / this->labels_training_.size();
      int n_epochs_span = n_reps_per_sample / this->input_data_training_.dimension(0);
      input_data.resize(this->input_data_training_.dimension(0), this->input_data_training_.dimension(1), n_features, n_epochs_span);
      input_data = this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, n_epochs_span * index }),
        Eigen::array<Eigen::Index, 4>({ this->input_data_training_.dimension(0), this->input_data_training_.dimension(1), n_features, n_epochs_span }));
    }
  }
  template<typename TensorT>
  inline void MetabolomicsReconstructionDataSimulator<TensorT>::getNonRandomizedDecoderTrainingOutputFromCacheByLabel(const std::string& label, const int& n_features, Eigen::Tensor<TensorT, 4>& output_data)
  {
    // Determine the offset of the label
    auto l = std::find(this->labels_training_.begin(), this->labels_training_.end(), label);

    // Assign the output data based on the offset
    if (l != std::end(this->labels_training_)) {
      int index = std::distance(this->labels_training_.begin(), l);
      int n_reps_per_sample = this->loss_output_data_training_.dimension(0) * this->loss_output_data_training_.dimension(3) / this->labels_training_.size();
      int n_epochs_span = n_reps_per_sample / this->loss_output_data_training_.dimension(0);
      output_data.resize(this->loss_output_data_training_.dimension(0), this->loss_output_data_training_.dimension(1), n_features, n_epochs_span);
      output_data = this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, n_epochs_span * index }),
        Eigen::array<Eigen::Index, 4>({ this->loss_output_data_training_.dimension(0), this->loss_output_data_training_.dimension(1), n_features, n_epochs_span }));
    }
  }
}

#endif //EVONET_METABOLOMICSRECONSTRUCTIONDATASIMULATOR_H