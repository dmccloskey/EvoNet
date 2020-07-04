/**TODO:  Add copyright*/

#ifndef SMARTPEAK_BIOCHEMICALDATASIMULATOR_H
#define SMARTPEAK_BIOCHEMICALDATASIMULATOR_H

// .h
#include <SmartPeak/simulator/BiochemicalReaction.h>
#include <SmartPeak/simulator/DataSimulator.h>

namespace SmartPeak
{
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
  inline void BiochemicalDataSimulator<TensorT>::getTrainingDataFromCache_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    // Check that we have not exceeded the number of cached training data
    if (this->n_epochs_training_ >= this->input_data_training_.dimension(3))
      this->n_epochs_training_ = 0;

    // Copy over the training data
    input_data = this->input_data_training_.chip(this->n_epochs_training_, 3);
    loss_output_data = this->loss_output_data_training_.chip(this->n_epochs_training_, 3);
    metric_output_data = this->metric_output_data_training_.chip(this->n_epochs_training_, 3);
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
    input_data = this->input_data_validation_.chip(this->n_epochs_validation_, 3);
    loss_output_data = this->loss_output_data_validation_.chip(this->n_epochs_validation_, 3);
    metric_output_data = this->metric_output_data_validation_.chip(this->n_epochs_validation_, 3);
    //time_steps = this->time_steps_validation_.chip(this->n_epochs_validation_, 2);

    // Increment the iterator
    this->n_epochs_validation_++;
  }
}

#endif //SMARTPEAK_BIOCHEMICALDATASIMULATOR_H