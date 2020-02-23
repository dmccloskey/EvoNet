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
    void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
      if (this->simulate_MARs_) this->simulateDataReconMARs_(input_data, Eigen::Tensor<TensorT, 3>(), Eigen::Tensor<TensorT, 3>(), time_steps, this->use_train_for_eval_, true);
      else this->simulateDataReconSampleConcs_(input_data, Eigen::Tensor<TensorT, 3>(), Eigen::Tensor<TensorT, 3>(), time_steps, this->use_train_for_eval_, true);
    }
    void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
      if (this->use_cache_) {
        this->getTrainingDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
      }
      else {
        if (this->simulate_MARs_) this->simulateDataReconMARs_(input_data, loss_output_data, metric_output_data, time_steps, true, false);
        else this->simulateDataReconSampleConcs_(input_data, loss_output_data, metric_output_data, time_steps, true, false);
      }
    }
    void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
      if (this->use_cache_) {
        this->getValidationDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
      }
      else {
        if (this->simulate_MARs_) this->simulateDataReconMARs_(input_data, loss_output_data, metric_output_data, time_steps, false, false);
        else this->simulateDataReconSampleConcs_(input_data, loss_output_data, metric_output_data, time_steps, false, false);
      }
    }

    std::vector<std::string> features_; ///< dim 0
    std::vector<std::string> samples_; ///< dim 1
    int n_replicates_; ///< dim 2
    bool use_train_for_eval_ = true;
    bool randomize_replicates_ = false; 
  protected:
    void getTrainingDataFromCache_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
      // Check that we have not exceeded the number of cached training data
      if (this->n_epochs_training_ >= this->input_data_training_.dimension(3))
        this->n_epochs_training_ = 0;

      // Copy over the training data
      input_data = this->input_data_training_.chip(this->n_epochs_training_, 3);
      loss_output_data = this->loss_output_data_training_.chip(this->n_epochs_training_, 3);
      metric_output_data = this->metric_output_data_training_.chip(this->n_epochs_training_, 3);
      time_steps = this->time_steps_training_.chip(this->n_epochs_training_, 2);

      // Increment the iterator
      this->n_epochs_training_++;
    }
    void getValidationDataFromCache_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
      // Check that we have not exceeded the number of cached validation data
      if (this->n_epochs_validation_ >= this->input_data_validation_.dimension(3))
        this->n_epochs_validation_ = 0;

      // Copy over the validation data
      input_data = this->input_data_validation_.chip(this->n_epochs_validation_, 3);
      loss_output_data = this->loss_output_data_validation_.chip(this->n_epochs_validation_, 3);
      metric_output_data = this->metric_output_data_validation_.chip(this->n_epochs_validation_, 3);
      time_steps = this->time_steps_validation_.chip(this->n_epochs_validation_, 2);

      // Increment the iterator
      this->n_epochs_validation_++;
    }
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
	};
}

#endif //SMARTPEAK_BIOCHEMICALDATASIMULATOR_H