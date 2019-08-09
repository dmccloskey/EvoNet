/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/simulator/BiochemicalReaction.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Other extended classes
template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerDefaultDevice<TensorT>
{};

template<typename TensorT>
class MetDataSim : public DataSimulator<TensorT>
{
public:
  void simulateDataReconMARs_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, 
    const bool& train, const bool& eval)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_.reaction_ids_.size();
    else
      n_input_pixels = this->model_validation_.reaction_ids_.size();

    assert(n_loss_output_nodes == n_input_pixels + 2 * n_encodings_);
    assert(n_metric_output_nodes % n_input_pixels == 0);
    assert(n_input_nodes == n_input_pixels + n_encodings_);

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> d{ 0.0f, 1.0f };

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_.sample_group_names_);

        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          if (nodes_iter < n_input_pixels) {
            TensorT value;
            if (train)
              value = this->model_training_.calculateMAR(
                this->model_training_.metabolomicsData_.at(sample_group_name),
                this->model_training_.biochemicalReactions_.at(this->model_training_.reaction_ids_.at(nodes_iter)));
            else
              value = this->model_validation_.calculateMAR(
                this->model_validation_.metabolomicsData_.at(sample_group_name),
                this->model_validation_.biochemicalReactions_.at(this->model_validation_.reaction_ids_.at(nodes_iter)));
            input_data(batch_iter, memory_iter, nodes_iter) = value;
            loss_output_data(batch_iter, memory_iter, nodes_iter) = 0;
            metric_output_data(batch_iter, memory_iter, nodes_iter) = 0;
          }
          if (nodes_iter < n_encodings_) {
            TensorT random_value = 0;
            if (train) {
              random_value = d(gen);
            }
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = random_value; // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
        }
      }
    }
  }
  void simulateDataReconSampleConcs_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, 
    const bool& train, const bool& eval)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_.component_group_names_.size();
    else
      n_input_pixels = this->model_validation_.component_group_names_.size();

    if (eval) {
      assert(n_input_nodes == n_input_pixels);
    }
    else {
      assert(n_loss_output_nodes == n_input_pixels + 2 * n_encodings_);
      assert(n_metric_output_nodes % n_input_pixels == 0);
      assert(n_input_nodes == n_input_pixels + n_encodings_);
    }

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> d{ 0.0f, 1.0f };

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_.sample_group_names_);

        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          if (nodes_iter < n_input_pixels) {
            TensorT value;
            if (train)
              value = this->model_training_.getRandomConcentration(
                this->model_training_.metabolomicsData_.at(sample_group_name),
                this->model_training_.component_group_names_.at(nodes_iter));
            else
              value = this->model_validation_.getRandomConcentration(
                this->model_validation_.metabolomicsData_.at(sample_group_name),
                this->model_validation_.component_group_names_.at(nodes_iter));
            input_data(batch_iter, memory_iter, nodes_iter) = value;
            if (!eval) {
              loss_output_data(batch_iter, memory_iter, nodes_iter) = 0;
              metric_output_data(batch_iter, memory_iter, nodes_iter) = 0;
            }
          }
          if (nodes_iter < n_encodings_ && !eval) {
            TensorT random_value = 0;
            if (train) {
              random_value = d(gen);
            }
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = random_value; // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
        }
      }
    }
  }

  BiochemicalReactionModel<TensorT> model_training_;
  BiochemicalReactionModel<TensorT> model_validation_;
  int n_encodings_;
  bool sample_concs_ = false;
  bool simulate_MARs_ = true;
};

template<typename TensorT>
class MetDataSimReconstruction : public MetDataSim<TensorT>
{
public:
  void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (this->simulate_MARs_) this->simulateDataReconMARs_(input_data, Eigen::Tensor<TensorT, 3>(), Eigen::Tensor<TensorT, 3>(), time_steps, this->use_train_for_eval_, true);
    else this->simulateDataReconSampleConcs_(input_data, Eigen::Tensor<TensorT, 3>(), Eigen::Tensor<TensorT, 3>(), time_steps, this->use_train_for_eval_, true);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (this->use_cache_) {
      this->getTrainingDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
    }
    else {
      if (this->simulate_MARs_) this->simulateDataReconMARs_(input_data, loss_output_data, metric_output_data, time_steps, true, false);
      else this->simulateDataReconSampleConcs_(input_data, loss_output_data, metric_output_data, time_steps, true, false);
    }
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (this->use_cache_) {
      this->getValidationDataFromCache_(input_data, loss_output_data, metric_output_data, time_steps);
    }
    else {
      if (this->simulate_MARs_) this->simulateDataReconMARs_(input_data, loss_output_data, metric_output_data, time_steps, false, false);
      else this->simulateDataReconSampleConcs_(input_data, loss_output_data, metric_output_data, time_steps, false, false);
    }
  }

  bool use_cache_ = false;
  bool use_train_for_eval_ = true;

  void makeTrainingDataCache(const Eigen::Tensor<TensorT, 4>& input_data, 
    const int& n_epochs, const int& batch_size, const int& memory_size,
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) {

    // infer the input sizes
    const int input_batch_size = input_data.dimension(0);
    const int input_memory_size = input_data.dimension(1);
    const int input_nodes = input_data.dimension(2);
    assert(input_batch_size == batch_size);
    assert(input_memory_size == memory_size);
    assert(n_input_nodes == input_nodes + this->n_encodings_);
    assert(n_loss_output_nodes == input_nodes + this->n_encodings_ * 2);
    assert(n_metric_output_nodes == input_nodes);

    // Gaussian Sampler
    Eigen::Tensor<TensorT, 4> gaussian_samples = GaussianSampler<TensorT>(batch_size, memory_size, this->n_encodings_, n_epochs);

    // Dummy data for the KL divergence losses
    Eigen::Tensor<TensorT, 4> KL_losses(batch_size, memory_size, this->n_encodings_, n_epochs);
    KL_losses.setZero();

    // initialize the Tensors
    this->input_data_training_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_training_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_training_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_training_.resize(batch_size, memory_size, n_epochs);

    // assign the input tensors
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = input_data;
    this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_, n_epochs })) = gaussian_samples;

    // assign the loss tensors
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = input_data;
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_, n_epochs })) = KL_losses;
    this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_, n_epochs })) = KL_losses;

    // assign the metric tensors
    this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = input_data;
  }

  void makeValidationDataCache(const Eigen::Tensor<TensorT, 4>& input_data,
    const int& n_epochs, const int& batch_size, const int& memory_size,
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) {

    // infer the input sizes
    const int input_batch_size = input_data.dimension(0);
    const int input_memory_size = input_data.dimension(1);
    const int input_nodes = input_data.dimension(2);
    assert(input_batch_size == batch_size);
    assert(input_memory_size == memory_size);
    assert(n_input_nodes == input_nodes + this->n_encodings_);
    assert(n_loss_output_nodes == input_nodes + this->n_encodings_ * 2);
    assert(n_metric_output_nodes == input_nodes);

    // Gaussian Sampler
    Eigen::Tensor<TensorT, 4> gaussian_samples = GaussianSampler<TensorT>(batch_size, memory_size, this->n_encodings_, n_epochs);

    // Dummy data for the KL divergence losses
    Eigen::Tensor<TensorT, 4> KL_losses(batch_size, memory_size, this->n_encodings_, n_epochs);
    KL_losses.setZero();

    // initialize the Tensors
    this->input_data_validation_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_validation_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_validation_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_validation_.resize(batch_size, memory_size, n_epochs);

    // assign the input tensors
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = input_data;
    this->input_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_, n_epochs })) = gaussian_samples;

    // assign the loss tensors
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = input_data;
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_, n_epochs })) = KL_losses;
    this->loss_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, input_nodes + this->n_encodings_, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, this->n_encodings_, n_epochs })) = KL_losses;

    // assign the metric tensors
    this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, input_nodes, n_epochs })) = input_data;
  }

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

template<typename TensorT>
class MetDataReconstruction : public DataSimulator<TensorT>
{
public:
  void simulateDataReconMARs_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_.reaction_ids_.size();
    else
      n_input_pixels = this->model_validation_.reaction_ids_.size();

    assert(n_loss_output_nodes == n_input_pixels + 2 * n_encodings_);
    assert(n_metric_output_nodes % n_input_pixels == 0);
    assert(n_input_nodes == n_input_pixels + n_encodings_);

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> d{ 0.0f, 1.0f };

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_.sample_group_names_);

        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          if (nodes_iter < n_input_pixels) {
            TensorT value;
            if (train)
              value = this->model_training_.calculateMAR(
                this->model_training_.metabolomicsData_.at(sample_group_name),
                this->model_training_.biochemicalReactions_.at(this->model_training_.reaction_ids_.at(nodes_iter)));
            else
              value = this->model_validation_.calculateMAR(
                this->model_validation_.metabolomicsData_.at(sample_group_name),
                this->model_validation_.biochemicalReactions_.at(this->model_validation_.reaction_ids_.at(nodes_iter)));
            input_data(batch_iter, memory_iter, nodes_iter) = value;
            loss_output_data(batch_iter, memory_iter, nodes_iter) = 0;
            metric_output_data(batch_iter, memory_iter, nodes_iter) = 0;
          }
          if (nodes_iter < n_encodings_) {
            TensorT random_value = 0;
            if (train) {
              random_value = d(gen);
            }
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = random_value; // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
        }
      }
    }
  }
  void simulateDataReconSampleConcs_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_.component_group_names_.size();
    else
      n_input_pixels = this->model_validation_.component_group_names_.size();

    assert(n_loss_output_nodes == n_input_pixels + 2 * n_encodings_);
    assert(n_metric_output_nodes % n_input_pixels == 0);
    assert(n_input_nodes == n_input_pixels + n_encodings_);

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> d{ 0.0f, 1.0f };

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_.sample_group_names_);

        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          if (nodes_iter < n_input_pixels) {
            TensorT value;
            if (train)
              value = this->model_training_.getRandomConcentration(
                this->model_training_.metabolomicsData_.at(sample_group_name),
                this->model_training_.component_group_names_.at(nodes_iter));
            else
              value = this->model_validation_.getRandomConcentration(
                this->model_validation_.metabolomicsData_.at(sample_group_name),
                this->model_validation_.component_group_names_.at(nodes_iter));
            input_data(batch_iter, memory_iter, nodes_iter) = value;
            loss_output_data(batch_iter, memory_iter, nodes_iter) = 0;
            metric_output_data(batch_iter, memory_iter, nodes_iter) = 0;
          }
          if (nodes_iter < n_encodings_) {
            TensorT random_value = 0;
            if (train) {
              random_value = d(gen);
            }
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = random_value; // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
        }
      }
    }
  }
  void simulateTrainingData_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (simulate_MARs_) simulateDataReconMARs_(input_data, loss_output_data, metric_output_data, time_steps, true);
    else simulateDataReconSampleConcs_(input_data, loss_output_data, metric_output_data, time_steps, true);
  }
  void simulateValidationData_(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (simulate_MARs_) simulateDataReconMARs_(input_data, loss_output_data, metric_output_data, time_steps, false);
    else simulateDataReconSampleConcs_(input_data, loss_output_data, metric_output_data, time_steps, false);
  }
  void makeTrainingDataCache(const int& n_epochs, const int& batch_size, const int& memory_size, 
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) {

    // initialize the Tensors
    this->input_data_training_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_training_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_training_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->time_steps_training_.resize(batch_size, memory_size, n_epochs);

    // generate the training values
    for (int i = 0; i < n_epochs; ++i) {
      simulateTrainingData_(this->input_data_training_.chip(0, 3), this->loss_output_data_training_.chip(0, 3), this->metric_output_data_training_.chip(0, 3), this->time_steps_training_.chip(0, 2));
    }
  };
  void makeValidationDataCache(const int& n_epochs, const int& batch_size, const int& memory_size,
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) {

    // initialize the Tensors
    this->input_data_validation_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_validation_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_validation_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->time_steps_validation_.resize(batch_size, memory_size, n_epochs);

    // generate the validation values
    for (int i = 0; i < n_epochs; ++i) {
      simulateValidationData_(this->input_data_validation_.chip(0, 3), this->loss_output_data_validation_.chip(0, 3), this->metric_output_data_validation_.chip(0, 3), this->time_steps_validation_.chip(0, 2));
    }
  };  
  
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
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
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
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

  BiochemicalReactionModel<TensorT> model_training_;
  BiochemicalReactionModel<TensorT> model_validation_;
  int n_encodings_;
  bool sample_concs_ = false;
  bool simulate_MARs_ = true;

protected:
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

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief Fully connected variational reconstruction model
    where the expected output is 0
  */
  void makeModelFCVAE_1(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const int& n_encodings, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, const bool& add_norm = false,
    const int& n_en_hidden_0 = 64, const int& n_en_hidden_1 = 64, const int& n_en_hidden_2 = 0, const int& n_de_hidden_0 = 64, const int& n_de_hidden_1 = 64, const int& n_de_hidden_2 = 0) {
    model.setId(0);
    model.setName("VAE");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, node_names_input, linear_scale_input, log_transform_input, standardize_input);

    // Add the encoding layers
    std::vector<std::string> node_names = node_names_input;
    if (n_en_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_en_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_en_hidden_1,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_1) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_en_hidden_2,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_2) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN2-Norm-gain", "EN2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }

    // Add the mu and log var layers
    std::vector<std::string> node_names_mu = model_builder.addFullyConnected(model, "Mu", "Mu", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
    std::vector<std::string> node_names_logvar = model_builder.addFullyConnected(model, "LogVar", "LogVar", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_mu)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_logvar)
      model.nodes_.at(node_name)->setType(NodeType::output);

    // Add the Variational Encoding layer
    node_names = model_builder.addGaussianEncoding(model, "Encoding", "Encoding", node_names_mu, node_names_logvar, true);

    // Add the decoding layers
    if (n_de_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_de_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE0-Norm-gain", "DE0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_de_hidden_1,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_1) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE1-Norm-gain", "DE1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names, n_de_hidden_2,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_2) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE2-Norm-gain", "DE2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }

    // Add the final output layer
    node_names = model_builder.addFullyConnected(model, "DE-Output", "DE-Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);

    // Add the final output layer
    std::vector<std::string> node_names_output;
    node_names_output = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
      std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Subtract out the pre-processed input data to test against all 0's
    model_builder.addSinglyConnected(model, "Output", node_names_input, node_names_output,
      std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(-1)),
      std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0f, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_output)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief Fully connected variational reconstruction model
    where the expected output is the input
  */
  void makeModelFCVAE_2(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const int& n_encodings, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, const bool& add_norm = false,
    const int& n_en_hidden_0 = 64, const int& n_en_hidden_1 = 64, const int& n_en_hidden_2 = 0, const int& n_de_hidden_0 = 64, const int& n_de_hidden_1 = 64, const int& n_de_hidden_2 = 0) {
    model.setId(0);
    model.setName("VAE");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, node_names_input, linear_scale_input, log_transform_input, standardize_input);

    // Add the encoding layers
    std::vector<std::string> node_names = node_names_input;
    if (n_en_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_en_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_en_hidden_1,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_1) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_en_hidden_2,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_2) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN2-Norm-gain", "EN2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }

    // Add the mu and log var layers
    std::vector<std::string> node_names_mu = model_builder.addFullyConnected(model, "Mu", "Mu", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
    std::vector<std::string> node_names_logvar = model_builder.addFullyConnected(model, "LogVar", "LogVar", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_mu)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_logvar)
      model.nodes_.at(node_name)->setType(NodeType::output);

    // Add the Variational Encoding layer
    node_names = model_builder.addGaussianEncoding(model, "Encoding", "Encoding", node_names_mu, node_names_logvar, true);

    // Add the decoding layers
    if (n_de_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_de_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE0-Norm-gain", "DE0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_de_hidden_1,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_1) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE1-Norm-gain", "DE1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names, n_de_hidden_2,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_2) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE2-Norm-gain", "DE2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }

    // Add the final output layer
    std::vector<std::string> node_names_output;
    node_names_output = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_output)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief Input normalization network
  */
  void makeModelNormalization(Model<TensorT>& model, const int& n_inputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input) {
    model.setId(0);
    model.setName("Normalization");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, node_names, linear_scale_input, log_transform_input, standardize_input);

    // Add the final output layer
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_inputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
      std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief Add data preprocessing steps
  */
  void addDataPreproccessingSteps(Model<TensorT>& model, std::vector<std::string>& node_names, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input) {
    ModelBuilder<TensorT> model_builder;
    // Data pre-processing steps
    if (log_transform_input) {
      node_names = model_builder.addSinglyConnected(model, "LogScaleInput", "LogScaleInput", node_names, node_names.size(),
        std::shared_ptr<ActivationOp<TensorT>>(new LogOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LogGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, false, true);
    }
    if (linear_scale_input) {
      node_names = model_builder.addLinearScale(model, "LinearScaleInput", "LinearScaleInput", node_names, 0, 1, true);
    }
    if (standardize_input) {
      node_names = model_builder.addNormalization(model, "StandardizeInput", "StandardizeInput", node_names, true);
    }
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    // Check point the model every 1000 epochs
    if (n_epochs % 200 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false);
      // save the model weights
      WeightFile<float> weight_data;
      weight_data.storeWeightValuesCsv(model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model.weights_);
      //// save the model and tensors to binary
      //ModelFile<TensorT> data;
      //data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      //ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      //interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const TensorT & model_error_train, const TensorT & model_error_test,
    const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test)
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedPredictedEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedPredictedEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedPredictedEpoch(true);
      model_interpreter.getModelResults(model, true, false, false);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_train_headers = { "Train_Error" };
    std::vector<std::string> log_test_headers = { "Test_Error" };
    std::vector<TensorT> log_train_values = { model_error_train };
    std::vector<TensorT> log_test_values = { model_error_test };
    int metric_iter = 0;
    for (const std::string& metric_name : this->metric_names_) {
      log_train_headers.push_back(metric_name);
      log_test_headers.push_back(metric_name);
      log_train_values.push_back(model_metrics_train(metric_iter));
      log_test_values.push_back(model_metrics_test(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, output_nodes, {});
  }
};

/// Script to run the reconstruction network
void main_reconstruction(const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
  const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
  const bool& make_model = true, const bool& simulate_MARs = true, const bool& sample_concs = true, const bool& make_data_caches = false)
{
  const int n_threads = 1;

  // define the data simulator
  BiochemicalReactionModel<float> reaction_model;
  MetDataSimReconstruction<float> metabolomics_data;

  // Read in the training and validation data

  // Training data
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename_train);
  reaction_model.readMetaData(meta_data_filename_train);
  reaction_model.findComponentGroupNames();
  if (simulate_MARs) {
    reaction_model.findMARs();
    reaction_model.findMARs(true, false);
    reaction_model.findMARs(false, true);
    reaction_model.removeRedundantMARs();
  }
  reaction_model.findLabels();
  metabolomics_data.model_training_ = reaction_model;

  // Validation data
  reaction_model.clear();
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename_test);
  reaction_model.readMetaData(meta_data_filename_test);
  reaction_model.findComponentGroupNames();
  if (simulate_MARs) {
    reaction_model.findMARs();
    reaction_model.findMARs(true, false);
    reaction_model.findMARs(false, true);
    reaction_model.removeRedundantMARs();
  }
  reaction_model.findLabels();
  metabolomics_data.model_validation_ = reaction_model;
  metabolomics_data.simulate_MARs_ = simulate_MARs;
  metabolomics_data.sample_concs_ = sample_concs;

  // Checks for the training and validation data
  assert(metabolomics_data.model_validation_.reaction_ids_.size() == metabolomics_data.model_training_.reaction_ids_.size());
  assert(metabolomics_data.model_validation_.labels_.size() == metabolomics_data.model_training_.labels_.size());
  assert(metabolomics_data.model_validation_.component_group_names_.size() == metabolomics_data.model_training_.component_group_names_.size());

  // Balance the sample group names
  metabolomics_data.model_training_.sample_group_names_ = {
  "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04",
  "Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP", "Evo04Evo01EP",
  "Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd", "Evo04gnd",
  "Evo04gndEvo01EP", "Evo04gndEvo01EP", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04gndEvo02EP", "Evo04gndEvo02EP",
  "Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB", "Evo04sdhCB",
  "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04sdhCBEvo02EP", "Evo04sdhCBEvo02EP",
  "Evo04pgi", "Evo04pgi", "Evo04pgi", "Evo04pgi", "Evo04pgi", "Evo04pgi",
  "Evo04pgiEvo01EP", "Evo04pgiEvo02EP", "Evo04pgiEvo03EP", "Evo04pgiEvo04EP", "Evo04pgiEvo05EP", "Evo04pgiEvo06EP",
  "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04ptsHIcrr",
  "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04ptsHIcrrEvo03EP", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04ptsHIcrrEvo03EP",
  "Evo04tpiA", "Evo04tpiA", "Evo04tpiA", "Evo04tpiA", "Evo04tpiA", "Evo04tpiA",
  "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP", "Evo04tpiAEvo03EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP", "Evo04tpiAEvo03EP"
  };
  //metabolomics_data.model_training_.sample_group_names_ = {
  //"S01_D01_PLT_25C_22hr","S01_D01_PLT_25C_6.5hr","S01_D01_PLT_25C_0hr","S01_D02_PLT_25C_22hr","S01_D02_PLT_25C_6.5hr","S01_D02_PLT_25C_0hr","S01_D05_PLT_25C_0hr","S01_D05_PLT_25C_22hr","S01_D05_PLT_25C_6.5hr","S01_D01_PLT_37C_22hr","S01_D02_PLT_37C_22hr","S01_D05_PLT_37C_22hr"
  //};
  //metabolomics_data.model_validation_.sample_group_names_ = {
  //"S02_D01_PLT_25C_22hr","S02_D01_PLT_25C_6.5hr","S02_D01_PLT_25C_0hr","S02_D02_PLT_25C_22hr","S02_D02_PLT_25C_6.5hr","S02_D02_PLT_25C_0hr","S02_D05_PLT_25C_0hr","S02_D05_PLT_25C_22hr","S02_D05_PLT_25C_6.5hr","S02_D01_PLT_37C_22hr","S02_D02_PLT_37C_22hr","S02_D05_PLT_37C_22hr"
  //};

  // Define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = n_input_nodes;
  const int encoding_size = 16;
  metabolomics_data.n_encodings_ = encoding_size;
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;

  // Make the input nodes
  std::vector<std::string> met_input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
    met_input_nodes.push_back(name);
  }

  // Make the encoding nodes and add them to the input
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Encoding_%012d-Sampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the reconstruction nodes
  for (int i = 0; i < n_output_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // Make the mu nodes
  std::vector<std::string> encoding_nodes_mu;
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Mu_%012d", i);
    std::string name(name_char);
    encoding_nodes_mu.push_back(name);
  }

  // Make the encoding nodes
  std::vector<std::string> encoding_nodes_logvar;
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "LogVar_%012d", i);
    std::string name(name_char);
    encoding_nodes_logvar.push_back(name);
  }

  std::vector<std::string> output_nodes_normalization;
  if (make_data_caches) {
    // Make the normalization nodes
    for (int i = 0; i < n_input_nodes; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes_normalization.push_back(name);
    }
  }

  // Define resources for the model interpreters
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }

  // define the model trainer
  ModelTrainerExt<float> model_trainer;

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // initialize the model replicator
  ModelReplicatorExt<float> model_replicator;

  // Generate the training/validation data caches 
  if (make_data_caches) {
    std::cout << "Making the data caches..." << std::endl;
    // Make the normalization model
    Model<float> model_normalization;
    model_trainer.makeModelNormalization(model_normalization, n_input_nodes, true, false, false); // normalization type 1

    // Set the model trainer parameters for normalizing the data
    model_trainer.setBatchSize(64);
    model_trainer.setMemorySize(1);
    model_trainer.setNEpochsTraining(5000);
    model_trainer.setNEpochsEvaluation(5000);
    model_trainer.setVerbosityLevel(1);
    model_trainer.setLogging(true, false, false);
    model_trainer.setFindCycles(false);
    model_trainer.setFastInterpreter(true);
    model_trainer.setPreserveOoO(true);

    // Apply the normalization model and make the caches
    model_trainer.setLossOutputNodes({ output_nodes_normalization });
    const int n_loss_output_nodes = output_nodes.size() + encoding_nodes_mu.size() + encoding_nodes_logvar.size();
    const int n_metric_output_nodes = output_nodes.size();
    std::cout << "Making the data cache for training..." << std::endl;
    metabolomics_data.use_train_for_eval_ = true;
    Eigen::Tensor<float, 4> input_data_training = model_trainer.evaluateModel(model_normalization, metabolomics_data, met_input_nodes, model_logger, model_interpreters.front());
    metabolomics_data.makeTrainingDataCache(input_data_training, model_trainer.getNEpochsTraining(), model_trainer.getBatchSize(), model_trainer.getMemorySize(),
      input_nodes.size(), n_loss_output_nodes, n_metric_output_nodes);
    std::cout << "Making the data cache for validation..." << std::endl;
    metabolomics_data.use_train_for_eval_ = false;
    Eigen::Tensor<float, 4> input_data_validation = model_trainer.evaluateModel(model_normalization, metabolomics_data, met_input_nodes, model_logger, model_interpreters.front());
    metabolomics_data.makeValidationDataCache(input_data_validation, model_trainer.getNEpochsTraining(), model_trainer.getBatchSize(), model_trainer.getMemorySize(),
      input_nodes.size(), n_loss_output_nodes, n_metric_output_nodes);
    metabolomics_data.use_cache_ = true;
  }

  // make the models
  Model<float> model_FCVAE;
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    //model_trainer.makeModelFCVAE_1(model_FCVAE, n_input_nodes, n_output_nodes, encoding_size, true, false, false, false,
    //  64, 64, 0, 64, 64, 0); // normalization type 1
    model_trainer.makeModelFCVAE_2(model_FCVAE, n_input_nodes, n_output_nodes, encoding_size, true, false, false, false,
      64, 64, 0, 64, 64, 0); // normalization type 1
  }
  else {
    // TODO: load in the trained model
  }

  // Set the model trainer parameters for training
  model_trainer.setBatchSize(64);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(5000);
  model_trainer.setNEpochsEvaluation(5000);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({
    std::shared_ptr<LossFunctionOp<float>>(new MAPELossOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceMuLossOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceLogVarOp<float>(1e-6, 1.0)) });
  model_trainer.setLossFunctionGrads({
    std::shared_ptr<LossFunctionGradOp<float>>(new MAPELossGradOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceMuLossGradOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceLogVarGradOp<float>(1e-6, 1.0)) });
  model_trainer.setLossOutputNodes({ output_nodes, encoding_nodes_mu, encoding_nodes_logvar });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new MAEOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes });
  model_trainer.setMetricNames({ "MAE" });

  // Train the model
  std::cout << "Training the model..." << std::endl;
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model_FCVAE, metabolomics_data,
    input_nodes, model_logger, model_interpreters.front());
}

void main_loadBinaryModelAndStoreWeightsCsv(const std::string& model_filename) {
  // load the binarized model
  Model<float> model;
  ModelFile<float> model_file;
  model_file.loadModelBinary(model_filename, model);

  // save the model weights
  WeightFile<float> data;
  data.storeWeightValuesCsv(model.getName() + "_weights.csv", model.weights_);
}

// Main
int main(int argc, char** argv)
{
  // Set the data directories
  //const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
  const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
  //const std::string data_dir = "/home/user/Data/";

  // Make the filenames
  const std::string biochem_rxns_filename = data_dir + "iJO1366.csv";
  //const std::string biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
  //const std::string biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";

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

  //// Platelets
  //const std::string metabo_data_filename_train = data_dir + "PLT_timeCourse_Metabolomics_train.csv";
  //const std::string meta_data_filename_train = data_dir + "PLT_timeCourse_MetaData_train.csv";
  //const std::string metabo_data_filename_test = data_dir + "PLT_timeCourse_Metabolomics_test.csv";
  //const std::string meta_data_filename_test = data_dir + "PLT_timeCourse_MetaData_test.csv";

  main_reconstruction(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test, true, false, true, true);

  return 0;
}