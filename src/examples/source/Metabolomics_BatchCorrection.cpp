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
class MetDataSimClassification : public DataSimulator<TensorT>
{
public:
  void simulateDataClassMARs(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    if (train)
      assert(n_input_nodes == this->model_training_.reaction_ids_.size());
    else
      assert(n_input_nodes == this->model_validation_.reaction_ids_.size());

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_.sample_group_names_);

        // assign the input data
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          //input_data(batch_iter, memory_iter, nodes_iter) = conc_data(nodes_iter);
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
        }

        // convert the label to a one hot vector        
        Eigen::Tensor<TensorT, 1> one_hot_vec((int)this->model_training_.labels_.size());
        if (train)
          one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_training_.metaData_.at(sample_group_name).condition, this->model_training_.labels_);
        else
          one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_validation_.metaData_.at(sample_group_name).condition, this->model_validation_.labels_);
        Eigen::Tensor<TensorT, 1> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

        // MSE or LogLoss only
        size_t n_labels;
        if (train)
          n_labels = this->model_training_.labels_.size();
        else
          n_labels = this->model_validation_.labels_.size();
        for (int nodes_iter = 0; nodes_iter < n_labels; ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = one_hot_vec(nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter + (int)n_labels) = one_hot_vec(nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = one_hot_vec(nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter + (int)n_labels) = one_hot_vec(nodes_iter);
        }
      }
    }

    // update the time_steps
    time_steps.setConstant(1.0f);
  }
  void simulateDataClassSampleConcs(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    if (train)
      assert(n_input_nodes == this->model_training_.component_group_names_.size());
    else
      assert(n_input_nodes == this->model_validation_.component_group_names_.size());

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_.sample_group_names_);

        // assign the input data
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
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
        }

        // convert the label to a one hot vector      
        Eigen::Tensor<TensorT, 1> one_hot_vec((int)this->model_training_.labels_.size());
        if (train)
          one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_training_.metaData_.at(sample_group_name).condition, this->model_training_.labels_);
        else
          one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_validation_.metaData_.at(sample_group_name).condition, this->model_validation_.labels_);
        Eigen::Tensor<TensorT, 1> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

        // MSE or LogLoss only
        size_t n_labels;
        if (train)
          n_labels = this->model_training_.labels_.size();
        else
          n_labels = this->model_validation_.labels_.size();
        for (int nodes_iter = 0; nodes_iter < n_labels; ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = one_hot_vec_smoothed(nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter + (int)n_labels) = one_hot_vec(nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = one_hot_vec_smoothed(nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter + (int)n_labels) = one_hot_vec_smoothed(nodes_iter);
        }
      }
    }

    // update the time_steps
    time_steps.setConstant(1.0f);
  }
  void simulateDataClassConcs(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    if (train)
      assert(n_input_nodes == this->model_training_.component_group_names_.size());
    else
      assert(n_input_nodes == this->model_validation_.component_group_names_.size());

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        int max_replicates = 0;
        if (train) {
          sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);
          max_replicates = this->model_training_.metabolomicsData_.at(sample_group_name).at(this->model_training_.component_group_names_.at(0)).size();
        }
        else {
          sample_group_name = selectRandomElement(this->model_validation_.sample_group_names_);
          max_replicates = this->model_validation_.metabolomicsData_.at(sample_group_name).at(this->model_validation_.component_group_names_.at(0)).size();
        }

        // pick a random replicate
        std::vector<int> replicates;
        for (int i = 0; i < max_replicates; ++i) {
          replicates.push_back(i);
        }
        const int replicate = selectRandomElement(replicates);

        // assign the input data
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          TensorT value;
          if (train)
            value = this->model_training_.metabolomicsData_.at(sample_group_name).at(this->model_training_.component_group_names_.at(nodes_iter)).at(replicate).calculated_concentration;
          else
            value = this->model_validation_.metabolomicsData_.at(sample_group_name).at(this->model_validation_.component_group_names_.at(nodes_iter)).at(replicate).calculated_concentration;
          input_data(batch_iter, memory_iter, nodes_iter) = value;
        }

        // convert the label to a one hot vector      
        Eigen::Tensor<TensorT, 1> one_hot_vec((int)this->model_training_.labels_.size());
        if (train)
          one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_training_.metaData_.at(sample_group_name).condition, this->model_training_.labels_);
        else
          one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_validation_.metaData_.at(sample_group_name).condition, this->model_validation_.labels_);
        Eigen::Tensor<TensorT, 1> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

        // MSE or LogLoss only
        size_t n_labels;
        if (train)
          n_labels = this->model_training_.labels_.size();
        else
          n_labels = this->model_validation_.labels_.size();
        for (int nodes_iter = 0; nodes_iter < n_labels; ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = one_hot_vec_smoothed(nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter + (int)n_labels) = one_hot_vec(nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = one_hot_vec_smoothed(nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter + (int)n_labels) = one_hot_vec_smoothed(nodes_iter);
        }
      }
    }

    // update the time_steps
    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (simulate_MARs_) simulateDataClassMARs(input_data, loss_output_data, metric_output_data, time_steps, true);
    else if (sample_concs_) simulateDataClassSampleConcs(input_data, loss_output_data, metric_output_data, time_steps, true);
    else simulateDataClassConcs(input_data, loss_output_data, metric_output_data, time_steps, true);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (simulate_MARs_) simulateDataClassMARs(input_data, loss_output_data, metric_output_data, time_steps, false);
    else if (sample_concs_) simulateDataClassSampleConcs(input_data, loss_output_data, metric_output_data, time_steps, false);
    else simulateDataClassConcs(input_data, loss_output_data, metric_output_data, time_steps, false);
  }

  BiochemicalReactionModel<TensorT> model_training_;
  BiochemicalReactionModel<TensorT> model_validation_;
  bool sample_concs_ = false;
  bool simulate_MARs_ = true;
};

template<typename TensorT>
class MetDataSimBatchCorrection : public DataSimulator<TensorT>
{
public:
  void simulateDataReconMARs(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_batch_1_.reaction_ids_.size();
    else
      n_input_pixels = this->model_validation_batch_1_.reaction_ids_.size();

    assert(n_loss_output_nodes == n_input_pixels);
    assert(n_metric_output_nodes % n_input_pixels == 0);
    assert(n_input_nodes == 2 * n_input_pixels);

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_batch_1_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_batch_2_.sample_group_names_);

        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          TensorT value_batch_1, value_batch_2;
          if (train) {
            value_batch_1 = this->model_training_batch_1_.calculateMAR(
              this->model_training_batch_1_.metabolomicsData_.at(sample_group_name),
              this->model_training_batch_1_.biochemicalReactions_.at(this->model_training_batch_1_.reaction_ids_.at(nodes_iter)));
            value_batch_2 = this->model_training_batch_2_.calculateMAR(
              this->model_training_batch_2_.metabolomicsData_.at(sample_group_name),
              this->model_training_batch_2_.biochemicalReactions_.at(this->model_training_batch_2_.reaction_ids_.at(nodes_iter)));
          }
          else {
            value_batch_1 = this->model_validation_batch_1_.calculateMAR(
              this->model_validation_batch_1_.metabolomicsData_.at(sample_group_name),
              this->model_validation_batch_1_.biochemicalReactions_.at(this->model_validation_batch_1_.reaction_ids_.at(nodes_iter)));
            value_batch_2 = this->model_validation_batch_2_.calculateMAR(
              this->model_validation_batch_2_.metabolomicsData_.at(sample_group_name),
              this->model_validation_batch_2_.biochemicalReactions_.at(this->model_validation_batch_2_.reaction_ids_.at(nodes_iter)));
          }
          input_data(batch_iter, memory_iter, nodes_iter) = value_batch_1;
          input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = value_batch_2;
          loss_output_data(batch_iter, memory_iter, nodes_iter) = 0;
          metric_output_data(batch_iter, memory_iter, nodes_iter) = 0;
        }
      }
    }
  }
  void simulateDataReconSampleConcs(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_loss_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_batch_1_.component_group_names_.size();
    else
      n_input_pixels = this->model_validation_batch_1_.component_group_names_.size();

    assert(n_loss_output_nodes == n_input_pixels);
    assert(n_metric_output_nodes % n_input_pixels == 0);
    assert(n_input_nodes == 2 * n_input_pixels);

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // pick a random sample group name
        std::string sample_group_name;
        if (train)
          sample_group_name = selectRandomElement(this->model_training_batch_1_.sample_group_names_);
        else
          sample_group_name = selectRandomElement(this->model_validation_batch_1_.sample_group_names_);

        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          TensorT value_batch_1, value_batch_2;
          if (train) {
            value_batch_1 = this->model_training_batch_1_.getRandomConcentration(
              this->model_training_batch_1_.metabolomicsData_.at(sample_group_name),
              this->model_training_batch_1_.component_group_names_.at(nodes_iter));
            value_batch_2 = this->model_training_batch_2_.getRandomConcentration(
              this->model_training_batch_2_.metabolomicsData_.at(sample_group_name),
              this->model_training_batch_2_.component_group_names_.at(nodes_iter));
          }
          else {
            value_batch_1 = this->model_validation_batch_1_.getRandomConcentration(
              this->model_validation_batch_1_.metabolomicsData_.at(sample_group_name),
              this->model_validation_batch_1_.component_group_names_.at(nodes_iter));
            value_batch_2 = this->model_validation_batch_2_.getRandomConcentration(
              this->model_validation_batch_2_.metabolomicsData_.at(sample_group_name),
              this->model_validation_batch_2_.component_group_names_.at(nodes_iter));
          }
          input_data(batch_iter, memory_iter, nodes_iter) = value_batch_1;
          input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = value_batch_2;
          loss_output_data(batch_iter, memory_iter, nodes_iter) = 0;
          metric_output_data(batch_iter, memory_iter, nodes_iter) = 0;
        }
      }
    }
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (simulate_MARs_) simulateDataReconMARs(input_data, loss_output_data, metric_output_data, time_steps, true);
    else simulateDataReconSampleConcs(input_data, loss_output_data, metric_output_data, time_steps, true);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {
    if (simulate_MARs_) simulateDataReconMARs(input_data, loss_output_data, metric_output_data, time_steps, false);
    else simulateDataReconSampleConcs(input_data, loss_output_data, metric_output_data, time_steps, false);
  }

  BiochemicalReactionModel<TensorT> model_training_batch_1_;
  BiochemicalReactionModel<TensorT> model_training_batch_2_;
  BiochemicalReactionModel<TensorT> model_validation_batch_1_;
  BiochemicalReactionModel<TensorT> model_validation_batch_2_;
  bool sample_concs_ = false;
  bool simulate_MARs_ = true;
};

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief Fully connected auto-encoder model
  */
  void makeModelBatchCorrectionAE(Model<TensorT>& model, const int& n_inputs, const int& n_encodings, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, const bool& add_norm = true,
    const int& n_en_hidden_0 = 64, const int& n_en_hidden_1 = 0, const int& n_en_hidden_2 = 0,
    const int& n_de_hidden_0 = 64, const int& n_de_hidden_1 = 0, const int& n_de_hidden_2 = 0) {
    model.setId(0);
    model.setName("AE");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, "Input", node_names_input, linear_scale_input, log_transform_input, standardize_input);

    // Add the encoding layers
    std::vector<std::string> node_names = node_names_input;
    if (n_en_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_en_hidden_0,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_0) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_en_hidden_1,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_1) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_en_hidden_2,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_2) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN2-Norm-gain", "EN2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }

    // Add the mu and log var layers
    //std::vector<std::string> node_names_mu = model_builder.addFullyConnected(model, "Mu", "Mu", node_names, n_encodings, // FIXME
    std::vector<std::string> node_names_mu = model_builder.addSinglyConnected(model, "Mu", "Mu", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);  // FIXME
    //std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Add a link between the mu and the encoding
    node_names = model_builder.addSinglyConnected(model, "Encoding", "Encoding", node_names_mu, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Add the decoding layers
    if (n_de_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_de_hidden_0,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_0) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE0-Norm-gain", "DE0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_de_hidden_1,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_1) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE1-Norm-gain", "DE1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names, n_de_hidden_2,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_2) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE2-Norm-gain", "DE2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }

    // Add the final output layer
    //node_names = model_builder.addFullyConnected(model, "Output-AE", "Output-AE", node_names, n_inputs, // FIXME
    node_names = model_builder.addSinglyConnected(model, "Output-AE", "Output-AE", node_names, n_inputs,
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_inputs) / 2, 1)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Add the inputs
    std::vector<std::string> node_names_expected = model_builder.addInputNodes(model, "Expected", "Expected", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, "Expected", node_names_expected, linear_scale_input, log_transform_input, standardize_input);

    // Subtract out the pre-processed input data to test against all 0's
    model_builder.addSinglyConnected(model, "Output-AE", node_names_expected, node_names,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(-1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  /*
  @brief Fully connected auto-encoder model
  */
  void makeModelBatchCorrectionClassifier(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const int& n_encodings, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, const bool& add_norm = true,
    const int& n_en_hidden_0 = 64, const int& n_en_hidden_1 = 0, const int& n_en_hidden_2 = 0,
    const int& n_de_hidden_0 = 64, const int& n_de_hidden_1 = 0, const int& n_de_hidden_2 = 0,
    const int& n_hidden_0 = 32, const int& n_hidden_1 = 0, const int& n_hidden_2 = 0) {
    model.setId(0);
    model.setName("AE");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, "Input", node_names_input, linear_scale_input, log_transform_input, standardize_input);

    // Add the encoding layers
    std::vector<std::string> node_names = node_names_input;
    if (n_en_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_en_hidden_0,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_0) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_en_hidden_1,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_1) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_en_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_en_hidden_2,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_en_hidden_2) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN2-Norm-gain", "EN2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }

    // Add the mu and log var layers
    //std::vector<std::string> node_names_mu = model_builder.addFullyConnected(model, "Mu", "Mu", node_names, n_encodings, //FIXME
    std::vector<std::string> node_names_mu = model_builder.addSinglyConnected(model, "Mu", "Mu", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);  // FIXME
    //std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Add a link between the mu and the encoding
    node_names = model_builder.addSinglyConnected(model, "Encoding", "Encoding", node_names_mu, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Add the decoding layers
    if (n_de_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_de_hidden_0,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_0) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE0-Norm-gain", "DE0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_de_hidden_1,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_1) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE1-Norm-gain", "DE1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_de_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names, n_de_hidden_2,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_de_hidden_2) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE2-Norm-gain", "DE2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }

    // Add the AE Output layer
    //node_names = model_builder.addFullyConnected(model, "Output-AE", "Output-AE", node_names, n_inputs, // FIXME
    node_names = model_builder.addSinglyConnected(model, "Output-AE", "Output-AE", node_names, n_inputs,
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_inputs) / 2, 1)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Add the classifier hidden layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_hidden_0,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC0-Norm", "FC0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC0-Norm-gain", "FC0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_hidden_1,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_1) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC1-Norm", "FC1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC1-Norm-gain", "FC1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC2", "FC2", node_names, n_hidden_2,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_2) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC2-Norm", "FC2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC2-Norm-gain", "FC2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  /*
  @brief Fully connected classifier
  */
  void makeModelFCClass(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, const bool& add_norm = true,
    const int& n_hidden_0 = 32, const int& n_hidden_1 = 0, const int& n_hidden_2 = 0) {
    model.setId(0);
    model.setName("Classifier");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, "Input", node_names, linear_scale_input, log_transform_input, standardize_input);

    // Add the hidden layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_hidden_0,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC0-Norm", "FC0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC0-Norm-gain", "FC0-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_hidden_1,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_1) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC1-Norm", "FC1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC1-Norm-gain", "FC1-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC2", "FC2", node_names, n_hidden_2,
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        //std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
        //std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_2) / 2, 1)),
        std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC2-Norm", "FC2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC2-Norm-gain", "FC2-Norm-gain", node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
          std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
          std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()),
      std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
      std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
      std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
      std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
      std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  /*
  @brief Add data preprocessing steps
  */
  void addDataPreproccessingSteps(Model<TensorT>& model, const std::string& module_name, std::vector<std::string>& node_names, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input) {
    ModelBuilder<TensorT> model_builder;
    // Data pre-processing steps
    if (log_transform_input) {
      std::string name = "LogScale" + module_name;
      node_names = model_builder.addSinglyConnected(model, name, name, node_names, node_names.size(),
        std::shared_ptr<ActivationOp<TensorT>>(new LogOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LogGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, true);
    }
    if (linear_scale_input) {
      std::string name = "LinearScale" + module_name;
      node_names = model_builder.addLinearScale(model, name, name, node_names, 0, 1, true);
    }
    if (standardize_input) {
      std::string name = "Standardize" + module_name;
      node_names = model_builder.addNormalization(model, name, name, node_names, true);
    }
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    // Check point the model every 1000 epochs
    if (n_epochs % 500 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false, false);
      // save the model weights
      WeightFile<float> weight_data;
      weight_data.storeWeightValuesCsv(model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model.weights_);
      // save the model and tensors to binary
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const TensorT & model_error_train, const TensorT & model_error_test,
    const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test)
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
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
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values);
  }
  void validationModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const TensorT & model_error_train, const TensorT & model_error_test,
    const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test)
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
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
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values);
  }
};

/// Script to train the batch correction network
void main_batchCorrectionAE(const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train_batch_1, const std::string& metabo_data_filename_test_batch_1,
  const std::string& meta_data_filename_train_batch_1, const std::string& meta_data_filename_test_batch_1,
  const std::string& metabo_data_filename_train_batch_2, const std::string& metabo_data_filename_test_batch_2,
  const std::string& meta_data_filename_train_batch_2, const std::string& meta_data_filename_test_batch_2,
  bool make_model = true, bool simulate_MARs = true, bool sample_concs = true)
{
  // define the multithreading parameters
  const int n_threads = 1;

  // define the data simulator
  BiochemicalReactionModel<float> reaction_model;
  MetDataSimBatchCorrection<float> metabolomics_data;

  // Training data batch 1
  reaction_model.clear();
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename_train_batch_1);
  reaction_model.readMetaData(meta_data_filename_train_batch_1);
  reaction_model.findComponentGroupNames();
  if (simulate_MARs) {
    reaction_model.findMARs();
    reaction_model.findMARs(true, false);
    reaction_model.findMARs(false, true);
    reaction_model.removeRedundantMARs();
  }
  reaction_model.findLabels();
  metabolomics_data.model_training_batch_1_ = reaction_model;

  // Training data batch 2
  reaction_model.clear();
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename_train_batch_2);
  reaction_model.readMetaData(meta_data_filename_train_batch_2);
  reaction_model.findComponentGroupNames();
  if (simulate_MARs) {
    reaction_model.findMARs();
    reaction_model.findMARs(true, false);
    reaction_model.findMARs(false, true);
    reaction_model.removeRedundantMARs();
  }
  reaction_model.findLabels();
  metabolomics_data.model_training_batch_2_ = reaction_model;

  // Validation data batch 1
  reaction_model.clear();
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename_test_batch_1);
  reaction_model.readMetaData(meta_data_filename_test_batch_1);
  reaction_model.findComponentGroupNames();
  if (simulate_MARs) {
    reaction_model.findMARs();
    reaction_model.findMARs(true, false);
    reaction_model.findMARs(false, true);
    reaction_model.removeRedundantMARs();
  }
  reaction_model.findLabels();
  metabolomics_data.model_validation_batch_1_ = reaction_model;

  // Validation data batch 1
  reaction_model.clear();
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename_test_batch_2);
  reaction_model.readMetaData(meta_data_filename_test_batch_2);
  reaction_model.findComponentGroupNames();
  if (simulate_MARs) {
    reaction_model.findMARs();
    reaction_model.findMARs(true, false);
    reaction_model.findMARs(false, true);
    reaction_model.removeRedundantMARs();
  }
  reaction_model.findLabels();
  metabolomics_data.model_validation_batch_2_ = reaction_model;
  metabolomics_data.simulate_MARs_ = simulate_MARs;
  metabolomics_data.sample_concs_ = sample_concs;

  // Checks for the training and validation data
  assert(metabolomics_data.model_training_batch_1_.reaction_ids_.size() == metabolomics_data.model_training_batch_2_.reaction_ids_.size());
  assert(metabolomics_data.model_validation_batch_1_.reaction_ids_.size() == metabolomics_data.model_validation_batch_2_.reaction_ids_.size());
  assert(metabolomics_data.model_training_batch_1_.component_group_names_.size() == metabolomics_data.model_training_batch_2_.component_group_names_.size());
  assert(metabolomics_data.model_validation_batch_1_.component_group_names_.size() == metabolomics_data.model_validation_batch_2_.component_group_names_.size());

  // Define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = n_input_nodes;
  const int encoding_size = 64;

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Expected_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the reconstruction nodes
  std::vector<std::string> output_nodes_ae;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Output-AE_%012d", i);
    std::string name(name_char);
    output_nodes_ae.push_back(name);
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(64);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(100000);
  model_trainer.setNEpochsValidation(0);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({
    std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) });
  model_trainer.setLossFunctionGrads({
    std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0)) });
  model_trainer.setLossOutputNodes({ output_nodes_ae });
  model_trainer.setMetricFunctions({ std::make_shared<MAEOp<float>>(MAEOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes_ae });
  model_trainer.setMetricNames({ "MAE" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // initialize the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the model
  Model<float> model;
  if (make_model) {
    model_trainer.makeModelBatchCorrectionAE(model, n_input_nodes, encoding_size, true, false, false, false,
      0, 0, 0, 0, 0, 0); // normalization type 1
  }
  else {
    // TODO: load in the trained model
  }

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, metabolomics_data,
    input_nodes, model_logger, model_interpreters.front());
}

/// Script to evaluate the batch correction AE + classifier networks
void main_batchCorrectionClassification(const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& metabo_data_filename_test,
  const std::string& meta_data_filename_train, const std::string& meta_data_filename_test,
  const std::string& model_ae_weight_filename, const std::string& model_ae_classifier_weights_filename, const std::string& model_classifier_weight_filename,
  bool simulate_MARs = true, bool sample_concs = true)
{
  // define the multithreading parameters
  const int n_threads = 1;

  // define the data simulator
  BiochemicalReactionModel<float> reaction_model;
  MetDataSimClassification<float> metabolomics_data;

  // Training data
  reaction_model.clear();
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
  assert(metabolomics_data.model_training_.reaction_ids_.size() == metabolomics_data.model_validation_.reaction_ids_.size());
  assert(metabolomics_data.model_validation_.labels_.size() == metabolomics_data.model_training_.labels_.size());
  assert(metabolomics_data.model_training_.component_group_names_.size() == metabolomics_data.model_validation_.component_group_names_.size());

  // Define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = reaction_model.labels_.size();
  const int encoding_size = 88;

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
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

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(64);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsValidation(100);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, true, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({
    std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>()),
    std::make_shared<MSELossOp<float>>(MSELossOp<float>()) });
  model_trainer.setLossFunctionGrads({
    std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>()),
    std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>()) });
  model_trainer.setLossOutputNodes({
    output_nodes,
    output_nodes });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new AccuracyMCMicroOp<float>()), std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes, output_nodes });
  model_trainer.setMetricNames({ "AccuracyMCMicro", "PrecisionMCMicro" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // define the models
  Model<float> model_batch_correction_classifier, model_classifier;
  model_trainer.makeModelBatchCorrectionClassifier(model_batch_correction_classifier, n_input_nodes, n_output_nodes, encoding_size, true, false, false, false,
    0, 0, 0, 0, 0, 0, 32, 0, 0); // normalization type 1
  model_trainer.makeModelFCClass(model_classifier, n_input_nodes, n_output_nodes, true, false, false, false,
    32, 0, 0); // normalization type 1

  // read in the BatchCorrection AE weights
  WeightFile<float> data;
  data.loadWeightValuesCsv(model_ae_weight_filename, model_batch_correction_classifier.getWeightsMap());

  // read in the Classifier weights
  data.loadWeightValuesCsv(model_ae_classifier_weights_filename, model_batch_correction_classifier.getWeightsMap());
  data.loadWeightValuesCsv(model_classifier_weight_filename, model_classifier.getWeightsMap());

  // check that all weights were read in correctly
  for (auto& weight_map : model_batch_correction_classifier.getWeightsMap()) {
    if (weight_map.second->getInitWeight()) {
      std::cout << "Model " << model_batch_correction_classifier.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
    }
  }
  for (auto& weight_map : model_classifier.getWeightsMap()) {
    if (weight_map.second->getInitWeight()) {
      std::cout << "Model " << model_classifier.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
    }
  }

  // Validate the models
  std::pair<std::vector<float>, std::vector<float>> model_errors_BCClass = model_trainer.validateModel(model_batch_correction_classifier, metabolomics_data,
    input_nodes, model_logger, model_interpreters.front());
  std::pair<std::vector<float>, std::vector<float>> model_errors_Class = model_trainer.validateModel(model_classifier, metabolomics_data,
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
  const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "/home/user/Data/";

  // Make the filenames
  const std::string biochem_rxns_filename = data_dir + "iJO1366.csv";

  // IndustrialStrains0103 Batch correction filenames
  const std::string metabo_data_filename_train_batch_1 = data_dir + "IndustrialStrains0103_Metabolomics_train_batch_1.csv";
  const std::string metabo_data_filename_test_batch_1 = data_dir + "IndustrialStrains0103_Metabolomics_test_batch_1.csv";
  const std::string metabo_data_filename_train_batch_2 = data_dir + "IndustrialStrains0103_Metabolomics_train_batch_2.csv";
  const std::string metabo_data_filename_test_batch_2 = data_dir + "IndustrialStrains0103_Metabolomics_test_batch_2.csv";
  const std::string meta_data_filename_train_batch_1 = data_dir + "IndustrialStrains0103_MetaData_train_batch_1.csv";
  const std::string meta_data_filename_test_batch_1 = data_dir + "IndustrialStrains0103_MetaData_test_batch_1.csv";
  const std::string meta_data_filename_train_batch_2 = data_dir + "IndustrialStrains0103_MetaData_train_batch_2.csv";
  const std::string meta_data_filename_test_batch_2 = data_dir + "IndustrialStrains0103_MetaData_test_batch_2.csv";

  // Run the batch correction
  main_batchCorrectionAE(biochem_rxns_filename,
    metabo_data_filename_train_batch_1, metabo_data_filename_test_batch_1,
    meta_data_filename_train_batch_1, meta_data_filename_test_batch_1,
    metabo_data_filename_train_batch_2, metabo_data_filename_test_batch_2,
    meta_data_filename_train_batch_2, meta_data_filename_test_batch_2, true, false, true);

  // IndustrialStrains0103 classification filenames
  const std::string metabo_data_filename_train = data_dir + "IndustrialStrains0103_Metabolomics_train.csv";
  const std::string meta_data_filename_train = data_dir + "IndustrialStrains0103_MetaData_train.csv";
  const std::string metabo_data_filename_test = data_dir + "IndustrialStrains0103_Metabolomics_test.csv";
  const std::string meta_data_filename_test = data_dir + "IndustrialStrains0103_MetaData_test.csv";

  // Model filenames
  const std::string model_ae_weights_filename = data_dir + "TrainTestData/BatchCorrection/AE_weights.csv";
  const std::string model_ae_classifier_weights_filename = data_dir + "TrainTestData/BatchCorrection/AE_Classifier_weights.csv";
  const std::string model_classifier_weights_filename = data_dir + "TrainTestData/BatchCorrection/Classifier_weights.csv";

  // Run the classification
  main_batchCorrectionClassification(biochem_rxns_filename,
    metabo_data_filename_train, metabo_data_filename_test,
    meta_data_filename_train, meta_data_filename_test,
    model_ae_weights_filename, model_ae_classifier_weights_filename, model_classifier_weights_filename, false, true);

  return 0;
}