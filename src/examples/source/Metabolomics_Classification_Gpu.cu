/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>
#include <SmartPeak/simulator/BiochemicalReaction.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Other extended classes
template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{};

template<typename TensorT>
class MetDataSimClassification : public DataSimulator<TensorT>
{
public:
  void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    // NOTE: used for testing
    //std::string sample_group_name = sample_group_names_[0];
    //std::vector<float> mars;
    //for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
    //	float mar = calculateMAR(metabolomicsData_.at(sample_group_name),
    //		biochemicalReactions_.at(reaction_ids_[nodes_iter]));
    //	mars.push_back(mar);
    //	//std::cout << "OutputNode: "<<nodes_iter<< " = " << mar << std::endl;
    //}

    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

          // pick a random sample group name
          std::string sample_group_name = selectRandomElement(this->model_training_.sample_group_names_);

          for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->model_training_.calculateMAR(
              this->model_training_.metabolomicsData_.at(sample_group_name),
              this->model_training_.biochemicalReactions_.at(this->model_training_.reaction_ids_[nodes_iter]));
            //input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mars[nodes_iter]; // NOTE: used for testing
          }

          // convert the label to a one hot vector
          Eigen::Tensor<TensorT, 1> one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_training_.metaData_.at(sample_group_name).condition, this->model_training_.labels_);
          Eigen::Tensor<TensorT, 1> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

          //// MSE + LogLoss
          //for (int nodes_iter = 0; nodes_iter < n_output_nodes/2; ++nodes_iter) {
          //	output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
          //	output_data(batch_iter, memory_iter, nodes_iter + n_output_nodes/2, epochs_iter) = one_hot_vec(nodes_iter);
          //	//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec_smoothed(nodes_iter);
          //}

          // MSE or LogLoss only
          for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
            //output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec_smoothed(nodes_iter);
          }
        }
      }
    }

    // update the time_steps
    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {
    simulateData(input_data, output_data, time_steps);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {
    simulateData(input_data, output_data, time_steps);
  }
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

        //// generate the input data
        //Eigen::Tensor<TensorT, 1> conc_data(n_input_nodes);
        //for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
        //  conc_data(nodes_iter) = this->model_training_.calculateMAR(
        //    this->model_training_.metabolomicsData_.at(sample_group_name),
        //    this->model_training_.biochemicalReactions_.at(this->model_training_.reaction_ids_[nodes_iter]));
        //  //input_data(batch_iter, memory_iter, nodes_iter) = mars[nodes_iter]; // NOTE: used for testing
        //}

        //// pre-process the data
        //if (this->log_transform_input_) {
        //  conc_data = conc_data.log();
        //  //std::cout << "Log transformed: \n" << conc_data << std::endl;
        //}
        //if (this->linear_scale_input_) {
        //  Eigen::Tensor<TensorT, 0> min_v = conc_data.minimum();
        //  Eigen::Tensor<TensorT, 0> max_v = conc_data.maximum();
        //  conc_data = conc_data.unaryExpr(LinearScale<TensorT>(min_v(0), max_v(0), 0, 1));
        //  //std::cout << "Linear scaled: \n"<< conc_data << std::endl;
        //}
        //if (this->standardize_input_) {
        //  // Calculate the mean
        //  Eigen::Tensor<TensorT, 0> mean_v = conc_data.mean();
        //  //std::cout << "Mean" << mean_v << std::endl;
        //  // Calculate the variance
        //  auto residuals = conc_data - conc_data.constant(mean_v(0));
        //  auto ssr = residuals.pow(2).sum();
        //  Eigen::Tensor<TensorT, 0> var_v = ssr / ssr.constant(n_input_nodes - 1);
        //  //std::cout << "Var" << var_v << std::endl;
        //  // Standardize
        //  conc_data = residuals / conc_data.constant(var_v(0)).pow(0.5);
        //  //std::cout << "Standardized: \n" << conc_data << std::endl;
        //}

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
  //bool log_transform_input_ = false;
  //bool linear_scale_input_ = false;
  //bool standardize_input_ = false;
  bool sample_concs_ = false;
  bool simulate_MARs_ = true;
};

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
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
    this->addDataPreproccessingSteps(model, node_names, linear_scale_input, log_transform_input, standardize_input);

    // Add the hidden layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC0-Norm", "FC0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC0-Norm-gain", "FC0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_hidden_1,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_1) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC1-Norm", "FC1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC1-Norm-gain", "FC1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC2", "FC2", node_names, n_hidden_2,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_2) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "FC2-Norm", "FC2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "FC2-Norm-gain", "FC2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief CovNet classifier
  */
  void makeModelCovNetClass(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input,
    int n_hidden_0 = 64, int n_depth_1 = 32, int n_depth_2 = 2, int n_fc = 16, bool add_norm = false, bool specify_layers = false) {
    model.setId(0);
    model.setName("CovNet");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Data pre-processing steps
    this->addDataPreproccessingSteps(model, node_names_input, linear_scale_input, log_transform_input, standardize_input);

    // Add the hidden layers
    if (n_hidden_0 > 0) {
      node_names_input = model_builder.addFullyConnected(model, "FC0", "FC0", node_names_input, n_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_input.size() + n_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names_input = model_builder.addNormalization(model, "FC0-Norm", "FC0-Norm", node_names_input, true);
        node_names_input = model_builder.addSinglyConnected(model, "FC0-Norm-gain", "FC0-Norm-gain", node_names_input, node_names_input.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
    }

    // Add the first convolution -> max pool -> LeakyReLU layers
    std::vector<std::vector<std::string>> node_names_l0;
    for (size_t d = 0; d < n_depth_1; ++d) {
      std::vector<std::string> node_names;
      std::string conv_name = "Conv0-" + std::to_string(d);
      node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_input,
        sqrt(node_names_input.size()), sqrt(node_names_input.size()), 0, 0,
        2, 2, 1, 0, 0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(5, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        std::string norm_name = "Norm0-" + std::to_string(d);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, true);
        std::string gain_name = "Gain0-" + std::to_string(d);
        node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)),
          0.0, 0.0, true, true);
      }
      //std::string pool_name = "Pool0-" + std::to_string(d);
      //node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
      //  sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
      //  2, 2, 2, 0, 0,
      //  std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      //  std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      //  std::shared_ptr<IntegrationOp<TensorT>>(new MaxOp<float>()),
      //  std::shared_ptr<IntegrationErrorOp<TensorT>>(new MaxErrorOp<TensorT>()),
      //  std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new MaxWeightGradOp<TensorT>()),
      //  std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1.0)),
      //  std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      node_names_l0.push_back(node_names);
    }

    // Add the second convolution -> max pool -> LeakyReLU layers
    std::vector<std::vector<std::string>> node_names_l1;
    int l_cnt = 0;
    for (const std::vector<std::string> &node_names_l : node_names_l0) {
      for (size_t d = 0; d < n_depth_2; ++d) {
        std::vector<std::string> node_names;
        std::string conv_name = "Conv1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_l,
          sqrt(node_names_l.size()), sqrt(node_names_l.size()), 0, 0,
          2, 2, 1, 0, 0,
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(5, 1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, specify_layers);
        if (add_norm) {
          std::string norm_name = "Norm1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, true);
          std::string gain_name = "Gain1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
            std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
            std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
            std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
            std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
            std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
            std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
            std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)),
            0.0, 0.0, true, true);
        }
        //std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        //node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
        //  sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
        //  2, 2, 2, 0, 0,
        //  std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //  std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        //  std::shared_ptr<IntegrationOp<TensorT>>(new MaxOp<float>()),
        //  std::shared_ptr<IntegrationErrorOp<TensorT>>(new MaxErrorOp<TensorT>()),
        //  std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new MaxWeightGradOp<TensorT>()),
        //  std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1.0)),
        //  std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
        node_names_l1.push_back(node_names);
      }
      ++l_cnt;
    }

    // Linearize the node names
    std::vector<std::string> node_names;
    if (node_names_l1.size()) {
      for (const std::vector<std::string> &node_names_l : node_names_l1) {
        for (const std::string &node_name : node_names_l) {
          node_names.push_back(node_name);
        }
      }
    }
    else {
      for (const std::vector<std::string> &node_names_l : node_names_l0) {
        for (const std::string &node_name : node_names_l) {
          node_names.push_back(node_name);
        }
      }
    }

    // Add the FC layers
    node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_fc,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_fc, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "FC1-Norm", "FC1-Norm", node_names, true);
      node_names = model_builder.addSinglyConnected(model, "FC1-Norm-gain", "FC1-Norm-gain", node_names, node_names.size(),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)),
        0.0, 0.0, true, true);
    }
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10)), 0.0f, 0.0f, false, true);

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
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    // Check point the model every 1000 epochs
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false);
      // save the model weights
      WeightFile<float> weight_data;
      weight_data.storeWeightValuesCsv(model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model.weights_);
      //// save the model and interpreter in binary format
      //ModelFile<TensorT> data;
      //data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      //ModelInterpreterFileGpu<TensorT> interpreter_data;
      //interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
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
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values);
  }
};

/// Script to run the classification network
void main_classification(const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
  const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
  bool make_model = true, bool simulate_MARs = true, bool sample_concs = true)
{
  // define the population trainer parameters
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setNTop(3);
  population_trainer.setNRandom(3);
  population_trainer.setNReplicatesPerModel(3);
  population_trainer.setLogging(true);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the multithreading parameters
  const int n_hard_threads = std::thread::hardware_concurrency();
  //const int n_threads = n_hard_threads / 2; // the number of threads
  //char threads_cout[512];
  //sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
  //	n_hard_threads, 2);
  //std::cout << threads_cout;
  const int n_threads = 1;

  // define the data simulator
  BiochemicalReactionModel<float> reaction_model;
  MetDataSimClassification<float> metabolomics_data;
  std::string model_name = "0_Metabolomics";

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

  // define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = reaction_model.labels_.size();

  // define the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // define the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_output_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(64);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(10000);
  model_trainer.setNEpochsValidation(0);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({
    std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>()),
    std::shared_ptr<LossFunctionOp<float>>(new MSELossOp<float>()) });
  model_trainer.setLossFunctionGrads({
    std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>()),
    std::shared_ptr<LossFunctionGradOp<float>>(new MSELossGradOp<float>()) });
  model_trainer.setLossOutputNodes({
    output_nodes,
    output_nodes });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new AccuracyMCMicroOp<float>()), std::shared_ptr<MetricFunctionOp<float>>(new PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes, output_nodes });
  model_trainer.setMetricNames({ "AccuracyMCMicro", "PrecisionMCMicro" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // initialize the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  //std::vector<Model<float>> population;
  Model<float> model;
  if (make_model) {
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, false, false, false, false, 64, 64, 0); // normalization type 0
    model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, false, false, false, 64, 64, 0); // normalization type 1
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, false, true, false, 64, 64, 0); // normalization type 2
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, false, false, 64, 64, 0); // normalization type 3
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, true, false, 64, 64, 0); // normalization type 4

    //model_trainer.makeModelCovNetClass(model, n_input_nodes, n_output_nodes, true, true, false, 64, 16, 0, 32, false, true); // normalization type 3

    //population = { model };
  }
  else {
    // TODO
  }

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, metabolomics_data,
    input_nodes, model_logger, model_interpreters.front());

  //// Evolve the population
  //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
  //	population, model_trainer, model_interpreters, model_replicator, metabolomics_data, model_logger, population_logger, input_nodes);

  //PopulationTrainerFile<float> population_trainer_file;
  //population_trainer_file.storeModels(population, "Metabolomics");
  //population_trainer_file.storeModelValidations("MetabolomicsValidationErrors.csv", models_validation_errors_per_generation);
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

  main_classification(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test, true, false, true);
  return 0;
}