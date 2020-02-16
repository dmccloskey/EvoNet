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

        //// generate the input data
        //Eigen::Tensor<TensorT, 1> conc_data(n_input_nodes);
        //for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
        //  conc_data(nodes_iter) = this->model_training_.calculateMAR(
        //    this->model_training_.metabolomicsData_.at(sample_group_name),
        //    this->model_training_.biochemicalReactions_.at(this->model_training_.reaction_ids_.at(nodes_iter)));
        //  //input_data(batch_iter, memory_iter, nodes_iter) = mars.at(nodes_iter); // NOTE: used for testing
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
          // Get the value and assign the input
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

          // Determine the fold change (if enabled) and update the input
          if (this->use_fold_change_) {
            TensorT ref;
            if (train)
              ref = this->model_training_.calculateMAR(
                this->model_training_.metabolomicsData_.at(this->ref_fold_change_),
                this->model_training_.biochemicalReactions_.at(this->model_training_.reaction_ids_.at(nodes_iter)));
            else
              ref = this->model_validation_.calculateMAR(
                this->model_validation_.metabolomicsData_.at(this->ref_fold_change_),
                this->model_validation_.biochemicalReactions_.at(this->model_validation_.reaction_ids_.at(nodes_iter)));
            if (ref == 0 || value == 0) {
              input_data(batch_iter, memory_iter, nodes_iter) = 0;
            }
            // Log10 is used with the assumption that the larges fold change will be on an order of ~10
            // thus, all values will be between -1 and 1
            TensorT fold_change = minFunc(maxFunc(std::log(value / ref) / std::log(10), -1), 1);
            input_data(batch_iter, memory_iter, nodes_iter) = fold_change;
          }
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
          // Get the value and assign the input
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

          // Determine the fold change (if enabled) and update the input
          if (this->use_fold_change_) {
            TensorT ref;
            if (train)
              ref = this->model_training_.getRandomConcentration(
                this->model_training_.metabolomicsData_.at(this->ref_fold_change_),
                this->model_training_.component_group_names_.at(nodes_iter));
            else
              ref = this->model_validation_.getRandomConcentration(
                this->model_validation_.metabolomicsData_.at(this->ref_fold_change_),
                this->model_validation_.component_group_names_.at(nodes_iter));
            if (ref == 0 || value == 0) {
              input_data(batch_iter, memory_iter, nodes_iter) = 0;
            }
            // Log10 is used with the assumption that the larges fold change will be on an order of ~10
            // thus, all values will be between -1 and 1
            TensorT fold_change = minFunc(maxFunc(std::log(value / ref) / std::log(10), -1), 1);
            input_data(batch_iter, memory_iter, nodes_iter) = fold_change;
          }
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

          // Determine the fold change (if enabled) and update the input
          if (this->use_fold_change_) {
            TensorT ref;
            if (train)
              ref = this->model_training_.metabolomicsData_.at(this->ref_fold_change_).at(this->model_training_.component_group_names_.at(nodes_iter)).at(replicate).calculated_concentration;
            else
              ref = this->model_validation_.metabolomicsData_.at(this->ref_fold_change_).at(this->model_validation_.component_group_names_.at(nodes_iter)).at(replicate).calculated_concentration;
            if (ref == 0 || value == 0) {
              input_data(batch_iter, memory_iter, nodes_iter) = 0;
            }
            // Log10 is used with the assumption that the largest fold change will be on an order of ~10
            // thus, all values will be between -1 and 1
            TensorT fold_change = minFunc(maxFunc(std::log(value / ref) / std::log(10), -1), 1);
            input_data(batch_iter, memory_iter, nodes_iter) = fold_change;
          }
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
  bool use_fold_change_ = false;
  std::string ref_fold_change_ = "";
};

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief Fully connected classifier
  */
  void makeModelFCClass(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input,
    const int& n_hidden_0 = 32, const int& n_hidden_1 = 0, const int& n_hidden_2 = 0) {
    model.setId(0);
    model.setName("Classifier");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data online pre-processing steps
    this->addDataPreproccessingSteps(model, node_names, linear_scale_input, log_transform_input, standardize_input);

    // Define the activation based on `add_feature_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10));

    // Add the hidden layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, true);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, true);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC2", "FC2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, true);
    }

    // Add the final output layer
    node_names = model_builder.addFullyConnected(model, "FC-Output", "FC-Output", node_names, n_outputs,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);

    // Add the dummy output layer
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Manually define the output nodes
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
        std::make_shared<LogOp<TensorT>>(LogOp<TensorT>()),
        std::make_shared<LogGradOp<TensorT>>(LogGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, true);
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
    if (n_epochs % 1000 == 0) { // store on n_epochs == 0
    //if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;

      //// Save weights to .csv
      //data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model, false, false, true);

      // Save to binary
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test,
    const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, false);
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
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
};

/// Script to run the classification network
void main_classification(const std::string& data_dir, const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
  const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
  const bool& make_model = true, const bool& train_model = true, const int& norm_method = 0,
  const bool& simulate_MARs = true, const bool& sample_concs = true, const bool& use_fold_change = false, const std::string& fold_change_ref = "Evo04")
{
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
  metabolomics_data.use_fold_change_ = use_fold_change;
  metabolomics_data.ref_fold_change_ = "Evo04";

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
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterGpu<float> model_interpreter(model_resources);
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
  model_trainer.setLossFunctions({ std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-8, 1)) });
  model_trainer.setLossFunctionGrads({ std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-8, 1)) });
  model_trainer.setLossOutputNodes({ output_nodes });
  model_trainer.setMetricFunctions({ std::make_shared<AccuracyMCMicroOp<float>>(AccuracyMCMicroOp<float>()), std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>())
    });
  model_trainer.setMetricOutputNodes({ output_nodes, output_nodes });
  model_trainer.setMetricNames({ "AccuracyMCMicro", "PrecisionMCMicro" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  Model<float> model;
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    bool linear_scale_input = false;
    bool log_transform_input = false;
    bool standardize_input = false;
    if (norm_method == 0) {// normalization type 0 (No normalization)
    }
    else if (norm_method == 1) {// normalization type 1 (Projection)
      linear_scale_input = true;
    }
    else if (norm_method == 2) {// normalization type 2 (Standardization + Projection)
      linear_scale_input = true;
      standardize_input = true;
    }
    else if (norm_method == 3) {// normalization type 3 (Log transformation + Projection)
      linear_scale_input = true;
      log_transform_input = true;
    }
    else if (norm_method == 4) {// normalization type 4 (Log transformation + Standardization + Projection)
      linear_scale_input = true;
      log_transform_input = true;
      standardize_input = true;
    }
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, true, 64, 64, 0); // normalization type 4 (Log transformation + Standardization + Projection)
    model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, linear_scale_input, log_transform_input, standardize_input, 8, 0, 0);
  }
  else {
    // TODO
  }
  model.setName(data_dir + "Classifier"); //So that all output will be written to a specific directory

  // Train the model
  std::cout << "Training the model..." << std::endl;
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, metabolomics_data,
    input_nodes, model_logger, model_interpreter);
}

template<typename TensorT>
void calculateInputLayer0Correlation() {
  // Read in the weights
  Model<TensorT> model;

  // Calculate the average per node weight magnitude for the first layer
  const int n_fc_0 = 64;
  const int n_input = 92;
  Eigen::Tensor<TensorT, 1> weight_ave_values(n_input);
  for (int i = 0; i < n_input; ++i) {
    TensorT weight_sum = 0;
    for (int j = 0; j < n_fc_0; ++j) {
      char weight_name_char[512];
      sprintf(weight_name_char, "Input_%012d_to_FC%012d", i, j);
      std::string weight_name(weight_name_char);
      TensorT weight_value = model.getWeightsMap().at(weight_name)->getWeight();
      weight_sum += weight_value;
    }
    weight_ave_values(i) = weight_sum / n_fc_0;
  }

  // Generate a large sample of input

  // Calculate the Pearson Correlation
}

// Main
int main(int argc, char** argv)
{
  // Set the data directories
  //const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";

  // Initialize the defaults
  std::string data_dir = "";
  std::string biochem_rxns_filename = data_dir + "iJO1366.csv";
  std::string metabo_data_filename_train = data_dir + "ALEsKOs01_Metabolomics_train.csv"; // IndustrialStrains0103_
  std::string meta_data_filename_train = data_dir + "ALEsKOs01_MetaData_train.csv";
  std::string metabo_data_filename_test = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  std::string meta_data_filename_test = data_dir + "ALEsKOs01_MetaData_test.csv";
  bool make_model = true;
  bool train_model = true;
  int norm_method = 0;
  bool simulate_MARs = false;
  bool sample_concs = true;
  bool use_fold_change = false;
  std::string fold_change_ref = "Evo04";

  // Parse the input
  std::cout << "Parsing the user input..." << std::endl;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    biochem_rxns_filename = data_dir + argv[2];
  }
  if (argc >= 4) {
    metabo_data_filename_train = data_dir + argv[3];
  }
  if (argc >= 5) {
    meta_data_filename_train = data_dir + argv[4];
  }
  if (argc >= 6) {
    metabo_data_filename_test = data_dir + argv[5];
  }
  if (argc >= 7) {
    meta_data_filename_test = data_dir + argv[6];
  }
  if (argc >= 8) {
    make_model = (argv[7] == std::string("true")) ? true : false;
  }
  if (argc >= 9) {
    train_model = (argv[8] == std::string("true")) ? true : false;
  }
  if (argc >= 10) {
    try {
      norm_method = (std::stoi(argv[9]) >= 0 && std::stoi(argv[9]) <= 4) ? std::stoi(argv[9]) : 0;
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }
  if (argc >= 11) {
    simulate_MARs = (argv[10] == std::string("true")) ? true : false;
  }
  if (argc >= 12) {
    sample_concs = (argv[11] == std::string("true")) ? true : false;
  }
  if (argc >= 13) {
    use_fold_change = (argv[12] == std::string("true")) ? true : false;
  }
  if (argc >= 14) {
    fold_change_ref = argv[13];
  }

  // Cout the parsed input
  std::cout << "data_dir: " << data_dir << std::endl;
  std::cout << "biochem_rxns_filename: " << biochem_rxns_filename << std::endl;
  std::cout << "metabo_data_filename_train: " << metabo_data_filename_train << std::endl;
  std::cout << "meta_data_filename_train: " << meta_data_filename_train << std::endl;
  std::cout << "metabo_data_filename_test: " << metabo_data_filename_test << std::endl;
  std::cout << "meta_data_filename_test: " << meta_data_filename_test << std::endl;
  std::cout << "make_model: " << make_model << std::endl;
  std::cout << "train_model: " << train_model << std::endl;
  std::cout << "norm_method: " << norm_method << std::endl;
  std::cout << "simulate_MARs: " << simulate_MARs << std::endl;
  std::cout << "sample_concs: " << sample_concs << std::endl;
  std::cout << "use_fold_change: " << use_fold_change << std::endl;
  std::cout << "fold_change_ref: " << fold_change_ref << std::endl;

  // Run the classification
  main_classification(data_dir, biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test,
    make_model, train_model, norm_method,
    simulate_MARs, sample_concs, use_fold_change, fold_change_ref
  );
  return 0;
}