/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>
#include <SmartPeak/io/ModelFile.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief Convolution classifier

  References:
  https://github.com/pytorch/examples/blob/master/mnist/main.py

  @param model The network model
  @param n_depth_1 The number of filters in the first Cov layer
  @param n_depth_2 The number of filters to create from each individual filter in the first Cov layer
    e.g., n_depth_1 = 32 and n_depth_2 = 2 the first Cov layer will have 32 filters and the second will have 64 layers
  @param n_fc The length of each fully connected layer
  @param add_norm Optional normalization layer after each convolution
  */
  void makeCovNet(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, int n_depth_1 = 32, int n_depth_2 = 2, int n_fc_1 = 128, int n_fc_2 = 32, int filter_size = 5, int pool_size = 2, bool add_norm = false, bool specify_layers = false) {
    model.setId(0);
    model.setName("CovNet");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Add the first convolution -> max pool -> ReLU layers
    std::vector<std::vector<std::string>> node_names_l0;
    for (size_t d = 0; d < n_depth_1; ++d) {
      std::vector<std::string> node_names;
      std::string conv_name = "Conv0-" + std::to_string(d);
      node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_input,
        sqrt(node_names_input.size()), sqrt(node_names_input.size()), 0, 0,
        filter_size, filter_size, 1, 0, 0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(filter_size*filter_size, 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        std::string norm_name = "Norm0-" + std::to_string(d);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
        std::string gain_name = "Gain0-" + std::to_string(d);
        node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, specify_layers);
      }
      std::string pool_name = "Pool0-" + std::to_string(d);
      node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
        sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
        pool_size, pool_size, 2, 0, 0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new MaxOp<float>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new MaxErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new MaxWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1.0)),
        std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      node_names_l0.push_back(node_names);
    }

    // Add the second convolution -> max pool -> ReLU layers
    std::vector<std::vector<std::string>> node_names_l1;
    int l_cnt = 0;
    for (const std::vector<std::string> &node_names_l : node_names_l0) {
      for (size_t d = 0; d < n_depth_2; ++d) {
        std::vector<std::string> node_names;
        std::string conv_name = "Conv1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_l,
          sqrt(node_names_l.size()), sqrt(node_names_l.size()), 0, 0,
          filter_size, filter_size, 1, 0, 0,
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
          //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(filter_size*filter_size, 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
        if (add_norm) {
          std::string norm_name = "Norm1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
          std::string gain_name = "Gain1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
            std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
            std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
            std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
            std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
            std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
            std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
            std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)),
            0.0, 0.0, true, specify_layers);
        }
        std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
          sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
          pool_size, pool_size, 2, 0, 0,
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new MaxOp<float>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new MaxErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new MaxWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1.0)),
          std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
        node_names_l1.push_back(node_names);
      }
      ++l_cnt;
    }

    // Linearize the node names
    std::vector<std::string> node_names;
    for (const std::vector<std::string> &node_names_l : node_names_l1) {
      for (const std::string &node_name : node_names_l) {
        node_names.push_back(node_name);
      }
    }

    // Add the FC layers
    node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_fc_1,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_fc_1, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "FC0-Norm", "FC0-Norm", node_names, true);
      node_names = model_builder.addSinglyConnected(model, "FC0-Norm-gain", "FC0-Norm-gain", node_names, node_names.size(),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)),
        0.0, 0.0, true, specify_layers);
    }
    node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_fc_2,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_fc_2, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "FC1-Norm", "FC1-Norm", node_names, true);
      node_names = model_builder.addSinglyConnected(model, "FC1-Norm-gain", "FC1-Norm-gain", node_names, node_names.size(),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)),
        0.0, 0.0, true, specify_layers);
    }

    // Add the final output layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-3, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Manually define the output nodes
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      ModelFile<TensorT> data;
      model_interpreter.getModelResults(model, false, true, false);

      //// Save to .csv
      //data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);

      // Save to binary
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values,
    const std::vector<std::string>& output_nodes,
    const TensorT& model_error)
  {
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedPredictedEpoch(false);
    if (n_epochs == 0) {
      model_logger.initLogs(model);
    }
    if (n_epochs % 1 == 0) {
      if (model_logger.getLogExpectedPredictedEpoch())
        model_interpreter.getModelResults(model, true, false, false);
      model_logger.writeLogs(model, n_epochs, { "Error" }, {}, { model_error }, {}, output_nodes, expected_values);
    }
  }
  void validationModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values,
    const std::vector<std::string>& output_nodes,
    const TensorT& model_error)
  {
    model_logger.setLogTimeEpoch(false);
    model_logger.setLogTrainValMetricEpoch(false);
    model_logger.setLogExpectedPredictedEpoch(true);
    if (n_epochs == 0) {
      model_logger.initLogs(model);
    }
    if (n_epochs % 1 == 0) {
      if (model_logger.getLogExpectedPredictedEpoch())
        model_interpreter.getModelResults(model, true, false, false);
      model_logger.writeLogs(model, n_epochs, {}, { "Error" }, {}, { model_error }, output_nodes, expected_values);
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
      //model_logger.setLogExpectedPredictedEpoch(true);
      model_logger.initLogs(model);
    }

    //// Per n epoch logging
    //if (n_epochs % 10 == 0) {
    //  model_logger.setLogExpectedPredictedEpoch(true);
    //  model_interpreter.getModelResults(model, true, false, false);
    //}

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

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == this->validation_data.dimension(1));

    // make a vector of sample_indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, n_epochs);

    // Reformat the input data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
          for (int nodes_iter = 0; nodes_iter < this->training_data.dimension(1); ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            //input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
          }
          for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            //output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[0], nodes_iter); // test on only 1 sample
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_epochs = input_data.dimension(3);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == this->validation_data.dimension(1));

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, n_epochs);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
          for (int nodes_iter = 0; nodes_iter < this->validation_data.dimension(1); ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
          }
          for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    assert(n_output_nodes == this->training_labels.dimension(1));
    assert(n_metric_output_nodes == this->training_labels.dimension(1));
    assert(n_input_nodes == 784);
    assert(memory_size == 1);

    // make the start and end sample indices [BUG FREE]
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training [BUG FREE]
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
        }
      }
    }
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    assert(n_output_nodes == this->validation_labels.dimension(1));
    assert(n_metric_output_nodes == this->validation_labels.dimension(1));
    assert(n_input_nodes == 784);
    assert(memory_size == 1);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
          loss_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
        }
      }
    }
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{};

/**
 @brief Image classification MNIST example whereby all pixels are
  linearized and read into the model.  The model then attempts to
  classify the image using a CovNet architecture

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
  - classifier (1 hot vector from 0 to 9)
 */
void main_MNIST(const std::string& data_dir, const bool& make_model, const bool& train_model) {

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = 1;

  // define the populatin trainer
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setLogging(false);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // define the data simulator
  const std::size_t input_size = 784;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;

  // read in the training data
  std::string training_data_filename = data_dir + "train-images.idx3-ubyte";
  std::string training_labels_filename = data_dir + "train-labels.idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // read in the validation data
  std::string validation_data_filename = data_dir + "t10k-images.idx3-ubyte";
  std::string validation_labels_filename = data_dir + "t10k-labels.idx1-ubyte";
  data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, input_size);
  data_simulator.unitScaleData();

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < input_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < data_simulator.mnist_labels.size(); ++i) {
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
  model_trainer.setNEpochsTraining(100001);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>()) });
  model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>()) });
  model_trainer.setLossOutputNodes({ output_nodes });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes });
  model_trainer.setMetricNames({ "PrecisionMCMicro" });

  // define the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    //model_trainer.makeCovNet(model, input_nodes.size(), output_nodes.size(), 2, 2, 32, 16, 5, 2, false, true);  // Sanity test
    model_trainer.makeCovNet(model, input_nodes.size(), output_nodes.size(), 8, 2, 128, 32, 5, 2, false, true);  // Minimal solving model
    //model_trainer.makeCovNet(model, input_nodes.size(), output_nodes.size(), 32, 2, 128, true, true); // Recommended model
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "CovNet_1000_model.binary";
    const std::string interpreter_filename = data_dir + "CovNet_1000_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("CovNet1");
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  std::vector<Model<float>> population = { model };

  if (train_model) {
    // Train the model
    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());

    //// Evolve the population
    //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    //PopulationTrainerFile<float> population_trainer_file;
    //population_trainer_file.storeModels(population, "MNIST");
    //population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation);
  }
  else {
    // Evaluate the population
    population_trainer.evaluateModels(
      population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
  }
}

int main(int argc, char** argv)
{
  // define the data directory
  //std::string data_dir = "/home/user/data/";
  std::string data_dir = "C:/Users/domccl/GitHub/mnist/";
  //std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";

  // run the application
  main_MNIST(data_dir, true, true);

  return 0;
}