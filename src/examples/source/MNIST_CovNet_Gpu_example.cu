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

/**
 * EXAMPLES using the MNIST data set
 *
 * EXAMPLE1:
 * - classification on MNIST using DAG
 * - whole image pixels (linearized) 28x28 normalized to 0 to 1
 * - classifier (1 hot vector from 0 to 9)
 */

 // Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief Convolution classifier

  @param n_depth_1 32 (32 filters)
  @param n_depth_2 2 (total of 64 filters)
  @param n_fc 1024
  @param add_norm Optional normalization layer after each convolution

  References:
  https://github.com/pytorch/examples/blob/master/mnist/main.py
  */
  void makeCovNet(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, int n_depth_1 = 32, int n_depth_2 = 2, int n_fc = 128, bool add_norm = false, bool specify_layers = false) {
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
        28, 28, 0, 0,
        5, 5, 1, 0, 0,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        std::string norm_name = "Norm0-" + std::to_string(d);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0, false, specify_layers);
      }
      std::string pool_name = "Pool0-" + std::to_string(d);
      node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
        sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
        2, 2, 2, 0, 0,
        std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
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
          5, 5, 1, 0, 0,
          std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
        if (add_norm) {
          std::string norm_name = "Norm1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names,
            std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
            std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
            std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
            std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0, false, specify_layers);
        }
        std::string pool_name = "Pool1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        node_names = model_builder.addConvolution(model, pool_name, pool_name, node_names,
          sqrt(node_names.size()), sqrt(node_names.size()), 1, 1,
          2, 2, 2, 0, 0,
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
          std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
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
    //for (const std::vector<std::string> &node_names_l : node_names_l0) {
    for (const std::vector<std::string> &node_names_l : node_names_l1) {
      for (const std::string &node_name : node_names_l) {
        node_names.push_back(node_name);
      }
    }

    // Add the FC layers
    //assert(node_names.size() == 320);
    node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_fc,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(180, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "NormFC0", "NormFC0", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.1, 0.9, 0.999, 1e-8)), 0.0, 0.0, false, specify_layers);
    }
    node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_fc, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
  }
  /*
  @brief Convolution classifier using compact convolutions
    Current work in progress because the nodes2layers algorithm
    Does not yet recognize when weights need to be allocated to a different layer
    in order to prevent over-writing the weight values

  @param n_depth_1 32 (32 filters)
  @param n_depth_2 32 (total of 64 filters)
  @param n_fc 1024
  @param add_norm Optional normalization layer after each convolution

  References:
  https://github.com/pytorch/examples/blob/master/mnist/main.py
  */
  void makeCovNetCompact(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, int n_depth_1 = 32, int n_depth_2 = 32, int n_fc = 128, int add_scalar = true) {
    model.setId(0);
    model.setName("CovNet");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Add the first convolution -> max pool -> ReLU layers
    std::vector<std::string> node_names_conv0;
    std::string conv_name = "Conv0-" + std::to_string(0);
    node_names_conv0 = model_builder.addConvolution(model, "Conv0", conv_name, node_names_input,
      28, 28, 0, 0,
      5, 5, 1, 0, 0,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
    for (size_t d = 1; d < n_depth_1; ++d) {
      std::string conv_name = "Conv0-" + std::to_string(d);
      model_builder.addConvolution(model, "Conv0", conv_name, node_names_input, node_names_conv0,
        28, 28, 0, 0,
        5, 5, 1, 0, 0,
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true);
    }
    if (add_scalar) {
      node_names_conv0 = model_builder.addScalar(model, "Scalar0", "Scalar0", node_names_conv0, 5 * n_inputs,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        true);
    }

    // Add the second convolution -> max pool -> ReLU layers
    std::vector<std::string> node_names_conv1;
    conv_name = "Conv1-" + std::to_string(0);
    node_names_conv1 = model_builder.addConvolution(model, "Conv1", conv_name, node_names_conv0,
      sqrt(node_names_conv0.size()), sqrt(node_names_conv0.size()), 0, 0,
      5, 5, 1, 0, 0,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
    for (size_t d = 1; d < n_depth_2; ++d) {
      std::string conv_name = "Conv1-" + std::to_string(d);
      model_builder.addConvolution(model, "Conv1", conv_name, node_names_conv0, node_names_conv1,
        sqrt(node_names_conv0.size()), sqrt(node_names_conv0.size()), 0, 0,
        5, 5, 1, 0, 0,
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_inputs, 2)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true);
    }
    if (add_scalar) {
      node_names_conv1 = model_builder.addScalar(model, "Scalar1", "Scalar1", node_names_conv1, 5 * node_names_conv0.size(),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        true);
    }

    // Add the FC layers
    //assert(node_names.size() == 320);
    std::vector<std::string> node_names;
    node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names_conv1, n_fc,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(180, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);
    node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new ReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(n_fc, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(0.001, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, true, true);

    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    //if (n_epochs = 1000) {
    //	// anneal the learning rate to 1e-4
    //}
    if (n_epochs % 999 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      //model_interpreter.getModelResults(model, false, true, false);
      ModelFile<TensorT> data;
      //data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model);
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
    if (n_epochs % 10 == 0) {
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
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};
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

    // make a vector of sample_indices [BUG FREE]
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, n_epochs);

    // Reformat the input data for training [BUG FREE]
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
          for (int nodes_iter = 0; nodes_iter < this->training_data.dimension(1); ++nodes_iter) {
            //input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->training_data(sample_indices[0], nodes_iter);  // test on only 1 sample
          }
          for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
            //output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->training_labels(sample_indices[0], nodes_iter); // test on only 1 sample
            //output_data(batch_iter, memory_iter, nodes_iter + this->training_labels.dimension(1), epochs_iter) = (TensorT)this->training_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            ////output_data(batch_iter, memory_iter, nodes_iter + this->training_labels.dimension(1), epochs_iter) = (TensorT)this->training_labels(sample_indices[0], nodes_iter); // test on only 1 sample
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

    // make the start and end sample indices [BUG FREE]
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, n_epochs);

    // Reformat the input data for validation [BUG FREE]
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {
          for (int nodes_iter = 0; nodes_iter < this->validation_data.dimension(1); ++nodes_iter) {
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->validation_data(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
          }
          for (int nodes_iter = 0; nodes_iter < this->validation_labels.dimension(1); ++nodes_iter) {
            output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (TensorT)this->validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
            //output_data(batch_iter, memory_iter, nodes_iter + this->validation_labels.dimension(1), epochs_iter) = (TensorT)this->validation_labels(sample_indices[epochs_iter*batch_size + batch_iter], nodes_iter);
          }
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{
public:
  void adaptiveReplicatorScheduler(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
  { //TODO
  }
};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{
public:
  void adaptivePopulationScheduler(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
  {
    // Population size of 16
    if (n_generations == 0)
    {
      this->setNTop(3);
      this->setNRandom(3);
      this->setNReplicatesPerModel(15);
    }
    else
    {
      this->setNTop(3);
      this->setNRandom(3);
      this->setNReplicatesPerModel(3);
    }
  }
};

void main_CovNet() {

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = 1;

  // define the populatin trainer
  PopulationTrainerExt<float> population_trainer;
  population_trainer.setNGenerations(1);
  population_trainer.setNTop(1);
  population_trainer.setNRandom(1);
  population_trainer.setNReplicatesPerModel(1);
  population_trainer.setLogging(true);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);
  //ModelLogger<float> model_logger(true, true, true, true, true, false, true, true);

  // define the data simulator
  const std::size_t input_size = 784;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;

  // read in the training data
  const std::string training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
  const std::string training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
  //const std::string training_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-images-idx3-ubyte";
  //const std::string training_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-labels-idx1-ubyte";
  //const std::string training_data_filename = "/home/user/data/train-images-idx3-ubyte";
  //const std::string training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // read in the validation data
  const std::string validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
  const std::string validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
  //const std::string validation_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-images-idx3-ubyte";
  //const std::string validation_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-labels-idx1-ubyte";
  //const std::string validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
  //const std::string validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
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
  std::vector<std::string> output_FC_nodes;
  for (int i = 0; i < data_simulator.mnist_labels.size(); ++i) {
    char name_char[512];
    sprintf(name_char, "FC1_%012d", i);
    std::string name(name_char);
    output_FC_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < data_simulator.mnist_labels.size(); ++i) {
    char name_char[512];
    sprintf(name_char, "SoftMax-Out_%012d", i);
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
  model_trainer.setNEpochsTraining(1000);
  model_trainer.setNEpochsValidation(1);
  model_trainer.setNEpochsEvaluation(100);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, true, true);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setLossFunctions({
    //std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>())//,
    std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>())
    });
  model_trainer.setLossFunctionGrads({
    //std::shared_ptr<LossFunctionGradOp<float>>({new MSEGradOp<float>())//,	
    std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>())
    });
  model_trainer.setOutputNodes({ output_FC_nodes//, output_nodes 
    });

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  //model_trainer.makeCovNet(model, input_nodes.size(), output_nodes.size(), 2, 2, 32, false, true);  // Sanity test
  model_trainer.makeCovNet(model, input_nodes.size(), output_nodes.size(), 8, 2, 128, false, true);  // Minimal solving model
  //model_trainer.makeCovNet(model, input_nodes.size(), output_nodes.size(), 32, 2, 128, true, true); // Recommended model
  //model_trainer.makeCovNetCompact(model, input_nodes.size(), output_nodes.size(), 12, 12, 128);  // Test
  std::vector<Model<float>> population = { model };

  // Evolve the population
  std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

  PopulationTrainerFile<float> population_trainer_file;
  population_trainer_file.storeModels(population, "MNIST");
  population_trainer_file.storeModelValidations("MNISTErrors.csv", models_validation_errors_per_generation);
}

int main(int argc, char** argv)
{
  // run the application
  main_CovNet();

  return 0;
}