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

// Extended 
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  /*
  @brief Basic AutoEncoder

  @param[in, out] model The network model
  @param[in] n_inputs The number of input pixels
  @param[in] n_encodings The length of the encodings layer
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeAE(Model<TensorT>& model, int n_inputs = 784, int n_encodings = 64, int n_hidden_0 = 512, bool specify_layer = true, bool add_norm = true) {
    model.setId(0);
    model.setName("VAE");
    bool included_bias = false;

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layer);

    // Add the Endocer FC layers
    std::vector<std::string> node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1/(TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names_input.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }
    node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }
    node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }
    node_names = model_builder.addFullyConnected(model, "EN3", "EN3", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "EN3-Norm", "EN3-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }

    node_names = model_builder.addFullyConnected(model, "Encoding", "Encoding", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_encodings), 10 / (TensorT)(node_names_input.size() + n_encodings))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_encodings, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "Encoding-Norm", "Encoding-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }

    // Add the Decoder FC layers
    node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }
    node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }
    node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }
    node_names = model_builder.addFullyConnected(model, "DE3", "DE3", node_names, n_hidden_0,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)(node_names_input.size() + n_hidden_0), 10 / (TensorT)(node_names_input.size() + n_hidden_0))),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size() + n_hidden_0, 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, specify_layer);
    if (add_norm) {
      node_names = model_builder.addNormalization(model, "DE3-Norm", "DE3-Norm", node_names,
        std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
        std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0, 0.0, included_bias, specify_layer);
    }
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_inputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>(1e-6, 0.001, 1.001)),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RangeWeightInitOp<TensorT>(0.1 / (TensorT)node_names_input.size(), 10 / (TensorT)node_names_input.size())),
      //std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>(node_names.size(), 2)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-6, 0.9, 0.999, 1e-3, 10.0)), 0.0f, 0.0f, included_bias, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    if (n_epochs % 1000 == 0 && n_epochs > 5000) {
      // anneal the learning rate by half on each plateau
      TensorT lr_new = this->reduceLROnPlateau(model_errors, 0.5, 100, 10, 0.05);
      if (lr_new < 1.0) {
        model_interpreter.updateSolverParams(0, lr_new);
        std::cout << "The learning rate has been annealed by a factor of " << lr_new << std::endl;
      }
    }
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const TensorT & model_error_train, const TensorT & model_error_test)
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
      if (model_logger.getLogExpectedPredictedEpoch())
        model_interpreter.getModelResults(model, true, false, false);
      model_logger.writeLogs(model, n_epochs, { "Train_Error" }, { "Test_Error" }, { model_error_train }, { model_error_test }, output_nodes, expected_values);
    }
    else if (n_epochs % 10 == 0) {
      model_logger.setLogExpectedPredictedEpoch(false);
      if (model_logger.getLogExpectedPredictedEpoch())
        model_interpreter.getModelResults(model, true, false, false);
      model_logger.writeLogs(model, n_epochs, { "Train_Error" }, { "Test_Error" }, { model_error_train }, { model_error_test }, output_nodes, expected_values);
    }
  }
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels);
    //assert(n_output_nodes == 2*n_input_pixels);
    assert(n_input_nodes == n_input_pixels);

    // make a vector of sample_indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the MNIST image data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter) + 1e-3;
          output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter) + 1e-3;
          //output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = this->training_data(sample_indices[batch_iter], nodes_iter);
        }
      }
    }

    time_steps.setConstant(1.0f);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels);
    //assert(n_output_nodes == 2 * n_input_pixels);
    assert(n_input_nodes == n_input_pixels);

    // make a vector of sample_indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the MNIST image data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter) + 1e-3;
          output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter) + 1e-3;
          //output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = this->validation_data(sample_indices[batch_iter], nodes_iter);
        }
      }
    }
    time_steps.setConstant(1.0f);
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{};

/**
 @brief Pixel reconstruction MNIST example whereby all pixels are
  linearized and read into the model.  The model then attempts to
  reconstruction the pixels using an Auto Encoder network

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
 */

void main_AE(const bool& make_model, const bool& train_model) {

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
  const std::size_t encoding_size = 16;
  const std::size_t n_hidden = 128;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;

  // read in the training data
  std::string training_data_filename, training_labels_filename;
  //training_data_filename = "/home/user/data/train-images-idx3-ubyte";
  //training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
  training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
  training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
  //training_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-images-idx3-ubyte";
  //training_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/train-labels-idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // read in the validation data
  std::string validation_data_filename, validation_labels_filename;
  //validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
  //validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
  validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
  validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
  //validation_data_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-images-idx3-ubyte";
  //validation_labels_filename = "C:/Users/dmccloskey/Documents/GitHub/mnist/t10k-labels-idx1-ubyte";
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
  for (int i = 0; i < input_size; ++i) {
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
  //model_trainer.setBatchSize(1); // evaluation only
  model_trainer.setBatchSize(64);
  model_trainer.setNEpochsTraining(200001);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNEpochsEvaluation(100);
  model_trainer.setMemorySize(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, true, true);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setLossFunctions({
    std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>(1e-6, 1.0))
    //,std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>(1e-6, 0.05))
    });
  model_trainer.setLossFunctionGrads({
    std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>(1e-6, 1.0))
    //,std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>(1e-6, 0.05))
    });
  model_trainer.setOutputNodes({ output_nodes
    //, output_nodes 
    });

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  Model<float> model;
  if (make_model) {
    ModelTrainerExt<float>().makeAE(model, input_size, encoding_size, n_hidden, true, false);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string data_dir = "C:/Users/domccl/Desktop/EvoNetExp/MNIST_AE_GPU/GPU8/";
    const std::string model_filename = data_dir + "VAE_6000_model.binary";
    const std::string interpreter_filename = data_dir + "VAE_6000_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("AE1");
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }
  std::vector<Model<float>> population = { model };

  if (train_model) {
    // Evolve the population
    //std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
    //  population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);

    std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
      input_nodes, model_logger, model_interpreters.front());

    PopulationTrainerFile<float> population_trainer_file;
    population_trainer_file.storeModels(population, "MNIST");
  }
  else {
    // Evaluate the population
    population_trainer.evaluateModels(
      population, model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
  }
}

int main(int argc, char** argv)
{
  // run the application
  main_AE(true, true);

  return 0;
}