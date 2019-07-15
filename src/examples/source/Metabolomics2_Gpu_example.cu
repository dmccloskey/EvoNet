/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerGpu.h>
#include <SmartPeak/ml/ModelTrainerGpu.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileGpu.h>

#include "Metabolomics_example.h"

using namespace SmartPeak;

// Other extended classes
template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerGpu<TensorT>
{};

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  Model<TensorT> makeModel() { return Model<TensorT>(); }
  /*
  @brief Fully connected classifier
  */
  void makeModelFCClass(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, bool add_norm = true) {
    model.setId(0);
    model.setName("Classifier");

    const int n_hidden_0 = 64;
    const int n_hidden_1 = 64;
    const int n_hidden_2 = 0;

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
  @brief Fully connected variational reconstruction model
  */
  void makeModelFCVAE(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const int& n_encodings, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, bool add_norm = true) {
    model.setId(0);
    model.setName("VAE");

    const int n_en_hidden_0 = 64;
    const int n_en_hidden_1 = 64;
    const int n_en_hidden_2 = 0;
    const int n_de_hidden_0 = 64;
    const int n_de_hidden_1 = 64;
    const int n_de_hidden_2 = 0;


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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN2-Norm-gain", "EN2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
    std::vector<std::string> node_names_logvar = model_builder.addFullyConnected(model, "LogVar", "LogVar", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE0-Norm-gain", "DE0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE1-Norm-gain", "DE1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE2-Norm-gain", "DE2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }

    // Add the final output layer
    node_names = model_builder.addFullyConnected(model, "Output", "Output", node_names, n_outputs,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
    // Subtract out the pre-processed input data to test against all 0's
    model_builder.addSinglyConnected(model, "Output", node_names_input, node_names,
      std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(-1)),
      std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0f, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief Fully connected multitask model for variational reconstruction and classification
  */
  void makeModelFCMultiTask(Model<TensorT>& model, const int& n_inputs, const int& n_outputs_recon, const int& n_outputs_class, const int& n_encodings, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, bool add_norm = true) {
    model.setId(0);
    model.setName("MultiTask");
    const int n_en_hidden_0 = 64;
    const int n_en_hidden_1 = 64;
    const int n_en_hidden_2 = 0;
    const int n_de_hidden_0 = 64;
    const int n_de_hidden_1 = 64;
    const int n_de_hidden_2 = 0;
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN2-Norm-gain", "EN2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
    std::vector<std::string> node_names_logvar = model_builder.addFullyConnected(model, "LogVar", "LogVar", node_names, n_encodings,
      std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_encodings) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Specify the Encoding Mu and LogVar output nodes
    for (const std::string& node_name : node_names_mu)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_logvar)
      model.nodes_.at(node_name)->setType(NodeType::output);

    // Add the Variational Encoding layer
    std::vector<std::string> node_names_encoding = model_builder.addGaussianEncoding(model, "Encoding", "Encoding", node_names_mu, node_names_logvar, true);

    // Add the decoding layers
    if (n_de_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names_encoding, n_de_hidden_0,
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
        std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()),
        //std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()),
        std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
        std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
        std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
        std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_encoding.size() + n_de_hidden_0) / 2, 1)),
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE0-Norm-gain", "DE0-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE1-Norm-gain", "DE1-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
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
        std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE2-Norm-gain", "DE2-Norm-gain", node_names, node_names.size(),
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
          std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
          std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
          std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
          std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(1)),
          std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)),
          0.0, 0.0, true, true);
      }
    }

    // Add the final reconstruction output layer
    node_names = model_builder.addFullyConnected(model, "Output-Recon", "Output-Recon", node_names, n_outputs_recon,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names.size() + n_outputs_recon) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);
    // Subtract out the pre-processed input data to test against all 0's
    model_builder.addSinglyConnected(model, "Output-Recon", node_names_input, node_names,
      std::shared_ptr<WeightInitOp<TensorT>>(new ConstWeightInitOp<TensorT>(-1)),
      std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0f, true);

    // Specify the reconstruction output node types
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);

    // Add the classification output layer
    node_names = model_builder.addFullyConnected(model, "Output-Class", "Output-Class", node_names_mu, n_outputs_class,
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUOp<TensorT>()),
      std::shared_ptr<ActivationOp<TensorT>>(new LeakyReLUGradOp<TensorT>()),
      std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()),
      std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()),
      std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()),
      std::shared_ptr<WeightInitOp<TensorT>>(new RandWeightInitOp<TensorT>((int)(node_names_mu.size() + n_outputs_class) / 2, 1)),
      std::shared_ptr<SolverOp<TensorT>>(new AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8)), 0.0f, 0.0f, false, true);

    // Specify the classificaiton output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);

    // Set the input and output nodes
    model.setInputAndOutputNodes();

    //// Check that the model is set-up correctly
    //if (!model.checkCompleteInputToOutput())
    //  std::cout << "There is a problem with the model!" << std::endl;
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
        std::shared_ptr<SolverOp<TensorT>>(new DummySolverOp<TensorT>()), 0.0, 0.0, true, true);
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
    //if (n_epochs % 1000 == 0 && n_epochs != 0) {
    //  model_interpreter.getModelResults(model, false, true, false);
    //  ModelFile<TensorT> data;
    //  data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
    //  ModelInterpreterFileGpu<TensorT> interpreter_data;
    //  interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    //}
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
    if (n_epochs % 10 == 0) {
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

/*
@brief Example using intracellular E. coli metabolomics data
  taken from re-grown glycerol stock solutions on Glucose M9 at mid-exponential phase
  from adaptive laboratory evolution (ALE) experiments following gene knockout (KO)
*/

/// Script to run the time-course Summary
void main_statistics_timecourseSummary(const std::string& data_dir,
  bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
  bool run_timeCourse_TpiA = false)
{
  // define the data simulator
  BiochemicalReactionModel<float> metabolomics_data;

  std::string
    timeCourse_Ref_filename, timeCourse_Gnd_filename, timeCourse_SdhCB_filename, timeCourse_Pgi_filename, timeCourse_PtsHIcrr_filename,
    timeCourse_TpiA_filename,
    timeCourseSampleSummary_Ref_filename, timeCourseSampleSummary_Gnd_filename, timeCourseSampleSummary_SdhCB_filename, timeCourseSampleSummary_Pgi_filename, timeCourseSampleSummary_PtsHIcrr_filename,
    timeCourseSampleSummary_TpiA_filename,
    timeCourseFeatureSummary_Ref_filename, timeCourseFeatureSummary_Gnd_filename, timeCourseFeatureSummary_SdhCB_filename, timeCourseFeatureSummary_Pgi_filename, timeCourseFeatureSummary_PtsHIcrr_filename,
    timeCourseFeatureSummary_TpiA_filename;

  // filenames
  timeCourse_Ref_filename = data_dir + "EColi_timeCourse_Ref.csv";
  timeCourse_Gnd_filename = data_dir + "EColi_timeCourse_Gnd.csv";
  timeCourse_SdhCB_filename = data_dir + "EColi_timeCourse_SdhCB.csv";
  timeCourse_Pgi_filename = data_dir + "EColi_timeCourse_Pgi.csv";
  timeCourse_PtsHIcrr_filename = data_dir + "EColi_timeCourse_PtsHIcrr.csv";
  timeCourse_TpiA_filename = data_dir + "EColi_timeCourse_TpiA.csv";
  timeCourseSampleSummary_Ref_filename = data_dir + "EColi_timeCourseSampleSummary_Ref.csv";
  timeCourseSampleSummary_Gnd_filename = data_dir + "EColi_timeCourseSampleSummary_Gnd.csv";
  timeCourseSampleSummary_SdhCB_filename = data_dir + "EColi_timeCourseSampleSummary_SdhCB.csv";
  timeCourseSampleSummary_Pgi_filename = data_dir + "EColi_timeCourseSampleSummary_Pgi.csv";
  timeCourseSampleSummary_PtsHIcrr_filename = data_dir + "EColi_timeCourseSampleSummary_PtsHIcrr.csv";
  timeCourseSampleSummary_TpiA_filename = data_dir + "EColi_timeCourseSampleSummary_TpiA.csv";
  timeCourseFeatureSummary_Ref_filename = data_dir + "EColi_timeCourseFeatureSummary_Ref.csv";
  timeCourseFeatureSummary_Gnd_filename = data_dir + "EColi_timeCourseFeatureSummary_Gnd.csv";
  timeCourseFeatureSummary_SdhCB_filename = data_dir + "EColi_timeCourseFeatureSummary_SdhCB.csv";
  timeCourseFeatureSummary_Pgi_filename = data_dir + "EColi_timeCourseFeatureSummary_Pgi.csv";
  timeCourseFeatureSummary_PtsHIcrr_filename = data_dir + "EColi_timeCourseFeatureSummary_PtsHIcrr.csv";
  timeCourseFeatureSummary_TpiA_filename = data_dir + "EColi_timeCourseFeatureSummary_TpiA.csv";

  if (run_timeCourse_Ref) {
    // Read in the data
    PWData timeCourseRef;
    ReadPWData(timeCourse_Ref_filename, timeCourseRef);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseRef, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_Ref_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_Ref_filename, pw_feature_summaries);
  }

  if (run_timeCourse_Gnd) {
    // Read in the data
    PWData timeCourseGnd;
    ReadPWData(timeCourse_Gnd_filename, timeCourseGnd);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseGnd, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_Gnd_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_Gnd_filename, pw_feature_summaries);
  }

  if (run_timeCourse_SdhCB) {
    // Read in the data
    PWData timeCourseSdhCB;
    ReadPWData(timeCourse_SdhCB_filename, timeCourseSdhCB);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseSdhCB, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_SdhCB_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_SdhCB_filename, pw_feature_summaries);
  }

  if (run_timeCourse_Pgi) {
    // Read in the data
    PWData timeCoursePgi;
    ReadPWData(timeCourse_Pgi_filename, timeCoursePgi);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCoursePgi, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_Pgi_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_Pgi_filename, pw_feature_summaries);
  }

  if (run_timeCourse_PtsHIcrr) {
    // Read in the data
    PWData timeCoursePtsHIcrr;
    ReadPWData(timeCourse_PtsHIcrr_filename, timeCoursePtsHIcrr);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCoursePtsHIcrr, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_PtsHIcrr_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_PtsHIcrr_filename, pw_feature_summaries);
  }

  if (run_timeCourse_TpiA) {
    // Read in the data
    PWData timeCourseTpiA;
    ReadPWData(timeCourse_TpiA_filename, timeCourseTpiA);

    // Summarize the data
    PWSampleSummaries pw_sample_summaries;
    PWFeatureSummaries pw_feature_summaries;
    PWTotalSummary pw_total_summary;
    PWSummary(timeCourseTpiA, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

    // Export to file
    WritePWSampleSummaries(timeCourseSampleSummary_TpiA_filename, pw_sample_summaries);
    WritePWFeatureSummaries(timeCourseFeatureSummary_TpiA_filename, pw_feature_summaries);
  }
}

/// Script to run the time-course MARs analysis
void main_statistics_timecourse(const std::string& data_dir,
  bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
  bool run_timeCourse_TpiA = false)
{
  // define the data simulator
  BiochemicalReactionModel<float> metabolomics_data;

  std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
    timeCourse_Ref_filename, timeCourse_Gnd_filename, timeCourse_SdhCB_filename, timeCourse_Pgi_filename, timeCourse_PtsHIcrr_filename,
    timeCourse_TpiA_filename;
  std::vector<std::string> pre_samples,
    timeCourse_Ref_samples, timeCourse_Gnd_samples, timeCourse_SdhCB_samples, timeCourse_Pgi_samples, timeCourse_PtsHIcrr_samples,
    timeCourse_TpiA_samples;
  // filenames
  biochem_rxns_filename = data_dir + "iJO1366.csv";
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData.csv";
  timeCourse_Ref_filename = data_dir + "EColi_timeCourse_Ref.csv";
  timeCourse_Gnd_filename = data_dir + "EColi_timeCourse_Gnd.csv";
  timeCourse_SdhCB_filename = data_dir + "EColi_timeCourse_SdhCB.csv";
  timeCourse_Pgi_filename = data_dir + "EColi_timeCourse_Pgi.csv";
  timeCourse_PtsHIcrr_filename = data_dir + "EColi_timeCourse_PtsHIcrr.csv";
  timeCourse_TpiA_filename = data_dir + "EColi_timeCourse_TpiA.csv";
  timeCourse_Ref_samples = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP" };
  timeCourse_Gnd_samples = { "Evo04", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04gndEvo03EP" };
  timeCourse_SdhCB_samples = { "Evo04", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04sdhCBEvo03EP", "Evo04sdhCBEvo03EP-2", "Evo04sdhCBEvo03EP-3", "Evo04sdhCBEvo03EP-4", "Evo04sdhCBEvo03EP-5", "Evo04sdhCBEvo03EP-6" };
  timeCourse_Pgi_samples = { "Evo04", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo01J01", "Evo04pgiEvo01J02", "Evo04pgiEvo02EP", "Evo04pgiEvo02J01", "Evo04pgiEvo02J02", "Evo04pgiEvo02J03", "Evo04pgiEvo03EP", "Evo04pgiEvo03J01", "Evo04pgiEvo03J02", "Evo04pgiEvo03J03", "Evo04pgiEvo04EP", "Evo04pgiEvo04J01", "Evo04pgiEvo04J02", "Evo04pgiEvo04J03", "Evo04pgiEvo05EP", "Evo04pgiEvo05J01", "Evo04pgiEvo05J02", "Evo04pgiEvo05J03", "Evo04pgiEvo06EP", "Evo04pgiEvo06J01", "Evo04pgiEvo06J02", "Evo04pgiEvo06J03", "Evo04pgiEvo07EP", "Evo04pgiEvo07J01", "Evo04pgiEvo07J02", "Evo04pgiEvo07J03", "Evo04pgiEvo08EP", "Evo04pgiEvo08J01", "Evo04pgiEvo08J02", "Evo04pgiEvo08J03" };
  timeCourse_PtsHIcrr_samples = { "Evo04", "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo01J01", "Evo04ptsHIcrrEvo01J03", "Evo04ptsHIcrrEvo02EP", "Evo04ptsHIcrrEvo02J01", "Evo04ptsHIcrrEvo02J03", "Evo04ptsHIcrrEvo03EP", "Evo04ptsHIcrrEvo03J01", "Evo04ptsHIcrrEvo03J03", "Evo04ptsHIcrrEvo03J04", "Evo04ptsHIcrrEvo04EP", "Evo04ptsHIcrrEvo04J01", "Evo04ptsHIcrrEvo04J03", "Evo04ptsHIcrrEvo04J04" };
  timeCourse_TpiA_samples = { "Evo04", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo01J01", "Evo04tpiAEvo01J03", "Evo04tpiAEvo02EP", "Evo04tpiAEvo02J01", "Evo04tpiAEvo02J03", "Evo04tpiAEvo03EP", "Evo04tpiAEvo03J01", "Evo04tpiAEvo03J03", "Evo04tpiAEvo04EP", "Evo04tpiAEvo04J01", "Evo04tpiAEvo04J03" };

  // read in the data
  metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
  metabolomics_data.readMetabolomicsData(metabo_data_filename);
  metabolomics_data.readMetaData(meta_data_filename);
  metabolomics_data.findComponentGroupNames();
  metabolomics_data.findMARs();
  metabolomics_data.findMARs(true, false);
  metabolomics_data.findMARs(false, true);
  metabolomics_data.findLabels();

  if (run_timeCourse_Ref) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseRef = PWComparison(metabolomics_data, timeCourse_Ref_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_Ref_filename, timeCourseRef);
  }

  if (run_timeCourse_Gnd) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseGnd = PWComparison(metabolomics_data, timeCourse_Gnd_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_Gnd_filename, timeCourseGnd);
  }

  if (run_timeCourse_SdhCB) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseSdhCB = PWComparison(metabolomics_data, timeCourse_SdhCB_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_SdhCB_filename, timeCourseSdhCB);
  }

  if (run_timeCourse_Pgi) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCoursePgi = PWComparison(metabolomics_data, timeCourse_Pgi_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_Pgi_filename, timeCoursePgi);
  }

  if (run_timeCourse_PtsHIcrr) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCoursePtsHIcrr = PWComparison(metabolomics_data, timeCourse_PtsHIcrr_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_PtsHIcrr_filename, timeCoursePtsHIcrr);
  }

  if (run_timeCourse_TpiA) {
    // Find significant pair-wise MARS between each sample (one vs one)
    PWData timeCourseTpiA = PWComparison(metabolomics_data, timeCourse_TpiA_samples, 10000, 0.05, 1.0);

    // Export to file
    WritePWData(timeCourse_TpiA_filename, timeCourseTpiA);
  }
}

/// Script to run the classification network
void main_classification(const std::string& data_dir, bool make_model = true, bool simulate_MARs = true, bool sample_concs = true)
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
  std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename;
  biochem_rxns_filename = data_dir + "iJO1366.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData_train.csv";

  // Training data
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics_train.csv";
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename);
  reaction_model.readMetaData(meta_data_filename);
  reaction_model.findComponentGroupNames();
  reaction_model.findMARs();
  reaction_model.findMARs(true, false);
  reaction_model.findMARs(false, true);
  reaction_model.removeRedundantMARs();
  reaction_model.findLabels();
  metabolomics_data.model_training_ = reaction_model;

  // Validation data
  reaction_model.clear();
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData_test.csv";
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename);
  reaction_model.readMetaData(meta_data_filename);
  reaction_model.findComponentGroupNames();
  reaction_model.findMARs();
  reaction_model.findMARs(true, false);
  reaction_model.findMARs(false, true);
  reaction_model.removeRedundantMARs();
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
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
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
    std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
  model_trainer.setLossFunctionGrads({
    std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>()),
    std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
  model_trainer.setLossOutputNodes({
    output_nodes,
    output_nodes });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new AccuracyMCMicroOp<float>()), std::shared_ptr<MetricFunctionOp<float>>(new PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes, output_nodes });
  model_trainer.setMetricNames({ "AccuracyMCMicro", "PrecisionMCMicro" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // initialize the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  //std::vector<Model<float>> population;
  Model<float> model;
  if (make_model) {
    model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, false, false, false, false); // normalization type 0
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, false, false, false); // normalization type 1
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, false, true, false); // normalization type 2
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, false, false); // normalization type 3
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, true, false); // normalization type 4

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

/// Script to run the reconstruction network
void main_reconstruction(const std::string& data_dir, bool make_model = true, bool simulate_MARs = true, bool sample_concs = true)
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
  MetDataSimReconstruction<float> metabolomics_data;
  std::string model_name = "0_Metabolomics";

  // Read in the training and validation data
  std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename;
  biochem_rxns_filename = data_dir + "iJO1366.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData_train.csv";

  // Training data
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics_train.csv";
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename);
  reaction_model.readMetaData(meta_data_filename);
  reaction_model.findComponentGroupNames();
  reaction_model.findMARs();
  reaction_model.findMARs(true, false);
  reaction_model.findMARs(false, true);
  reaction_model.removeRedundantMARs();
  reaction_model.findLabels();
  metabolomics_data.model_training_ = reaction_model;

  // Validation data
  reaction_model.clear();
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData_test.csv";
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename);
  reaction_model.readMetaData(meta_data_filename);
  reaction_model.findComponentGroupNames();
  reaction_model.findMARs();
  reaction_model.findMARs(true, false);
  reaction_model.findMARs(false, true);
  reaction_model.removeRedundantMARs();
  reaction_model.findLabels();
  metabolomics_data.model_validation_ = reaction_model;
  metabolomics_data.simulate_MARs_ = simulate_MARs;
  metabolomics_data.sample_concs_ = sample_concs;

  // Checks for the training and validation data
  assert(metabolomics_data.model_validation_.reaction_ids_.size() == metabolomics_data.model_training_.reaction_ids_.size());
  assert(metabolomics_data.model_validation_.labels_.size() == metabolomics_data.model_training_.labels_.size());
  assert(metabolomics_data.model_validation_.component_group_names_.size() == metabolomics_data.model_training_.component_group_names_.size());

  // Define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = n_input_nodes;
  const int encoding_size = 2;
  metabolomics_data.n_encodings_ = encoding_size;
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;

  // Make the input nodes
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
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

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(16);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(100000);
  model_trainer.setNEpochsValidation(0);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({
    std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>(1e-6, 1.0)),
    //std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceMuOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceLogVarOp<float>(1e-6, 0.1)) });
  model_trainer.setLossFunctionGrads({
    std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>(1e-6, 1.0)),
    //std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceMuGradOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceLogVarGradOp<float>(1e-6, 0.1)) });
  model_trainer.setLossOutputNodes({ output_nodes, encoding_nodes_mu, encoding_nodes_logvar });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new MAEOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes });
  model_trainer.setMetricNames({ "MAE" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // initialize the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  //std::vector<Model<float>> population;
  Model<float> model;
  if (make_model) {
    model_trainer.makeModelFCVAE(model, n_input_nodes, n_output_nodes, encoding_size, true, true, false, false); // normalization type 3
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

/// Script to run the reconstruction network
void main_multiTask(const std::string& data_dir, bool make_model = true, bool simulate_MARs = true, bool sample_concs = true)
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
  MetDataSimMultiTask<float> metabolomics_data;
  std::string model_name = "0_Metabolomics";

  // Read in the training and validation data
  std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename;
  biochem_rxns_filename = data_dir + "iJO1366.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData_train.csv";

  // Training data
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics_train.csv";
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename);
  reaction_model.readMetaData(meta_data_filename);
  reaction_model.findComponentGroupNames();
  reaction_model.findMARs();
  reaction_model.findMARs(true, false);
  reaction_model.findMARs(false, true);
  reaction_model.removeRedundantMARs();
  reaction_model.findLabels();
  metabolomics_data.model_training_ = reaction_model;

  // Validation data
  reaction_model.clear();
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData_test.csv";
  reaction_model.readBiochemicalReactions(biochem_rxns_filename, true);
  reaction_model.readMetabolomicsData(metabo_data_filename);
  reaction_model.readMetaData(meta_data_filename);
  reaction_model.findComponentGroupNames();
  reaction_model.findMARs();
  reaction_model.findMARs(true, false);
  reaction_model.findMARs(false, true);
  reaction_model.removeRedundantMARs();
  reaction_model.findLabels();
  metabolomics_data.model_validation_ = reaction_model;
  metabolomics_data.simulate_MARs_ = simulate_MARs;
  metabolomics_data.sample_concs_ = sample_concs;

  // Checks for the training and validation data
  assert(metabolomics_data.model_validation_.reaction_ids_.size() == metabolomics_data.model_training_.reaction_ids_.size());
  assert(metabolomics_data.model_validation_.labels_.size() == metabolomics_data.model_training_.labels_.size());
  assert(metabolomics_data.model_validation_.component_group_names_.size() == metabolomics_data.model_training_.component_group_names_.size());

  // Define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes_recon = n_input_nodes;
  const int n_output_nodes_class = reaction_model.labels_.size();
  const int encoding_size = 3;
  metabolomics_data.n_encodings_ = encoding_size;

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the encoding nodes and add them to the input
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Encoding_%012d-Sampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the reconstruction nodes
  std::vector<std::string> output_nodes_recon;
  for (int i = 0; i < n_output_nodes_recon; ++i) {
    char name_char[512];
    sprintf(name_char, "Output-Recon_%012d", i);
    std::string name(name_char);
    output_nodes_recon.push_back(name);
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

  // Make the classification nodes
  std::vector<std::string> output_nodes_class;
  for (int i = 0; i < n_output_nodes_class; ++i) {
    char name_char[512];
    sprintf(name_char, "Output-Class_%012d", i);
    std::string name(name_char);
    output_nodes_class.push_back(name);
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  for (size_t i = 0; i < n_threads; ++i) {
    ModelResources model_resources = { ModelDevice(0, 1) };
    ModelInterpreterGpu<float> model_interpreter(model_resources);
    model_interpreters.push_back(model_interpreter);
  }
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(16);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsTraining(100000);
  model_trainer.setNEpochsValidation(0);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({
    std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>(1e-6, 1.0)),
    //std::shared_ptr<LossFunctionOp<float>>(new BCEWithLogitsOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceMuOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionOp<float>>(new KLDivergenceLogVarOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>()),
    std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
  model_trainer.setLossFunctionGrads({
    std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>(1e-6, 1.0)),
    //std::shared_ptr<LossFunctionGradOp<float>>(new BCEWithLogitsGradOp<float>(1e-6, 1.0)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceMuGradOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionGradOp<float>>(new KLDivergenceLogVarGradOp<float>(1e-6, 0.1)),
    std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>()),
    std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
  model_trainer.setLossOutputNodes({ output_nodes_recon, encoding_nodes_mu, encoding_nodes_logvar,
    output_nodes_class, output_nodes_class });
  model_trainer.setMetricFunctions({ std::shared_ptr<MetricFunctionOp<float>>(new MAEOp<float>()), 
    std::shared_ptr<MetricFunctionOp<float>>(new AccuracyMCMicroOp<float>()), 
    std::shared_ptr<MetricFunctionOp<float>>(new PrecisionMCMicroOp<float>()) });
  model_trainer.setMetricOutputNodes({ output_nodes_recon, output_nodes_class, output_nodes_class });
  model_trainer.setMetricNames({ "MAE", "AccuracyMCMicro", "PrecisionMCMicro" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // initialize the model replicator
  ModelReplicatorExt<float> model_replicator;

  // define the initial population
  std::cout << "Initializing the population..." << std::endl;
  //std::vector<Model<float>> population;
  Model<float> model;
  if (make_model) {
    model_trainer.makeModelFCMultiTask(model, n_input_nodes, n_output_nodes_recon, n_output_nodes_class, encoding_size, true, false, false, false); // normalization type 1
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
  //std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "/home/user/Data/";

  //main_statistics_timecourse(data_dir, 
  //	true, true, true, true, true,
  //	true);
  //main_statistics_timecourseSummary(data_dir, 
  //	true, true, true, true, true,
  //	true);
  main_classification(data_dir, true, true, true);
  //main_reconstruction(data_dir, true, false, true);
  //main_multiTask(data_dir, true, false, true);
  return 0;
}