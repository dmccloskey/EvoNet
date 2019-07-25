/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/simulator/BiochemicalReaction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/io/ModelFile.h>

using namespace SmartPeak;

// Other extended classes
template<typename TensorT>
class ModelReplicatorExt : public ModelReplicator<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerDefaultDevice<TensorT>
{};

template<typename TensorT>
class MetDataSimLatentArithmetic : public DataSimulator<TensorT>
{
public:
  void simulateDataMARs(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_epochs = input_data.dimension(3);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_.reaction_ids_.size();
    else
      n_input_pixels = this->model_validation_.reaction_ids_.size();
    assert(n_input_nodes == n_input_pixels);

    for (int epoch_iter = 0; epoch_iter < n_epochs; ++epoch_iter) {
      for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
            TensorT value;
            if (train)
              value = this->model_training_.calculateMAR(
                this->model_training_.metabolomicsData_.at(sample_group_name_),
                this->model_training_.biochemicalReactions_.at(this->model_training_.reaction_ids_.at(nodes_iter)));
            else
              value = this->model_validation_.calculateMAR(
                this->model_validation_.metabolomicsData_.at(sample_group_name_),
                this->model_validation_.biochemicalReactions_.at(this->model_validation_.reaction_ids_.at(nodes_iter)));
            input_data(batch_iter, memory_iter, nodes_iter, epoch_iter) = value;
          }
        }
      }
    }
  }
  void simulateDataSampleConcs(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps, const bool& train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_epochs = input_data.dimension(3);
    int n_input_pixels;
    if (train)
      n_input_pixels = this->model_training_.component_group_names_.size();
    else
      n_input_pixels = this->model_validation_.component_group_names_.size();

    assert(n_input_nodes == n_input_pixels);

    for (int epoch_iter = 0; epoch_iter < n_epochs; ++epoch_iter) {
      for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
        for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
          for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
            TensorT value;
            if (train)
              value = this->model_training_.getRandomConcentration(
                this->model_training_.metabolomicsData_.at(sample_group_name_),
                this->model_training_.component_group_names_.at(nodes_iter));
            else
              value = this->model_validation_.getRandomConcentration(
                this->model_validation_.metabolomicsData_.at(sample_group_name_),
                this->model_validation_.component_group_names_.at(nodes_iter));
            input_data(batch_iter, memory_iter, nodes_iter, epoch_iter) = value;
          }
        }
      }
    }
  }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {
    if (simulate_MARs_) simulateDataMARs(input_data, time_steps, use_train_);
    else simulateDataSampleConcs(input_data, time_steps, use_train_);
  }

  BiochemicalReactionModel<TensorT> model_training_;
  BiochemicalReactionModel<TensorT> model_validation_;
  int n_encodings_;
  bool sample_concs_ = false;
  bool simulate_MARs_ = true;
  bool use_train_ = true;
  std::string sample_group_name_;
};

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief Fully connected variational reconstruction model
  */
  void makeModelFCVAE_Encoder(Model<TensorT>& model, const int& n_inputs, const int& n_encodings, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input, bool add_norm = true) {
    model.setId(0);
    model.setName("VAE-Encoder");
    const int n_en_hidden_0 = 64;
    const int n_en_hidden_1 = 64;
    const int n_en_hidden_2 = 0;
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

    // Specify the output node types manually
    for (const std::string& node_name : node_names_mu)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief Fully connected variational reconstruction model
  */
  void makeModelFCVAE_Decoder(Model<TensorT>& model, const int& n_outputs, const int& n_encodings, bool add_norm = true) {
    model.setId(0);
    model.setName("VAE-Decoder");
    const int n_de_hidden_0 = 64;
    const int n_de_hidden_1 = 64;
    const int n_de_hidden_2 = 0;
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Encoding", "Encoding", n_encodings, true);

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

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief Fully connected classifier
  */
  void makeModelFCClass(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, bool add_norm = true) {
    model.setId(0);
    model.setName("Classifier");

    const int n_hidden_0 = 64;
    const int n_hidden_1 = 0;
    const int n_hidden_2 = 0;

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

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
};

template<typename TensorT>
class LatentArithmetic {
public:
  LatentArithmetic(const int& encoding_size, const bool& simulate_MARs, const bool& sample_concs) :
    encoding_size(encoding_size), simulate_MARs(simulate_MARs), sample_concs(sample_concs) {};
  ~LatentArithmetic() = default;
  /*
  @brief Write the raw classification results to the console for debugging

  @param[in] classification_labels The labels used for classificaiton
  @param[in] classification_results Tensor of the classification output
    dimensions: 0: batch_size, 1: labels_size
  */
  void writeClassificationResults(const std::vector<std::string>& classification_labels, const Eigen::Tensor<TensorT, 2>& classification_results) {
    // Determine the dimensions and # of input
    const int batch_size = classification_results.dimension(0);
    const int labels_size = classification_results.dimension(1);
    assert(labels_size == classification_labels.size());

    for (const std::string& label : classification_labels) {
      std::cout << label << "[tab]";
    }
    std::cout << std::endl;
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int node_iter = 0; node_iter < labels_size; ++node_iter) {
        std::cout << classification_results(batch_iter, node_iter) << "[tab]";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  };

  /*
  @brief Calculate the percent true

  @param[in] classification_labels The labels used for classificaiton
  @param[in] classification_results Tensor of the classification output
    dimensions: 0: batch_size, 1: labels_size
  @param[in] class_expected_index Index for the true label

  @returns The percent true
  */
  TensorT calculatePercTrue(const std::vector<std::string>& classification_labels, const Eigen::Tensor<TensorT, 2>& classification_results, const int& class_expected_index) {
    // Determine the dimensions and # of input
    const int batch_size = classification_results.dimension(0);
    const int labels_size = classification_results.dimension(1);
    assert(labels_size == classification_labels.size());

    // Calculate the percent true
    Eigen::Tensor<TensorT, 3> max_tensor = classification_results.reshape(Eigen::array<Eigen::Index, 3>({ batch_size, labels_size, 1 }));
    auto max_bcast = max_tensor.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, labels_size }));
    auto selected = (max_bcast == classification_results).select(classification_results.constant(1), classification_results.constant(0));
    Eigen::Tensor<TensorT, 0> predicted_true = selected.chip(class_expected_index, 1).sum();
    TensorT perc_true = predicted_true(0) / batch_size;
    return perc_true;
  };

  /*
  @brief Read in and create the metabolomics data

  @param[in] biochem_rxns_filename
  @param[in] metabo_data_filename_train
  @param[in] meta_data_filename_train
  @param[in] metabo_data_filename_test
  @param[in] meta_data_filename_test
  */
  void setMetabolomicsData(const std::string& biochem_rxns_filename,
    const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
    const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test) {
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
    metabolomics_data.use_train_ = true;

    // Checks for the training and validation data
    assert(metabolomics_data.model_validation_.reaction_ids_.size() == metabolomics_data.model_training_.reaction_ids_.size());
    assert(metabolomics_data.model_validation_.labels_.size() == metabolomics_data.model_training_.labels_.size());
    assert(metabolomics_data.model_validation_.component_group_names_.size() == metabolomics_data.model_training_.component_group_names_.size());

    // Set the encoding size and define the input/output sizes
    metabolomics_data.n_encodings_ = encoding_size;
    if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
    else n_input_nodes = reaction_model.component_group_names_.size();
    n_output_nodes = reaction_model.labels_.size();
  };

  /*
  @brief Make the models

  @param[in] model_encoder_weights_filename
  @param[in] model_decoder_weights_filename
  */
  void setEncDecModels(ModelTrainerExt<TensorT>& model_trainer, const std::string& model_encoder_weights_filename, const std::string& model_decoder_weights_filename) {
    // initialize the models
    model_encoder.clear();
    model_decoder.clear();

    // define the encoder and decoders
    model_trainer.makeModelFCVAE_Encoder(model_encoder, n_input_nodes, encoding_size, true, false, false, false); // normalization type 1
    model_trainer.makeModelFCVAE_Decoder(model_decoder, n_input_nodes, encoding_size, false);

    // read in the encoder and decoder weights
    WeightFile<TensorT> data;
    data.loadWeightValuesCsv(model_encoder_weights_filename, model_encoder.getWeightsMap());
    data.loadWeightValuesCsv(model_decoder_weights_filename, model_decoder.getWeightsMap());

    // check that all weights were read in correctly
    for (auto& weight_map : model_encoder.getWeightsMap()) {
      if (weight_map.second->getInitWeight()) {
        std::cout << "Model " << model_encoder.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
      }
    }
    for (auto& weight_map : model_decoder.getWeightsMap()) {
      if (weight_map.second->getInitWeight()) {
        std::cout << "Model " << model_decoder.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
      }
    }
  };

  /*
  @brief Make the Classification model

  @param[in] model_classifier_weights_filename
  */
  void setClassifierModel(ModelTrainerExt<TensorT>& model_trainer, const std::string& model_classifier_weights_filename) {
    // initialize the models
    model_classifier.clear();

    // define the encoder and decoders
    model_trainer.makeModelFCClass(model_classifier, n_input_nodes, n_output_nodes, false);

    // read in the encoder and decoder weights
    WeightFile<TensorT> data;
    data.loadWeightValuesCsv(model_classifier_weights_filename, model_classifier.getWeightsMap());

    // check that all weights were read in correctly
    for (auto& weight_map : model_classifier.getWeightsMap()) {
      if (weight_map.second->getInitWeight()) {
        std::cout << "Model " << model_classifier.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
      }
    }
  };

  /*
  @brief Generate a reconstruction based on the arithmetic between two latent spaces

  @param[in] sample_group_name_1
  @param[in] sample_group_name_2
  @param[in] class_expected_index
  @param[in] latent_arithmetic Options are the following
    "None": no latent arithmetic is performed and only sample_group_1 is ran through the encoder, decoder, and classification
    "Test": no latent arithmetic is performed and only the input for sample_group_1 is ran the the classifier
    "-": subtraction is performed
    "+": addition is performed

  TODO: break up into generateEncoding and generateReconstruction
  */
  Eigen::Tensor<TensorT, 4> calculateLatentArithmetic(
    const std::string& sample_group_name_1, const std::string& sample_group_name_2, const std::string& latent_arithmetic, 
    ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger, ModelInterpreterDefaultDevice<TensorT>& model_interpreter)
  {
    std::cout << sample_group_name_1 << " " << latent_arithmetic << " " << sample_group_name_2 << " =" << std::endl;
    
    // Make the input nodes
    std::vector<std::string> input_nodes;
    for (int i = 0; i < n_input_nodes; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }

    // Make the reconstruction nodes
    std::vector<std::string> output_nodes_reconstruction;
    for (int i = 0; i < n_input_nodes; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes_reconstruction.push_back(name);
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

    // Make the encoding nodes
    std::vector<std::string> encoding_nodes;
    for (int i = 0; i < encoding_size; ++i) {
      char name_char[512];
      sprintf(name_char, "Encoding_%012d", i);
      std::string name(name_char);
      encoding_nodes.push_back(name);
    }

    // generate the input for condition_1 and condition_2
    Eigen::Tensor<TensorT, 4> condition_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    metabolomics_data.sample_group_name_ = sample_group_name_1;
    metabolomics_data.simulateEvaluationData(condition_1_input, time_steps_1_input);
    Eigen::Tensor<TensorT, 4> condition_2_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_2_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    metabolomics_data.sample_group_name_ = sample_group_name_2;
    if (latent_arithmetic != "None" && latent_arithmetic != "Test")
      metabolomics_data.simulateEvaluationData(condition_2_input, time_steps_2_input);

    // evaluate the encoder for condition_1 and condition_2
    model_trainer.setLossOutputNodes({ encoding_nodes_mu });
    Eigen::Tensor<TensorT, 4> condition_1_output(model_trainer.getBatchSize(), model_trainer.getMemorySize(), encoding_size, model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 4> condition_2_output(model_trainer.getBatchSize(), model_trainer.getMemorySize(), encoding_size, model_trainer.getNEpochsEvaluation());
    condition_1_output = model_trainer.evaluateModel(
      model_encoder, condition_1_input, time_steps_1_input, input_nodes, model_logger, model_interpreter);
    if (latent_arithmetic != "None" && latent_arithmetic != "Test")
      condition_2_output = model_trainer.evaluateModel(
        model_encoder, condition_2_input, time_steps_2_input, input_nodes, model_logger, model_interpreter);

    // perform an embeddings subtraction or addition using condition_1 and condition_2
    Eigen::Tensor<TensorT, 4> embeddings_calculated(model_trainer.getBatchSize(), model_trainer.getMemorySize(), encoding_size, model_trainer.getNEpochsEvaluation());
    for (int epoch_iter = 0; epoch_iter < model_trainer.getNEpochsEvaluation(); ++epoch_iter) {
      for (int node_iter = 0; node_iter < encoding_size; ++node_iter) {
        // Could be improved by using the `.slice` operator
        for (int batch_iter = 0; batch_iter < model_trainer.getBatchSize(); ++batch_iter) {
          for (int memory_iter = 0; memory_iter < model_trainer.getMemorySize(); ++memory_iter) {
            if (latent_arithmetic == "None" || latent_arithmetic == "Test")
              embeddings_calculated(batch_iter, memory_iter, node_iter, epoch_iter) = condition_1_output(batch_iter, memory_iter, node_iter, epoch_iter);
            else if (latent_arithmetic == "-")
              embeddings_calculated(batch_iter, memory_iter, node_iter, epoch_iter) = condition_1_output(batch_iter, memory_iter, node_iter, epoch_iter) - condition_2_output(batch_iter, memory_iter, node_iter, epoch_iter);
            else if (latent_arithmetic == "+")
              embeddings_calculated(batch_iter, memory_iter, node_iter, epoch_iter) = condition_1_output(batch_iter, memory_iter, node_iter, epoch_iter) + condition_2_output(batch_iter, memory_iter, node_iter, epoch_iter);
            else
              std::cout << "Latent arithmetic " << latent_arithmetic << " is not supported." << std::endl;
          }
        }
      }
    }

    // evaluate the decoder
    Eigen::Tensor<TensorT, 4> reconstructed_output(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, model_trainer.getNEpochsEvaluation());
    model_trainer.setLossOutputNodes({ output_nodes_reconstruction });
    if (latent_arithmetic == "Test")
      reconstructed_output = condition_1_input;
    else
      reconstructed_output = model_trainer.evaluateModel(
        model_decoder, embeddings_calculated, time_steps_1_input, encoding_nodes, model_logger, model_interpreter);

    return reconstructed_output;
  }

  /*
  @brief Generate an encoded latent space

  @param[in] sample_group_name

  @returns 4D Tensor of the encoded latent space
  */
  Eigen::Tensor<TensorT, 4> generateEncoding(
    const std::string& sample_group_name, 
    ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger, ModelInterpreterDefaultDevice<TensorT>& model_interpreter)
  {
    // Make the input nodes
    std::vector<std::string> input_nodes;
    for (int i = 0; i < n_input_nodes; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }

    // Make the mu nodes
    std::vector<std::string> encoding_nodes_mu;
    for (int i = 0; i < encoding_size; ++i) {
      char name_char[512];
      sprintf(name_char, "Mu_%012d", i);
      std::string name(name_char);
      encoding_nodes_mu.push_back(name);
    }

    // generate the input for condition_1 and condition_2
    Eigen::Tensor<TensorT, 4> condition_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    metabolomics_data.sample_group_name_ = sample_group_name;
    metabolomics_data.simulateEvaluationData(condition_1_input, time_steps_1_input);;

    // evaluate the encoder for condition_1 and condition_2
    model_trainer.setLossOutputNodes({ encoding_nodes_mu });
    Eigen::Tensor<TensorT, 4> condition_1_output = model_trainer.evaluateModel(
      model_encoder, condition_1_input, time_steps_1_input, input_nodes, model_logger, model_interpreter);

    return condition_1_output;
  }

  /*
  @brief Generate a reconstruction of the latent space

  @param[in] encoding_output 4D Tensor of the encoded latent space

  @returns 4D Tensor of the reconstruction
  */
  Eigen::Tensor<TensorT, 4> generateReconstruction(
    Eigen::Tensor<TensorT, 4>& encoding_output,
    ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger, ModelInterpreterDefaultDevice<TensorT>& model_interpreter)
  {
    // Make the reconstruction nodes
    std::vector<std::string> output_nodes_reconstruction;
    for (int i = 0; i < n_input_nodes; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes_reconstruction.push_back(name);
    }

    // Make the encoding nodes
    std::vector<std::string> encoding_nodes;
    for (int i = 0; i < encoding_size; ++i) {
      char name_char[512];
      sprintf(name_char, "Encoding_%012d", i);
      std::string name(name_char);
      encoding_nodes.push_back(name);
    }

    // evaluate the decoder
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    model_trainer.setLossOutputNodes({ output_nodes_reconstruction });
    Eigen::Tensor<TensorT, 4> reconstructed_output = model_trainer.evaluateModel(
        model_decoder, encoding_output, time_steps_1_input, encoding_nodes, model_logger, model_interpreter);

    return reconstructed_output;
  }

  /*
  @brief Classify a reconstruction or input using a classification model

  @param[in] sample_group_name_expected
  */
  void classifyReconstruction(const std::string& sample_group_name_expected, const Eigen::Tensor<TensorT, 4>& reconstructed_output,
    ModelTrainerDefaultDevice<TensorT>& model_trainer, ModelLogger<TensorT>& model_logger, ModelInterpreterDefaultDevice<TensorT>& model_interpreter) {

    // Make the input nodes
    std::vector<std::string> input_nodes;
    for (int i = 0; i < n_input_nodes; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }

    // Make the classification nodes
    std::vector<std::string> output_nodes_classification;
    for (int i = 0; i < n_output_nodes; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes_classification.push_back(name);
    }
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());

    // score the decoded data using the classification model
    model_trainer.setLossOutputNodes({ output_nodes_classification });
    Eigen::Tensor<TensorT, 4> classification_output = model_trainer.evaluateModel(
      model_classifier, reconstructed_output, time_steps_1_input, input_nodes, model_logger, model_interpreter);

    const Eigen::Tensor<TensorT, 2> classification_results = classification_output.chip(0, 3).chip(0, 1);
    // write out the results to the console
    writeClassificationResults(reaction_model.labels_, classification_results);

    // calculate the percent true
    auto labels_iterator = std::find(reaction_model.labels_.begin(), reaction_model.labels_.end(), sample_group_name_expected);
    if (labels_iterator != reaction_model.labels_.end()) {
      int class_expected_index = -1;
      class_expected_index = std::distance(reaction_model.labels_.begin(), labels_iterator);
      TensorT perc_true = calculatePercTrue(reaction_model.labels_, classification_results, class_expected_index);
      std::cout << "Percent true: " << perc_true << std::endl;
    }
    else {
      std::cout << sample_group_name_expected << " was not found in the labels." << std::endl;
    }
  };

  /*
  @brief Score a reconstruction or input using a similarity metric function

  @param[in] sample_group_name_expected
  */
  void scoreReconstructionSimilarity(const std::string& sample_group_name_expected, const Eigen::Tensor<TensorT, 4>& reconstructed_output,
    MetricFunctionTensorOp<TensorT, Eigen::DefaultDevice>& metric_function, ModelTrainerDefaultDevice<TensorT>& model_trainer) {

    // generate the input for the expected
    Eigen::Tensor<TensorT, 4> condition_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, model_trainer.getNEpochsEvaluation());
    Eigen::Tensor<TensorT, 3> time_steps_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
    metabolomics_data.sample_group_name_ = sample_group_name_expected;
    metabolomics_data.simulateEvaluationData(condition_1_input, time_steps_1_input);

    // score the decoded data using the classification model
    Eigen::Tensor<TensorT, 3> expected = condition_1_input.chip(0, 1);
    Eigen::Tensor<TensorT, 2> predicted = reconstructed_output.chip(0, 3).chip(0, 1);
    Eigen::Tensor<TensorT, 2> score(1, 1); score.setZero();
    Eigen::DefaultDevice device;
    metric_function(predicted.data(), expected.data(), score.data(), model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, 1, 0, 0, device);
    std::cout << "Score: " << score(0,0)/TensorT(model_trainer.getBatchSize()) << std::endl;

  };

  int getNInputNodes() const { return n_input_nodes; }
  int getNOutputNodes() const { return n_output_nodes; }

  int encoding_size = 16;
  bool simulate_MARs = false;
  bool sample_concs = true;

protected:
  /// Defined in setMetabolomicsData
  MetDataSimLatentArithmetic<TensorT> metabolomics_data;
  BiochemicalReactionModel<TensorT> reaction_model;
  int n_input_nodes = -1;
  int n_output_nodes = -1; 

  /// Defined in setEncDecModels
  Model<TensorT> model_decoder;
  Model<TensorT> model_encoder;

  /// Defined in setClassifierModels
  Model<TensorT> model_classifier;
};

// Main
int main(int argc, char** argv)
{
  // Set the data directories
  //const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "/home/user/Data/";

  // Make the filenames
  const std::string biochem_rxns_filename = data_dir + "iJO1366.csv";
  const std::string model_encoder_weights_filename = data_dir + "TrainTestData/SampledArithmeticMath/VAE_4000_weights.csv"; // encoding size of 3
  const std::string model_decoder_weights_filename = data_dir + "TrainTestData/SampledArithmeticMath/VAE_4000_weights.csv"; // encoding size of 6
  // NOTE: be sure to re-name the Input_000000000000-LinearScale_to_... weights to Input_000000000000_to_...
  //       using regex "-LinearScale_to_FC0" with "_to_FC0"
  const std::string model_classifier_weights_filename = data_dir + "TrainTestData/SampledArithmeticMath/Classifier_5000_weights.csv";

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

  // define the model trainers and resources for the trainers
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(128);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochsEvaluation(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(false, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // read in the metabolomics data and models
  LatentArithmetic<float> latentArithmetic(16, false, true);
  latentArithmetic.setMetabolomicsData(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test);
  latentArithmetic.setEncDecModels(model_trainer, model_encoder_weights_filename, model_decoder_weights_filename);
  latentArithmetic.setClassifierModel(model_trainer, model_classifier_weights_filename);

  // define the reconstruction output
  Eigen::Tensor<float, 4> reconstruction_output(model_trainer.getBatchSize(), model_trainer.getMemorySize(), latentArithmetic.getNInputNodes(), model_trainer.getNEpochsEvaluation());

  // 0. Control (Test and None)
  const std::vector<std::string> condition_1_test_none = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
    "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
  const std::vector<std::string> condition_2_test_none = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
    "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
  const std::vector<std::string> expected_test_none = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
    "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
  assert(condition_1_test_none.size() == condition_2_test_none.size());
  assert(expected_test_none.size() == condition_2_test_none.size());
  assert(condition_1_test_none.size() == expected_test_none.size());

  //std::cout << "Running Control (Test)" << std::endl;
  //for (int case_iter = 0; case_iter < condition_1_test_none.size(); ++case_iter) {
  //  reconstruction_output = latentArithmetic.calculateLatentArithmetic(condition_1_test_none.at(case_iter), condition_2_test_none.at(case_iter), "Test", model_trainer, model_logger, model_interpreter);
  //  latentArithmetic.classifyReconstruction(expected_test_none.at(case_iter), reconstruction_output, model_trainer, model_logger, model_interpreter);
  //}

  std::cout << "Running Control (None)" << std::endl;
  for (int case_iter = 0; case_iter < condition_1_test_none.size(); ++case_iter) {
    std::cout << condition_1_test_none.at(case_iter) << " None:" << std::endl;
    auto encoding_output = latentArithmetic.generateEncoding(condition_1_test_none.at(case_iter), model_trainer, model_logger, model_interpreter);
    reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger, model_interpreter);
    latentArithmetic.scoreReconstructionSimilarity(expected_test_none.at(case_iter), reconstruction_output, PearsonRTensorOp<float, Eigen::DefaultDevice>(), model_trainer);
  }

  // 1. EPi - KO -> Ref
  const std::vector<std::string> condition_1_arithmetic_1 = { "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
    "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
  const std::vector<std::string> condition_2_arithmetic_1 = { "Evo04gnd", "Evo04gnd", "Evo04pgi", "Evo04pgi",
    "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04sdhCB", "Evo04sdhCB", "Evo04tpiA", "Evo04tpiA" };
  const std::vector<std::string> expected_arithmetic_1 = { "Evo04", "Evo04", "Evo04", "Evo04",
    "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04" };  
  assert(condition_1_arithmetic_1.size() == condition_2_arithmetic_1.size());
  assert(expected_arithmetic_1.size() == condition_2_arithmetic_1.size());
  assert(condition_1_arithmetic_1.size() == expected_arithmetic_1.size());

  std::cout << "Running EPi - KO -> Ref" << std::endl;
  for (int case_iter = 0; case_iter < condition_1_arithmetic_1.size(); ++case_iter) {
    auto encoding_output_1 = latentArithmetic.generateEncoding(condition_1_arithmetic_1.at(case_iter), model_trainer, model_logger, model_interpreter);
    auto encoding_output_2 = latentArithmetic.generateEncoding(condition_2_arithmetic_1.at(case_iter), model_trainer, model_logger, model_interpreter);
    Eigen::Tensor<float, 4> encoding_output = encoding_output_1 - encoding_output_2;
    reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger, model_interpreter);
    latentArithmetic.scoreReconstructionSimilarity(expected_arithmetic_1.at(case_iter), reconstruction_output, PearsonRTensorOp<float, Eigen::DefaultDevice>(), model_trainer);
  }

  // 2. EPi - Ref -> KO
  const std::vector<std::string> condition_1_arithmetic_2 = { "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04pgiEvo01EP", "Evo04pgiEvo02EP",
    "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo02EP", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04tpiAEvo01EP", "Evo04tpiAEvo02EP" };
  const std::vector<std::string> condition_2_arithmetic_2 = { "Evo04", "Evo04", "Evo04", "Evo04",
    "Evo04", "Evo04", "Evo04", "Evo04", "Evo04", "Evo04" };
  const std::vector<std::string> expected_arithmetic_2 = { "Evo04gnd", "Evo04gnd", "Evo04pgi", "Evo04pgi",
    "Evo04ptsHIcrr", "Evo04ptsHIcrr", "Evo04sdhCB", "Evo04sdhCB", "Evo04tpiA", "Evo04tpiA" };
  assert(condition_1_arithmetic_2.size() == condition_2_arithmetic_2.size());
  assert(expected_arithmetic_2.size() == condition_2_arithmetic_2.size());
  assert(condition_1_arithmetic_2.size() == expected_arithmetic_2.size());

  std::cout << "Running EPi - Ref -> KO" << std::endl;
  for (int case_iter = 0; case_iter < condition_1_arithmetic_2.size(); ++case_iter) {
    auto encoding_output_1 = latentArithmetic.generateEncoding(condition_1_arithmetic_2.at(case_iter), model_trainer, model_logger, model_interpreter);
    auto encoding_output_2 = latentArithmetic.generateEncoding(condition_2_arithmetic_2.at(case_iter), model_trainer, model_logger, model_interpreter);
    Eigen::Tensor<float, 4> encoding_output = encoding_output_1 - encoding_output_2;
    reconstruction_output = latentArithmetic.generateReconstruction(encoding_output, model_trainer, model_logger, model_interpreter);
    latentArithmetic.scoreReconstructionSimilarity(expected_arithmetic_2.at(case_iter), reconstruction_output, PearsonRTensorOp<float, Eigen::DefaultDevice>(), model_trainer);
  }

  //// 3. KOi + Ref -> EPi
  //std::cout << "Running KOi + Ref -> EPi" << std::endl;
  //latentArithmetic.calculateLatentArithmetic("Evo04gnd", "Evo04", 2, "+");
  ////...

  return 0;
}