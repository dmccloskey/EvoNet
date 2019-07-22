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
    assert(n_input_nodes == n_input_pixels + n_encodings_);

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
            input_data(epoch_iter, batch_iter, memory_iter, nodes_iter) = value;
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

    assert(n_input_nodes == n_input_pixels + n_encodings_);

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
            input_data(batch_iter, memory_iter, nodes_iter) = value;
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
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Mu", "Mu", n_encodings, true);

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

/// Script to run the reconstruction network
void main_latentArithmetic(const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
  const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
  const std::string& model_encoder_weights_filename, const std::string& model_decoder_weights_filename, const std::string& model_classifier_weights_filename, 
  const std::string& sample_group_name_1, const std::string& sample_group_name_2,
  const bool& embeddings_subtraction,
  bool make_model = true, bool simulate_MARs = true, bool sample_concs = true)
{
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
  MetDataSimLatentArithmetic<float> metabolomics_data;
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
  reaction_model.findLabels("subject");
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
  reaction_model.findLabels("subject");
  metabolomics_data.model_validation_ = reaction_model;
  metabolomics_data.simulate_MARs_ = simulate_MARs;
  metabolomics_data.sample_concs_ = sample_concs;
  metabolomics_data.use_train_ = true;

  // Checks for the training and validation data
  assert(metabolomics_data.model_validation_.reaction_ids_.size() == metabolomics_data.model_training_.reaction_ids_.size());
  assert(metabolomics_data.model_validation_.labels_.size() == metabolomics_data.model_training_.labels_.size());
  assert(metabolomics_data.model_validation_.component_group_names_.size() == metabolomics_data.model_training_.component_group_names_.size());

  // Define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = n_input_nodes;
  const int encoding_size = 8;
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

  //// Make the encoding nodes and add them to the input
  //for (int i = 0; i < encoding_size; ++i) {
  //  char name_char[512];
  //  sprintf(name_char, "Encoding_%012d-Sampler", i);
  //  std::string name(name_char);
  //  input_nodes.push_back(name);
  //}

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

  // define the encoder and decoders
  std::cout << "Initializing the models..." << std::endl;
  Model<float> model_decoder, model_encoder, model_classifier;
  model_trainer.makeModelFCVAE_Encoder(model_encoder, n_input_nodes, encoding_size, true, false, false, false); // normalization type 1
  model_trainer.makeModelFCVAE_Decoder(model_decoder, n_output_nodes, encoding_size, false); // normalization type 1

  // read in the encoder and decoder weights
  ModelFile<float> data;
  data.loadModelCsv("", "", model_encoder_weights_filename, model_encoder, false, false, true);
  data.loadModelCsv("", "", model_decoder_weights_filename, model_decoder, false, false, true);
  data.loadModelCsv("", "", model_classifier_weights_filename, model_classifier, false, false, true);
  // CHECK that the weights will not be re-initialized by the model interpreter...
   
  // generate the input for condition_1 and condition_2
  Eigen::Tensor<float, 4> condition_1_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, model_trainer.getNEpochsEvaluation());
  metabolomics_data.sample_group_name_ = sample_group_name_1;
  metabolomics_data.simulateEvaluationData(condition_1_input, Eigen::Tensor<float, 3>());
  Eigen::Tensor<float, 4> condition_2_input(model_trainer.getBatchSize(), model_trainer.getMemorySize(), n_input_nodes, model_trainer.getNEpochsEvaluation());
  metabolomics_data.sample_group_name_ = sample_group_name_2;
  metabolomics_data.simulateEvaluationData(condition_2_input, Eigen::Tensor<float, 3>());

  // evaluate the encoder for condition_1 and condition_2
  Eigen::Tensor<float, 4> condition_1_output = model_trainer.evaluateModel(
    model_encoder, condition_1_input, Eigen::Tensor<float, 3>(), input_nodes, ModelLogger<float>(), model_interpreter);
  Eigen::Tensor<float, 4> condition_2_output = model_trainer.evaluateModel(
    model_encoder, condition_2_input, Eigen::Tensor<float, 3>(), input_nodes, ModelLogger<float>(), model_interpreter);

  // perform an embeddings subtraction or addition using condition_1 and condition_2
  Eigen::Tensor<float, 4> embeddings_calculated(model_trainer.getBatchSize(), model_trainer.getMemorySize(), encoding_size, model_trainer.getNEpochsEvaluation());
  for (int epoch_iter = 0; epoch_iter < model_trainer.getNEpochsEvaluation(); ++epoch_iter) {
    for (int node_iter = 0; node_iter < encoding_size; ++node_iter) {
      // Could be improved by using the `.slice` operator
      for (int batch_iter = 0; batch_iter < model_trainer.getBatchSize(); ++batch_iter) {
        for (int memory_iter = 0; memory_iter < model_trainer.getMemorySize(); ++memory_iter)
        if (embeddings_subtraction) {
          embeddings_calculated(batch_iter, memory_iter, node_iter, epoch_iter) = condition_1_output(batch_iter, memory_iter, node_iter, epoch_iter) - condition_2_output(batch_iter, memory_iter, node_iter, epoch_iter);
        }
        else {
          embeddings_calculated(batch_iter, memory_iter, node_iter, epoch_iter) = condition_1_output(batch_iter, memory_iter, node_iter, epoch_iter) + condition_2_output(batch_iter, memory_iter, node_iter, epoch_iter);
        }
      }
    }
  }

  // evaluate the decoder
  Eigen::Tensor<float, 4> reconstructed_output = model_trainer.evaluateModel(
    model_decoder, embeddings_calculated, Eigen::Tensor<float, 3>(), encoding_nodes_mu, ModelLogger<float>(), model_interpreter);

  // score the decoded data using the classification model
  Eigen::Tensor<float, 4> classification_output = model_trainer.evaluateModel(
    model_classifier, reconstructed_output, Eigen::Tensor<float, 3>(), input_nodes, ModelLogger<float>(), model_interpreter);
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
  const std::string model_encoder_weights_filename = data_dir + "Metabolomics_VAE.csv";
  const std::string model_decoder_weights_filename = data_dir + "Metabolomics_VAE.csv";
  const std::string model_classifier_weights_filename = data_dir + "Metabolomics_Classification.csv";

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

  // 1. EPi - KO -> Ref
  main_latentArithmetic(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test, 
    model_encoder_weights_filename, model_decoder_weights_filename, model_classifier_weights_filename,
    "Evo04gndEvo01EP", "Evo04gnd", true, false, true);
  main_latentArithmetic(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test,
    model_encoder_weights_filename, model_decoder_weights_filename, model_classifier_weights_filename,
    "Evo04gndEvo02EP", "Evo04gnd", true, false, true);
  //...

  // 2. EPi - Ref -> KO
  main_latentArithmetic(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test,
    model_encoder_weights_filename, model_decoder_weights_filename, model_classifier_weights_filename,
    "Evo04gndEvo01EP", "Evo04", true, false, true);
  main_latentArithmetic(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test,
    model_encoder_weights_filename, model_decoder_weights_filename, model_classifier_weights_filename,
    "Evo04gndEvo02EP", "Evo04", true, false, true);
  //...

  // 3. KOi + Ref -> EPi
  main_latentArithmetic(biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test,
    model_encoder_weights_filename, model_decoder_weights_filename, model_classifier_weights_filename,
    "Evo04gnd", "Evo04", false, false, true);
  //...

  return 0;
}