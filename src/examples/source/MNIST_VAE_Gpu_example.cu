/**TODO:  Add copyright*/

#include <EvoNet/ml/PopulationTrainerGpu.h>
#include <EvoNet/ml/ModelTrainerGpu.h>
#include <EvoNet/ml/ModelReplicator.h>
#include <EvoNet/ml/ModelBuilder.h>
#include <EvoNet/ml/Model.h>
#include <EvoNet/io/PopulationTrainerFile.h>
#include <EvoNet/io/ModelInterpreterFileGpu.h>
#include <EvoNet/io/ModelFile.h>

#include <EvoNet/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace EvoNet;

// Extended 
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerGpu<TensorT>
{
public:
  bool KL_divergence_warmup_ = false;
  TensorT beta_ = 1;
  TensorT capacity_c_ = 0;
  /*
  @brief Basic VAE with	Xavier-like initialization

  References:
  Based on Kingma et al, 2014: https://arxiv.org/pdf/1312.6114
  https://github.com/pytorch/examples/blob/master/vae/main.py

  @param[in, out] model The network model
  @param[in] n_inputs The number of input pixels
  @param[in] n_encodings The length of the encodings layer
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeVAEFullyConn(Model<TensorT>& model,
    const int& n_inputs = 784, const int& n_encodings = 64, const int& n_hidden_0 = 512, const int& n_hidden_1 = 256, const int& n_hidden_2 = 64,
    const bool& add_norm = true, const bool& add_bias = true, const bool& specify_layers = false) {
    model.setId(0);
    model.setName("VAE");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation, activation_grad;
    if (add_norm) {
      activation = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
      activation_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());
    }
    else {
      activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
      activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
      //activation = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>(1e-24, 0, 1));
      //activation_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());
    }
    std::shared_ptr<ActivationOp<TensorT>> activation_norm = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_norm_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    //std::shared_ptr<ActivationOp<TensorT>> activation_norm = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>(1e-24, 0, 1));
    //std::shared_ptr<ActivationOp<TensorT>> activation_norm_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-5, 0.9, 0.999, 1e-8, 10));

    // Add the Encoder FC layers
    std::vector<std::string> node_names_mu, node_names_logvar;
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN0-Norm", "EN0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN0-Norm-gain", "EN0-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
        //std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN1-Norm", "EN1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN1-Norm-gain", "EN1-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
        //std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      }
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "EN2-Norm", "EN2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "EN2-Norm-gain", "EN2-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
        //std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      }
    }
    node_names_mu = model_builder.addFullyConnected(model, "MuEnc", "MuEnc", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names_logvar = model_builder.addFullyConnected(model, "LogVarEnc", "LogVarEnc", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Add the Encoding layers
    node_names = model_builder.addGaussianEncoding(model, "Encoding", "Encoding", node_names_mu, node_names_logvar, specify_layers);

    // Add the Decoder FC layers
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE2-Norm", "DE2-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE2-Norm-gain", "DE2-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
        //std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      }
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE1-Norm", "DE1-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE1-Norm-gain", "DE1-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
        //std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      }
    }
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      if (add_norm) {
        node_names = model_builder.addNormalization(model, "DE0-Norm", "DE0-Norm", node_names, true);
        node_names = model_builder.addSinglyConnected(model, "DE0-Norm-gain", "DE0-Norm-gain", node_names, node_names.size(),
          activation_norm, activation_norm_grad,
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op, 0.0, 0.0, true, specify_layers);
        //std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layers);
      }
    }
    node_names = model_builder.addFullyConnected(model, "DE-Output", "DE-Output", node_names, n_inputs,
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()),
      //std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),
      activation_norm, activation_norm_grad,
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, add_bias, true);

    // Add the actual output nodes
    node_names_mu = model_builder.addSinglyConnected(model, "Mu", "Mu", node_names_mu, node_names_mu.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    node_names_logvar = model_builder.addSinglyConnected(model, "LogVar", "LogVar", node_names_logvar, node_names_logvar.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_mu)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_logvar)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  /*
  @brief Convolution Variational Autoencoder

  References:
  Inspired by: https://github.com/pytorch/examples/blob/master/mnist/main.py
  Based on Dupont et al, 2018: 	arXiv:1804.00104
  https://github.com/Schlumberger/joint-vae
  */
  void makeVAECovNet(Model<TensorT>& model, const int& n_inputs, const int& n_encodings,
    const int& n_enc_depth_1 = 32, const int& n_enc_depth_2 = 2, const int& n_enc_depth_3 = 2,
    const int& n_dec_depth_1 = 2, const int& n_dec_depth_2 = 2, const int& n_dec_depth_3 = 1,
    const int& n_enc_fc_1 = 128, const int& n_dec_fc_1 = 126, const int& filter_size = 4, const int& stride_size = 2, const bool& add_norm = false, const bool& specify_layers = false) {
    model.setId(0);
    model.setName("VAE");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation, activation_grad;
    if (add_norm) {
      activation = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
      activation_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());
    }
    else {
      activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
      activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    }

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(5e-4, 0.9, 0.999, 1e-8, 10));

    // Add the first convolution -> ReLU layers
    std::vector<std::vector<std::string>> node_names_l0;
    for (size_t d = 0; d < n_enc_depth_1; ++d) {
      std::vector<std::string> node_names;
      std::string conv_name = "Enc-Conv0-" + std::to_string(d);
      node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_input,
        sqrt(node_names_input.size()), sqrt(node_names_input.size()), 0, 0,
        filter_size, filter_size, stride_size, 0, 0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        std::string norm_name = "Enc-Norm0-" + std::to_string(d);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
        std::string gain_name = "Enc-Gain0-" + std::to_string(d);
        node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
      node_names_l0.push_back(node_names);
    }

    // Add the second convolution -> ReLU layers
    std::vector<std::vector<std::string>> node_names_l1;
    int l_cnt = 0;
    for (const std::vector<std::string>& node_names_l : node_names_l0) {
      for (size_t d = 0; d < n_enc_depth_2; ++d) {
        std::vector<std::string> node_names;
        std::string conv_name = "Enc-Conv1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_l,
          sqrt(node_names_l.size()), sqrt(node_names_l.size()), 0, 0,
          filter_size, filter_size, stride_size, 0, 0,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        if (add_norm) {
          std::string norm_name = "Enc-Norm1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
          std::string gain_name = "Enc-Gain1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
            std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
            std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
            integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
            solver_op,
            0.0, 0.0, true, specify_layers);
        }
        node_names_l1.push_back(node_names);
      }
      ++l_cnt;
    }

    // Add the third convolution -> ReLU layers
    std::vector<std::vector<std::string>> node_names_l2;
    l_cnt = 0;
    for (const std::vector<std::string>& node_names_l : node_names_l1) {
      for (size_t d = 0; d < n_enc_depth_3; ++d) {
        std::vector<std::string> node_names;
        std::string conv_name = "Enc-Conv2-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        node_names = model_builder.addConvolution(model, conv_name, conv_name, node_names_l,
          sqrt(node_names_l.size()), sqrt(node_names_l.size()), 0, 0,
          filter_size, filter_size, stride_size, 0, 0,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
        if (add_norm) {
          std::string norm_name = "Enc-Norm2-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, specify_layers);
          std::string gain_name = "Enc-Gain2-" + std::to_string(l_cnt) + "-" + std::to_string(d);
          node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
            std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
            std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
            integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
            solver_op,
            0.0, 0.0, true, specify_layers);
        }
        node_names_l2.push_back(node_names);
      }
      ++l_cnt;
    }

    // Linearize the node names
    std::vector<std::string> node_names_conv_linearized;
    int last_conv_depth;
    if (node_names_l2.size()) {
      for (const std::vector<std::string>& node_names_l : node_names_l2) {
        for (const std::string& node_name : node_names_l) {
          node_names_conv_linearized.push_back(node_name);
        }
      }
      last_conv_depth = n_enc_depth_3 * n_enc_depth_2 * n_enc_depth_1;
    }
    else if (node_names_l1.size()) {
      for (const std::vector<std::string>& node_names_l : node_names_l1) {
        for (const std::string& node_name : node_names_l) {
          node_names_conv_linearized.push_back(node_name);
        }
      }
      last_conv_depth = n_enc_depth_2 * n_enc_depth_1;
    }
    else {
      for (const std::vector<std::string>& node_names_l : node_names_l0) {
        for (const std::string& node_name : node_names_l) {
          node_names_conv_linearized.push_back(node_name);
        }
      }
      last_conv_depth = n_enc_depth_1;
    }

    // Add the FC layers
    std::vector<std::string> node_names_enc_fc = model_builder.addFullyConnected(model, "Enc-FC0", "Enc-FC0", node_names_conv_linearized, n_enc_fc_1,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names_conv_linearized.size() + n_enc_fc_1, 2)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      node_names_enc_fc = model_builder.addNormalization(model, "Enc-FC0-Norm", "Enc-FC0-Norm", node_names_enc_fc, true);
      node_names_enc_fc = model_builder.addSinglyConnected(model, "Enc-FC0-Norm-gain", "Enc-FC0-Norm-gain", node_names_enc_fc, node_names_enc_fc.size(),
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        solver_op,
        0.0, 0.0, true, specify_layers);
    }

    // Add the Encoding layers
    std::vector<std::string> node_names_mu = model_builder.addFullyConnected(model, "MuEnc", "MuEnc", node_names_enc_fc, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_enc_fc.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    std::vector<std::string> node_names_logvar = model_builder.addFullyConnected(model, "LogVarEnc", "LogVarEnc", node_names_enc_fc, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_enc_fc.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Add the Encoding layers
    std::vector<std::string> node_names_encoder = model_builder.addGaussianEncoding(model, "Encoding", "Encoding", node_names_mu, node_names_logvar, specify_layers);

    // Add the Decoder FC layers
    std::vector<std::string> node_names_dec_fc0 = model_builder.addFullyConnected(model, "Dec-FC0", "Dec-FC0", node_names_encoder, n_dec_fc_1,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_encoder.size() + n_dec_fc_1) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    if (add_norm) {
      node_names_dec_fc0 = model_builder.addNormalization(model, "Dec-FC0-Norm", "Dec-FC0-Norm", node_names_dec_fc0, true);
      node_names_dec_fc0 = model_builder.addSinglyConnected(model, "Dec-FC0-Norm-gain", "Dec-FC0-Norm-gain", node_names_dec_fc0, node_names_dec_fc0.size(),
        std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
        std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
        integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        solver_op,
        0.0, 0.0, true, specify_layers);
    }

    // Add the Decoder FC layer to begin the transpose convolutions
    const int n_dec_fc = node_names_conv_linearized.size() / last_conv_depth;
    int node_iter = 0;
    std::vector<std::vector<std::string>> node_names_dec_fc1;
    for (size_t d = 0; d < last_conv_depth; ++d) {
      std::vector<std::string> node_names;
      std::string fc_name = "Dec-FC1-" + std::to_string(d);
      node_names = model_builder.addFullyConnected(model, fc_name, fc_name, node_names_dec_fc0, n_dec_fc,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_dec_fc) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, specify_layers);
      if (add_norm) {
        std::string norm_name = "Dec-FC1-Norm-" + std::to_string(d);
        node_names = model_builder.addNormalization(model, norm_name, norm_name, node_names, true);
        std::string gain_name = "Dec-FC1-Norm-gain-" + std::to_string(d);
        node_names = model_builder.addSinglyConnected(model, gain_name, gain_name, node_names, node_names.size(),
          std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>()), // Nonlinearity occures after the normalization
          std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>()),
          integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_op,
          0.0, 0.0, true, specify_layers);
      }
      node_names_dec_fc1.push_back(node_names);
      node_iter += n_dec_fc;
    }

    // Add the first transpose convolution -> ReLU layers
    node_names_l0.clear();
    l_cnt = 0;
    for (const std::vector<std::string>& node_names_l : node_names_dec_fc1) {
      for (size_t d = 0; d < n_dec_depth_1; ++d) {
        std::vector<std::string> node_names;
        std::string conv_module = "Dec-Conv0-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        std::string conv_name = "Dec-Conv0-" + std::to_string(d);
        if (l_cnt == 0) {
          node_names = model_builder.addConvolution(model, conv_name, conv_module, node_names_l,
            sqrt(node_names_l.size()), sqrt(node_names_l.size()), filter_size - 1, filter_size - 1,
            filter_size, filter_size, stride_size, 0, 0,
            activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
            solver_op, 0.0f, 0.0f, false, specify_layers);
          node_names_l0.push_back(node_names);
        }
        else {
          model_builder.addConvolution(model, conv_name, conv_module, node_names_l, node_names_l0.at(d),
            sqrt(node_names_l.size()), sqrt(node_names_l.size()), filter_size - 1, filter_size - 1,
            filter_size, filter_size, stride_size, 0, 0,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
            solver_op, 0.0f, 0.0f, specify_layers);
        }
        // TODO: Norms
      }
      ++l_cnt;
    }

    // Add the second transpose convolution -> ReLU layers
    node_names_l1.clear();
    l_cnt = 0;
    for (const std::vector<std::string>& node_names_l : node_names_l0) {
      for (size_t d = 0; d < n_dec_depth_2; ++d) {
        std::vector<std::string> node_names;
        std::string conv_module = "Dec-Conv1-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        std::string conv_name = "Dec-Conv1-" + std::to_string(d);
        if (l_cnt == 0) {
          node_names = model_builder.addConvolution(model, conv_name, conv_module, node_names_l,
            sqrt(node_names_l.size()), sqrt(node_names_l.size()), filter_size - 1, filter_size - 1,
            filter_size, filter_size, stride_size, 0, 0,
            activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
            solver_op, 0.0f, 0.0f, false, specify_layers);
          node_names_l1.push_back(node_names);
        }
        else {
          model_builder.addConvolution(model, conv_name, conv_module, node_names_l, node_names_l1.at(d),
            sqrt(node_names_l.size()), sqrt(node_names_l.size()), filter_size - 1, filter_size - 1,
            filter_size, filter_size, stride_size, 0, 0,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
            solver_op, 0.0f, 0.0f, specify_layers);
        }
      }
      ++l_cnt;
    }

    // Add the third transpose convolution -> ReLU layers
    node_names_l2.clear();
    l_cnt = 0;
    for (const std::vector<std::string>& node_names_l : node_names_l1) {
      for (size_t d = 0; d < n_dec_depth_3; ++d) {
        std::vector<std::string> node_names;
        std::string conv_module = "Dec-Conv2-" + std::to_string(l_cnt) + "-" + std::to_string(d);
        std::string conv_name = "Dec-Conv2-" + std::to_string(d);
        if (l_cnt == 0) {
          node_names = model_builder.addConvolution(model, conv_name, conv_module, node_names_l,
            sqrt(node_names_l.size()), sqrt(node_names_l.size()), filter_size - 1, filter_size - 1,
            filter_size, filter_size, stride_size, 0, 0,
            activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
            solver_op, 0.0f, 0.0f, false, specify_layers);
          node_names_l2.push_back(node_names);
        }
        else {
          model_builder.addConvolution(model, conv_name, conv_module, node_names_l, node_names_l2.at(d),
            sqrt(node_names_l.size()), sqrt(node_names_l.size()), filter_size - 1, filter_size - 1,
            filter_size, filter_size, stride_size, 0, 0,
            std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(filter_size * filter_size, 2)),
            solver_op, 0.0f, 0.0f, specify_layers);
        }
      }
      ++l_cnt;
    }
    // TODO: last layer should be 
      //std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()),
      //std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()),

    node_names_conv_linearized.clear();
    if (node_names_l2.size() > 0) {
      for (const std::vector<std::string>& node_names_l : node_names_l2) {
        for (const std::string& node_name : node_names_l) {
          node_names_conv_linearized.push_back(node_name);
        }
      }
    }
    else if (node_names_l1.size() > 0) {
      for (const std::vector<std::string>& node_names_l : node_names_l1) {
        for (const std::string& node_name : node_names_l) {
          node_names_conv_linearized.push_back(node_name);
        }
      }
    }
    else if (node_names_l0.size() > 0) {
      for (const std::vector<std::string>& node_names_l : node_names_l0) {
        for (const std::string& node_name : node_names_l) {
          node_names_conv_linearized.push_back(node_name);
        }
      }
    }
    else {
      for (const std::vector<std::string>& node_names_l : node_names_dec_fc1) {
        for (const std::string& node_name : node_names_l) {
          node_names_conv_linearized.push_back(node_name);
        }
      }
    }
    assert(node_names_conv_linearized.size() == n_inputs);

    // Add the actual output nodes
    node_names_mu = model_builder.addSinglyConnected(model, "Mu", "Mu", node_names_mu, node_names_mu.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    node_names_logvar = model_builder.addSinglyConnected(model, "LogVar", "LogVar", node_names_logvar, node_names_logvar.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    std::vector<std::string> node_names_output = model_builder.addSinglyConnected(model, "Output", "Output", node_names_conv_linearized, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_mu)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_logvar)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_output)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();

  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    //if (n_epochs % 1000 == 0 && n_epochs > 5000) {
    //  // anneal the learning rate by half on each plateau
    //  TensorT lr_new = this->reduceLROnPlateau(model_errors, 0.5, 100, 10, 0.05);
    //  if (lr_new < 1.0) {
    //    model_interpreter.updateSolverParams(0, lr_new);
    //    std::cout << "The learning rate has been annealed by a factor of " << lr_new << std::endl;
    //  }
    //}

    // copy the loss function helpers
    auto lossFunctionHelpers = this->getLossFunctionHelpers();

    // Increase the KL divergence beta and capacity
    TensorT beta = this->beta_;
    TensorT capacity_c = this->capacity_c_;
    if (this->KL_divergence_warmup_) {
      TensorT scale_factor1 = (n_epochs - 100 > 0) ? n_epochs - 100 : 1;
      beta /= (2.5e4 / scale_factor1);
      if (beta > this->beta_) beta = this->beta_;
      TensorT scale_factor2 = (n_epochs - 100 > 0) ? n_epochs - 100 : 1;
      capacity_c /= (2.5e4 / scale_factor2);
      if (capacity_c > this->capacity_c_) capacity_c = this->capacity_c_;
    }
    lossFunctionHelpers.at(1).loss_functions_.at(0) = std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, beta, capacity_c));
    lossFunctionHelpers.at(2).loss_functions_.at(0) = std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, beta, capacity_c));
    lossFunctionHelpers.at(1).loss_function_grads_.at(0) = std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, beta, capacity_c));
    lossFunctionHelpers.at(2).loss_function_grads_.at(0) = std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, beta, capacity_c));

    // Update the loss function helpers
    this->setLossFunctionHelpers(lossFunctionHelpers);
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;

      //// save the model weights to .csv
      //data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model, false, false, true);

      // save the model and tensors to binary
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
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_train_headers = { "Train_Error" };
    std::vector<std::string> log_test_headers = { "Test_Error" };
    std::vector<TensorT> log_train_values = { model_error_train };
    std::vector<TensorT> log_test_values = { model_error_test };
    int metric_iter = 0;
    for (const std::string& metric_name : this->getMetricNamesLinearized()) {
      log_train_headers.push_back(metric_name);
      log_test_headers.push_back(metric_name);
      log_train_values.push_back(model_metrics_train(metric_iter));
      log_test_values.push_back(model_metrics_test(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  int n_encodings_;
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_);
    assert(n_metric_output_nodes == n_metric_output_nodes);
    assert(n_input_nodes == n_input_pixels + n_encodings_);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // Gaussian Sampler
        Eigen::Tensor<TensorT, 2> gaussian_samples = GaussianSampler<TensorT>(1, n_encodings_);

        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = gaussian_samples(0, nodes_iter); // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
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
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_);
    assert(n_metric_output_nodes == n_metric_output_nodes);
    assert(n_input_nodes == n_input_pixels + n_encodings_);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
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
 @brief Pixel reconstruction MNIST example whereby all pixels are
  linearized and read into the model.  The model then attempts to
  reconstruction the pixels using a Variational Auto Encoder network

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
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
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // define the data simulator
  const std::size_t input_size = 784;
  const std::size_t encoding_size = 2;
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
  data_simulator.n_encodings_ = encoding_size;

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < input_size; ++i) {
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

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < input_size; ++i) {
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
    ModelResources model_resources = { ModelDevice(1, 1) };
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
  model_trainer.setLogging(true, true, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper1);
  loss_function_helper2.output_nodes_ = encoding_nodes_mu;
  loss_function_helper2.loss_functions_ = { std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper2.loss_function_grads_ = { std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper2);
  loss_function_helper3.output_nodes_ = encoding_nodes_logvar;
  loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper3);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
  metric_function_helper1.metric_names_ = { "MAE" };
  metric_function_helpers.push_back(metric_function_helper1);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the initial population
  Model<float> model;
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    ModelTrainerExt<float>().makeVAEFullyConn(model, input_size, encoding_size, 128, 128, 0, false, false, true);
    //ModelTrainerExt<float>().makeVAECovNet(model, input_size, encoding_size, 32, 1, 0, 2, 1, 0, 128, 128, 7, 1, false, true);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "VAE_model.binary";
    const std::string interpreter_filename = data_dir + "VAE_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("VAE1");
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]); // FIX ME!
    //std::cout << "Modifying the learning rate..." << std::endl;
    //for (auto& weight_map : model.weights_) {
    //  if (weight_map.second->getSolverOp()->getName() == "AdamOp") {
    //    weight_map.second->getSolverOpShared()->setLearningRate(5e-6);
    //  }
    //}
  }

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
    input_nodes, model_logger, model_interpreters.front());
}

/// cmd: MNIST_VAE_example C:/Users/dmccloskey/Documents/GitHub/mnist/ true true
int main(int argc, char** argv)
{
  // Parse the user commands
  std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
  //std::string data_dir = "/home/user/data/";
  //std::string data_dir = "C:/Users/domccl/GitHub/mnist/";
  bool make_model = true, train_model = true;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    make_model = (argv[2] == std::string("true")) ? true : false;
  }
  if (argc >= 4) {
    train_model = (argv[3] == std::string("true")) ? true : false;
  }

  // run the application
  main_MNIST(data_dir, make_model, train_model);

  return 0;
}