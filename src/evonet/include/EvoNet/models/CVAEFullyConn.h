/**TODO:  Add copyright*/

#ifndef EVONET_CVAEFULLYCONN_H
#define EVONET_CVAEFULLYCONN_H

// .h
#include <EvoNet/ml/ModelTrainer.h>
#include <EvoNet/ml/ModelBuilder.h>

// .cpp

namespace EvoNet
{
  template<typename TensorT, typename InterpreterT>
  class CVAEFullyConn : public ModelTrainer<TensorT, InterpreterT>
  {
  public:
    /*
    @brief Variational autoencoder that encodes the labels using a concrete distribution
      and style using a gaussian distribution

    References:
      arXiv:1804.00104
      https://github.com/Schlumberger/joint-vae

    @param[in, out] model The network model
    @param[in] n_pixels The number of input/output pixels
    @param[in] n_categorical The length of the categorical layer
    @param[in] n_encodings The length of the encodings layer
    @param[in] n_hidden The length of the hidden layers
    @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
    */
    void makeCVAE(Model<TensorT>& model,
      const int& n_inputs = 784, const int& n_encodings = 64, const int& n_categorical = 10, const int& n_hidden_0 = 512, const int& n_hidden_1 = 256, const int& n_hidden_2 = 64,
      const bool& add_bias = true, const bool& specify_layers = false); 
    
    /*
    @brief Decoder that generates pixels from a concrete distribution and a gaussian distribution

    References:
      arXiv:1804.00104
      https://github.com/Schlumberger/joint-vae

    @param[in, out] model The network model
    @param[in] n_input The number of output nodes
    @param[in] n_categorical The length of the categorical layer
    @param[in] n_encodings The length of the encodings layer
    @param[in] n_hidden The length of the hidden layers
    @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation

    */
    void makeCVAEDecoder(Model<TensorT>& model,
      const int& n_inputs = 784, const int& n_encodings = 64, const int& n_categorical = 10, const int& n_hidden_0 = 512, const int& n_hidden_1 = 256, const int& n_hidden_2 = 64,
      const bool& add_bias = true, const bool& specify_layers = false);

    /*
    @brief Encoder that encodes pixels to a concrete distribution and a gaussian distribution

    References:
      arXiv:1804.00104
      https://github.com/Schlumberger/joint-vae

    @param[in, out] model The network model
    @param[in] n_input The number of input nodes
    @param[in] n_categorical The length of the categorical layer
    @param[in] n_encodings The length of the encodings layer
    @param[in] n_hidden The length of the hidden layers
    @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation

    */
    void makeCVAEEncoder(Model<TensorT>& model,
      const int& n_inputs = 784, const int& n_encodings = 64, const int& n_categorical = 10, const int& n_hidden_0 = 512, const int& n_hidden_1 = 256, const int& n_hidden_2 = 64,
      const bool& add_bias = true, const bool& specify_layers = false); 

    /// Members
    bool KL_divergence_warmup_ = true;
    TensorT beta_ = 30;
    TensorT capacity_c_ = 5;
    TensorT capacity_d_ = 5;
    TensorT learning_rate_ = 1e-5;
    TensorT gradient_clipping_ = 10;
  };

  template <typename TensorT, typename InterpreterT>
  inline void CVAEFullyConn<TensorT, InterpreterT>::makeCVAE(Model<TensorT>& model,
    const int& n_inputs, const int& n_encodings, const int& n_categorical, const int& n_hidden_0, const int& n_hidden_1, const int& n_hidden_2, const bool& add_bias, const bool& specify_layers) {
    model.setId(0);
    model.setName("VAE");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(this->learning_rate_, 0.9, 0.999, 1e-8, this->gradient_clipping_));

    // Add the Endocer FC layers
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_logalpha;
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
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
    node_names_logalpha = model_builder.addFullyConnected(model, "LogAlphaEnc", "LogAlphaEnc", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Add the Encoding layers
    std::vector<std::string> node_names_Gencoder = model_builder.addGaussianEncoding(model, "Gaussian_encoding", "Gaussian_encoding", node_names_mu, node_names_logvar, true);
    std::vector<std::string> node_names_Cencoder = model_builder.addCategoricalEncoding(model, "Categorical_encoding", "Categorical_encoding", node_names_logalpha, true);

    // Add the Decoder FC layers
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names_Gencoder, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE2", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    if (n_hidden_1 > 0 && n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    else if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names_Gencoder, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE1", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    if (n_hidden_0 > 0 && n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    else if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names_Gencoder, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE1", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    node_names = model_builder.addFullyConnected(model, "DE-Output", "DE-Output", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
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
    node_names_logalpha = model_builder.addSinglyConnected(model, "LogAlpha", "LogAlpha", node_names_logalpha, node_names_logalpha.size(),
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
    for (const std::string& node_name : node_names_logalpha)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_Cencoder)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  template <typename TensorT, typename InterpreterT>
  inline void CVAEFullyConn<TensorT, InterpreterT>::makeCVAEDecoder(Model<TensorT>& model,
    const int& n_inputs, const int& n_encodings, const int& n_categorical, const int& n_hidden_0, const int& n_hidden_1, const int& n_hidden_2, const bool& add_bias, const bool& specify_layers) {
    model.setId(0);
    model.setName("CVAEDecoder");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_Gencoder = model_builder.addInputNodes(model, "Gaussian_encoding", "Gaussian_encoding", n_encodings, specify_layers); // just Mu
    std::vector<std::string> node_names_logalpha = model_builder.addInputNodes(model, "LogAlphaEnc", "LogAlphaEnc", n_encodings, specify_layers);

    // Make the softmax layer
    std::vector<std::string> node_names_Cencoder = model_builder.addStableSoftMax(model, "Categorical_encoding-SoftMax-Out", "Categorical_encoding-SoftMax-Out", node_names_logalpha, specify_layers);

    // Define the activation based on `add_feature_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(this->learning_rate_, 0.9, 0.999, 1e-8, this->gradient_clipping_));

    // Add the Decoder FC layers
    std::vector<std::string> node_names;
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE2", "DE2", node_names_Gencoder, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE2", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    if (n_hidden_1 > 0 && n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    else if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names_Gencoder, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE1", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    if (n_hidden_0 > 0 && n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    else if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names_Gencoder, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      model_builder.addFullyConnected(model, "DE1", node_names_Cencoder, node_names,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, specify_layers);
    }
    node_names = model_builder.addFullyConnected(model, "DE-Output", "DE-Output", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, add_bias, true);

    // Add the actual output nodes
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }

  template <typename TensorT, typename InterpreterT>
  inline void CVAEFullyConn<TensorT, InterpreterT>::makeCVAEEncoder(Model<TensorT>& model,
    const int& n_inputs, const int& n_encodings, const int& n_categorical, const int& n_hidden_0, const int& n_hidden_1, const int& n_hidden_2, const bool& add_bias, const bool& specify_layers) {
    model.setId(0);
    model.setName("CVAEEncoder");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_feature_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(this->learning_rate_, 0.9, 0.999, 1e-8, this->gradient_clipping_));

    // Add the Endocer FC layers
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_logalpha;
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
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
    node_names_logalpha = model_builder.addFullyConnected(model, "LogAlphaEnc", "LogAlphaEnc", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

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
    node_names_logalpha = model_builder.addSinglyConnected(model, "LogAlpha", "LogAlpha", node_names_logalpha, node_names_logalpha.size(),
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
    for (const std::string& node_name : node_names_logalpha)
      model.nodes_.at(node_name)->setType(NodeType::output);
  }
}
#endif //EVONET_CVAEFULLYCONN_H