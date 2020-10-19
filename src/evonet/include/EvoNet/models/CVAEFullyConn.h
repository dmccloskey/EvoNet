/**TODO:  Add copyright*/

#ifndef EVONET_CVAEFULLYCONN_H
#define EVONET_CVAEFULLYCONN_H

// .h
#include <EvoNet/ml/ModelTrainer.h>
#include <EvoNet/ml/ModelBuilder.h>
#include <EvoNet/io/Parameters.h>

// .cpp

namespace EvoNet
{
  /// Helper methods
  static void makeInputNodes(std::vector<std::string>& input_nodes, const int& n_features) {
    for (int i = 0; i < n_features; ++i) {
      char name_char[512];
      sprintf(name_char, "Input_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeGaussianEncodingSamplerNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Gaussian_encoding_%012d-Sampler", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeCategoricalEncodingSamplerNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Categorical_encoding_%012d-GumbelSampler", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeCategoricalEncodingTauNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Categorical_encoding_%012d-InverseTau", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeAlphaEncodingNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      //sprintf(name_char, "Alpha_%012d", i);
      sprintf(name_char, "Categorical_encoding-SoftMax-Out_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  template<class ...ParameterTypes>
  static void makeMuEncodingNodes(std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Gaussian_encoding_%012d", i);
      std::string name(name_char);
      input_nodes.push_back(name);
    }
  }
  static std::vector<std::string> makeOutputNodes(const int& n_features) {
    std::vector<std::string> output_nodes;
    for (int i = 0; i < n_features; ++i) {
      char name_char[512];
      sprintf(name_char, "Output_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeMuEncodingNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Mu_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeLogVarEncodingNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "LogVar_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeAlphaEncodingNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Alpha_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<class ...ParameterTypes>
  static std::vector<std::string> makeCategoricalSoftmaxNodes(const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    std::vector<std::string> output_nodes;
    for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(); ++i) {
      char name_char[512];
      sprintf(name_char, "Categorical_encoding-SoftMax-Out_%012d", i);
      std::string name(name_char);
      output_nodes.push_back(name);
    }
    return output_nodes;
  }
  template<typename TensorT, typename TrainerT, typename InterpreterT, typename InterpreterFileT, class ...ParameterTypes>
  static void makeModelAndInterpreters(Model<TensorT>& model, TrainerT& model_trainer, std::vector<InterpreterT>& model_interpreters, InterpreterFileT& model_interpreter_file, const int& n_features, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);

    // define the model interpreters
    setModelInterpreterParameters(model_interpreters, args...);

    // define the model
    if (std::get<EvoNetParameters::Main::MakeModel>(parameters).get()) {
      std::cout << "Making the model..." << std::endl;
      if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "EncDec") {
        model_trainer.makeCVAE(model, n_features,
          std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);
      }
      else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Enc") {
        // make the encoder only
        model_trainer.makeCVAEEncoder(model, n_features,
          std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);

        // read in the weights
        ModelFile<TensorT> model_file;
        model_file.loadWeightValuesBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model.weights_);

        // check that all weights were read in correctly
        for (auto& weight_map : model.getWeightsMap()) {
          if (weight_map.second->getInitWeight()) {
            std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
          }
        }
      }
      else if (std::get<EvoNetParameters::Examples::ModelType>(parameters).get() == "Dec") {
        // make the decoder only
        model_trainer.makeCVAEDecoder(model, n_features,
          std::get<EvoNetParameters::ModelTrainer::NEncodingsContinuous>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NEncodingsCategorical>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(),
          std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), false, true);

        // read in the weights
        ModelFile<TensorT> model_file;
        model_file.loadWeightValuesBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model.weights_);

        // check that all weights were read in correctly
        for (auto& weight_map : model.getWeightsMap()) {
          if (weight_map.second->getInitWeight()) {
            std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
          }
        }
      }
    }
    else {
      ModelFile<TensorT> model_file;
      loadModelFromParameters(model, model_interpreters.at(0), model_file, model_interpreter_file, args...);
    }
    model.setName(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory
  }

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

    /*
    @brief Variational autoencoder that encodes the labels using a concrete distribution
      and style using a gaussian distribution that allows for addition or subtraction
      of two different encodings

    References:
      arXiv:1804.00104
      https://github.com/Schlumberger/joint-vae

    @param[in, out] model The network model
    @param[in] n_pixels The number of input/output pixels
    @param[in] n_categorical The length of the categorical layer
    @param[in] n_encodings The length of the encodings layer
    @param[in] arithmetic_type "+" for addition or "-" for subtraction
    @param[in] n_hidden The length of the hidden layers
    @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
    */
    void makeCVAELatentArithmetic(Model<TensorT>& model,
      const int& n_inputs = 784, const int& n_encodings = 64, const int& n_categorical = 10, const char& arithmetic_type = '+', const int& n_hidden_0 = 512, const int& n_hidden_1 = 256, const int& n_hidden_2 = 64,
      const bool& add_bias = true, const bool& specify_layers = false);

    /// Members
    bool KL_divergence_warmup_ = true;
    bool supervision_warmup_ = true;
    int supervision_percent_ = 100;
    TensorT classification_loss_weight_ = 1.0;
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
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_alpha;
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
    node_names_alpha = model_builder.addFullyConnected(model, "AlphaEncNonProp", "AlphaEncNonProp", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names_alpha = model_builder.addStableSoftMax(model, "AlphaEnc", "AlphaEnc", node_names_alpha, specify_layers);

    // Add the Encoding layers
    std::vector<std::string> node_names_Gencoder = model_builder.addGaussianEncoding(model, "Gaussian_encoding", "Gaussian_encoding", node_names_mu, node_names_logvar, true);
    std::vector<std::string> node_names_Cencoder = model_builder.addCategoricalEncoding(model, "Categorical_encoding", "Categorical_encoding", node_names_alpha, true);

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
    node_names_alpha = model_builder.addSinglyConnected(model, "Alpha", "Alpha", node_names_alpha, node_names_alpha.size(),
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
    for (const std::string& node_name : node_names_alpha)
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
    std::vector<std::string> node_names_Cencoder = model_builder.addInputNodes(model, "Categorical_encoding-SoftMax-Out", "Categorical_encoding-SoftMax-Out", n_categorical, specify_layers);

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

    // Add the Encoder FC layers
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_alpha;
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
    node_names_alpha = model_builder.addFullyConnected(model, "AlphaEncNonProb", "AlphaEncNonProb", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Make the softmax layer
    std::vector<std::string> node_names_Cencoder = model_builder.addStableSoftMax(model, "Categorical_encoding-SoftMax", "Categorical_encoding-SoftMax", node_names_alpha, specify_layers);

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
    node_names_alpha = model_builder.addSinglyConnected(model, "Alpha", "Alpha", node_names_alpha, node_names_alpha.size(),
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
    for (const std::string& node_name : node_names_alpha)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_Cencoder)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  template<typename TensorT, typename InterpreterT>
  inline void CVAEFullyConn<TensorT, InterpreterT>::makeCVAELatentArithmetic(Model<TensorT>& model, const int& n_inputs, const int& n_encodings, const int& n_categorical, const char& arithmetic_type, const int& n_hidden_0, const int& n_hidden_1, const int& n_hidden_2, const bool& add_bias, const bool& specify_layers)
  {
    model.setId(0);
    model.setName("VAELatentArithmetic");

    ModelBuilder<TensorT> model_builder;

    // Define the activation based on `add_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(this->learning_rate_, 0.9, 0.999, 1e-8, this->gradient_clipping_));

    // Add the inputs (Left hand side)
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input@L", "Input@L", n_inputs, specify_layers);

    // Add the Endocer FC layers
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_alpha;
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0@L", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
      
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1@L", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2@L", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    node_names_mu = model_builder.addFullyConnected(model, "MuEnc", "MuEnc@L", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names_logvar = model_builder.addFullyConnected(model, "LogVarEnc", "LogVarEnc@L", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names_alpha = model_builder.addFullyConnected(model, "AlphaEnc", "AlphaEnc@L", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Make the Gaussian layer
    std::vector<std::string> node_names_Gencoder_L = model_builder.addSinglyConnected(model, "Gaussian_encoding_LR", "Gaussian_encoding_L", node_names_mu, node_names_mu.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, specify_layers);

    // Make the softmax layer
    std::vector<std::string> node_names_Cencoder_L = model_builder.addStableSoftMax(model, "Categorical_encoding-SoftMax_LR", "Categorical_encoding-SoftMax_L", node_names_alpha, specify_layers);

    // Add the inputs (Right hand side)
    node_names = model_builder.addInputNodes(model, "Input@R", "Input@R", n_inputs, specify_layers);

    // Add the Endocer FC layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN0", "EN0@R", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);

    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN1", "EN1@R", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "EN2", "EN2@R", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, add_bias, specify_layers);
    }
    node_names_mu = model_builder.addFullyConnected(model, "MuEnc", "MuEnc@R", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names_logvar = model_builder.addFullyConnected(model, "LogVarEnc", "LogVarEnc@R", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);
    node_names_alpha = model_builder.addFullyConnected(model, "AlphaEnc", "AlphaEnc@R", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layers);

    // Make the Gaussian layer
    std::vector<std::string> node_names_Gencoder_R = model_builder.addSinglyConnected(model, "Gaussian_encoding_LR", "Gaussian_encoding_R", node_names_mu, node_names_mu.size(),
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, specify_layers);

    // Make the softmax layer
    std::vector<std::string> node_names_Cencoder_R = model_builder.addStableSoftMax(model, "Categorical_encoding-SoftMax_LR", "Categorical_encoding-SoftMax_R", node_names_alpha, specify_layers);

    // Rename the input nodes
    std::vector<std::vector<std::string>> tokens_vec = { {"Input@L"}, {"Input@R"} };
    std::vector<std::string> replacement_vec = { "Input_L", "Input_R" };
    for (auto& node_map : model.nodes_) {
      for (int i = 0; i < replacement_vec.size(); ++i) {
        if (node_map.first == tokens_vec.at(i).front()) {
          std::string new_node_name = ReplaceTokens(node_map.first, tokens_vec.at(i), replacement_vec.at(i));
          node_map.first = new_node_name;
          node_map.second->setName(new_node_name);
        }
      }
    }
    for (auto& link_map : model.links_) {
      for (int i = 0; i < replacement_vec.size(); ++i) {
        if (link_map.second->getSourceNodeName() == tokens_vec.at(i).front()) {
          std::string new_source_name = ReplaceTokens(link_map.second->getSourceNodeName(), tokens_vec.at(i), replacement_vec.at(i));
          link_map.second->setSourceNodeName(new_source_name);
        }
      }
    }

    // Rename all of the weights
    std::vector<std::string> tokens = { "@L", "@R" };
    std::string replacement = "";
    for (auto& weight_map : model.weights_) {
      std::string new_weight_name = ReplaceTokens(weight_map.first,tokens, replacement);
      weight_map.first = new_weight_name;
      weight_map.second->setName(new_weight_name);
    }
    for (auto& link_map : model.links_) {
      std::string new_weight_name = ReplaceTokens(link_map.first, tokens, replacement);
      link_map.second->setWeightName(new_weight_name);
    }

    // Add or subtract the left hand side latent space from the right
    auto weight_average = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0.5));
    auto weight_sign = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1));
    if (arithmetic_type == '-') weight_sign = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1));
    std::vector<std::string> node_names_Gencoder = model_builder.addSinglyConnected(model, "Gaussian_encoding", "Gaussian_encoding", node_names_Gencoder_L, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, specify_layers);
    model_builder.addSinglyConnected(model, "Gaussian_encoding", node_names_Gencoder_R, node_names_Gencoder,
      weight_sign,
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, specify_layers);
    std::vector<std::string> node_names_Cencoder = model_builder.addSinglyConnected(model, "Categorical_encoding-SoftMax", "Categorical_encoding-SoftMax", node_names_Cencoder_L, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      weight_average,
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, specify_layers);
    model_builder.addSinglyConnected(model, "Categorical_encoding-SoftMax", node_names_Cencoder_R, node_names_Cencoder,
      weight_average,
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, specify_layers);

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
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_inputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_Gencoder)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_Cencoder)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
}
#endif //EVONET_CVAEFULLYCONN_H