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
  void makeCVAE(Model<TensorT>& model, int n_pixels = 784, int n_categorical = 10, int n_encodings = 8, int n_hidden_0 = 512, bool specify_layer = true) {
    model.setId(0);
    model.setName("CVAE");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_pixels, specify_layer);

    // Define the activation based on `add_feature_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-5, 0.9, 0.999, 1e-8, 10));

    // Add the Endocer FC layers
    std::vector<std::string> node_names, node_names_mu, node_names_logvar, node_names_logalpha;
    node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + node_names.size()) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + node_names.size()) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names_mu = model_builder.addFullyConnected(model, "EN-Mu", "EN-Mu", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names_logvar = model_builder.addFullyConnected(model, "EN-LogVar", "EN-LogVar", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names_logalpha = model_builder.addFullyConnected(model, "EN-LogAlpha", "EN-LogAlpha", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);

    // Add the Encoding layers
    std::vector<std::string> node_names_Gencoder = model_builder.addGaussianEncoding(model, "Gaussian_encoding", "Gaussian_encoding", node_names_mu, node_names_logvar, true);
    std::vector<std::string> node_names_Cencoder = model_builder.addCategoricalEncoding(model, "Categorical_encoding", "Categorical_encoding", node_names_logalpha, true);

    // Add the Decoder FC layers
    node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names_Gencoder, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    model_builder.addFullyConnected(model, "DE0", node_names_Cencoder, node_names,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, specify_layer);
    node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names = model_builder.addFullyConnected(model, "DE-Output", "DE-Output", node_names, n_pixels,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, false, true);

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
    //node_names_Cencoder = model_builder.addSinglyConnected(model, "Categorical_encoding-Out", "Categorical_encoding-Out", node_names_Cencoder, node_names_Cencoder.size(),
    //  std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
    //  std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
    //  integration_op, integration_error_op, integration_weight_grad_op,
    //  std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
    //  std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_pixels,
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
      model.getNodesMap().at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
  /*
  @brief Decoder that generates pixels from a concrete distribution and a gaussian distribution

  References:
    arXiv:1804.00104
    https://github.com/Schlumberger/joint-vae

  @param[in, out] model The network model
  @param[in] n_pixels The number of output pixels
  @param[in] n_categorical The length of the categorical layer
  @param[in] n_encodings The length of the encodings layer
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation

  */
  void makeCVAEDecoder(Model<TensorT>& model, int n_pixels = 784, int n_categorical = 10, int n_encodings = 8, int n_hidden_0 = 512, bool specify_layer = true) {
    model.setId(0);
    model.setName("CVAEDecoder");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_Gencoder = model_builder.addInputNodes(model, "Gaussian_encoding", "Gaussian_encoding", n_encodings, specify_layer);
    std::vector<std::string> node_names_Cencoder = model_builder.addInputNodes(model, "Categorical_encoding-SoftMax-Out", "Categorical_encoding-SoftMax-Out", n_categorical, specify_layer);

    // Define the activation based on `add_feature_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-5, 0.9, 0.999, 1e-8, 10));

    // Add the Decoder FC layers
    std::vector<std::string> node_names = model_builder.addFullyConnected(model, "DE0", "DE0", node_names_Gencoder, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Gencoder.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    model_builder.addFullyConnected(model, "DE0", node_names_Cencoder, node_names,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_Cencoder.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, specify_layer);
    node_names = model_builder.addFullyConnected(model, "DE1", "DE1", node_names, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_hidden_0) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names = model_builder.addFullyConnected(model, "DE-Output", "DE-Output", node_names, n_pixels,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 1)),
      solver_op, 0.0f, 0.0f, false, true);

    // Add the actual output nodes
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_pixels,
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
  /*
  @brief Encoder that encodes pixels to a concrete distribution and a gaussian distribution

  References:
    arXiv:1804.00104
    https://github.com/Schlumberger/joint-vae

  @param[in, out] model The network model
  @param[in] n_pixels The number of input pixels
  @param[in] n_categorical The length of the categorical layer
  @param[in] n_encodings The length of the encodings layer
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation

  */
  void makeCVAEEncoder(Model<TensorT>& model, int n_pixels = 784, int n_categorical = 10, int n_encodings = 8, int n_hidden_0 = 512, bool specify_layer = true) {
    model.setId(0);
    model.setName("CVAEEncoder");

    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names_input = model_builder.addInputNodes(model, "Input", "Input", n_pixels, specify_layer);

    // Define the activation based on `add_feature_norm`
    std::shared_ptr<ActivationOp<TensorT>> activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    std::shared_ptr<ActivationOp<TensorT>> activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-5, 0.9, 0.999, 1e-8, 10));

    // Add the Endocer FC layers
    std::vector<std::string> node_names, node_names_mu, node_names_logvar, node_names_logalpha;
    node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + node_names.size()) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names, n_hidden_0,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + node_names.size()) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names_mu = model_builder.addFullyConnected(model, "EN-Mu", "EN-Mu", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names_logvar = model_builder.addFullyConnected(model, "EN-LogVar", "EN-LogVar", node_names, n_encodings,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_encodings) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);
    node_names_logalpha = model_builder.addFullyConnected(model, "EN-LogAlpha", "EN-LogAlpha", node_names, n_categorical,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names.size() + n_categorical) / 2, 1)),
      solver_op, 0.0f, 0.0f, false, specify_layer);

    // Add the Encoding layers
    std::vector<std::string> node_names_Gencoder = model_builder.addGaussianEncoding(model, "Gaussian_encoding", "Gaussian_encoding", node_names_mu, node_names_logvar, true);
    std::vector<std::string> node_names_Cencoder = model_builder.addCategoricalEncoding(model, "Categorical_encoding", "Categorical_encoding", node_names_logalpha, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names_Gencoder)
      model.nodes_.at(node_name)->setType(NodeType::output);
    for (const std::string& node_name : node_names_Cencoder)
      model.nodes_.at(node_name)->setType(NodeType::output);
  }
  void adaptiveTrainerScheduler(
    const int& n_generations,
    const int& n_epochs,
    Model<TensorT>& model,
    ModelInterpreterGpu<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {

    // Increase the KL divergence beta after [...] number of iterations
    TensorT scale_factor1 = (n_epochs - 100 > 0) ? n_epochs - 100 : 1;
    TensorT beta = 30 / 2.5e4 * scale_factor1;
    if (beta > 30) beta = 30;
    TensorT scale_factor2 = (n_epochs - 1.0e4 > 0) ? n_epochs - 1.0e4 : 1;
    TensorT capacity_c = 5 / 1.5e4 * scale_factor2;
    if (capacity_c > 5) capacity_c = 5;
    TensorT capacity_d = 5 / 1.5e4 * scale_factor2;
    if (capacity_d > 5) capacity_d = 5;
    this->getLossFunctionHelpers().at(1).loss_functions_.at(0) = std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, beta, capacity_c));
    this->getLossFunctionHelpers().at(2).loss_functions_.at(0) = std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, beta, capacity_c));
    this->getLossFunctionHelpers().at(3).loss_functions_.at(0) = std::make_shared<KLDivergenceCatLossOp<float>>(KLDivergenceCatLossOp<float>(1e-6, beta, capacity_d));
    this->getLossFunctionHelpers().at(1).loss_function_grads_.at(0) = std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, beta, capacity_c));
    this->getLossFunctionHelpers().at(2).loss_function_grads_.at(0) = std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, beta, capacity_c));
    this->getLossFunctionHelpers().at(3).loss_function_grads_.at(0) = std::make_shared<KLDivergenceCatLossGradOp<float>>(KLDivergenceCatLossGradOp<float>(1e-6, beta, capacity_d));

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
  void trainingModelLogger(const int & n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT & model_error_train, const TensorT & model_error_test,
    const Eigen::Tensor<TensorT, 1> & model_metrics_train, const Eigen::Tensor<TensorT, 1> & model_metrics_test) override
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
  int n_categorical_;
  int encodings_traversal_iter_ = 0;
  int categorical_traversal_iter_ = 0;
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_ + n_categorical_); // mu, logvar, logalpha
    assert(n_metric_output_nodes == n_input_pixels);
    assert(n_input_nodes == n_input_pixels + n_encodings_ + 2 * n_categorical_); // Guassian sampler, Gumbel sampler, inverse tau

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getTrainingIndices(batch_size, 1);

    // Reformat the input data for training
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // Gaussian Sampler
        Eigen::Tensor<TensorT, 2> gaussian_samples = GaussianSampler<TensorT>(1, n_encodings_);

        // Concrete Sampler
        Eigen::Tensor<TensorT, 2> categorical_samples = GumbelSampler<TensorT>(1, n_categorical_);
        TensorT inverse_tau = 3.0 / 2.0; //1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

        // Assign the input/output values
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices(batch_iter), nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices(batch_iter), nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices(batch_iter), nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = gaussian_samples(0, nodes_iter); // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
          if (nodes_iter < n_categorical_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = categorical_samples(0, nodes_iter); // sample from gumbel distribution
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_ + n_categorical_) = inverse_tau; // inverse tau
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_) = 0; // Dummy data for the KL divergence cat
          }
        }
      }
    }
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);
    const int n_input_pixels = this->validation_data.dimension(1);

    assert(n_output_nodes == n_input_pixels + 2 * n_encodings_ + n_categorical_); // mu, logvar, logalpha
    assert(n_metric_output_nodes == n_input_pixels);
    assert(n_input_nodes == n_input_pixels + n_encodings_ + 2 * n_categorical_); // Guassian sampler, Gumbel sampler, inverse tau

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices = this->getValidationIndices(batch_size, 1);

    // Reformat the input data for validation
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {

        // Gaussian Sampler
        Eigen::Tensor<TensorT, 2> gaussian_samples = GaussianSampler<TensorT>(1, n_encodings_);

        // Concrete Sampler
        Eigen::Tensor<TensorT, 2> categorical_samples = GumbelSampler<TensorT>(1, n_categorical_);
        TensorT inverse_tau = 3.0 / 2.0; //1.0 / 0.5; // Madison 2017 recommended 2/3 for tau

        // Assign the input/output values
        for (int nodes_iter = 0; nodes_iter < n_input_pixels; ++nodes_iter) {
          input_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices(batch_iter), nodes_iter);
          loss_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices(batch_iter), nodes_iter);
          metric_output_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices(batch_iter), nodes_iter);
          if (nodes_iter < n_encodings_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = gaussian_samples(0, nodes_iter); // sample from a normal distribution
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels) = 0; // Dummy data for KL divergence mu
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = 0; // Dummy data for KL divergence logvar
          }
          if (nodes_iter < n_categorical_) {
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_) = categorical_samples(0, nodes_iter); // sample from gumbel distribution
            input_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + n_encodings_ + n_categorical_) = inverse_tau; // inverse tau
            loss_output_data(batch_iter, memory_iter, nodes_iter + n_input_pixels + 2 * n_encodings_) = 0; // Dummy data for KL divergence cat
          }
        }
      }
    }
  }
  void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);

    assert(n_input_nodes == n_encodings_ + n_categorical_); // Guassian encoding, Gumbel categorical

    // Initialize the gaussian encodings to random and all categorical encodings to 0
    input_data = input_data.constant(TensorT(0)); // initialize the input to 0;
    Eigen::array<Eigen::Index, 3> offsets = { 0, 0, 0 };
    Eigen::array<Eigen::Index, 3> extents = { batch_size, memory_size, n_encodings_ };
    input_data.slice(offsets, extents) = input_data.slice(offsets, extents).random();

    // Assign the encoding values by sampling the 95% confidence limits of the inverse normal distribution
    const TensorT step_size = (0.95 - 0.05) / batch_size;
    input_data.chip(encodings_traversal_iter_, 2) = (input_data.chip(encodings_traversal_iter_, 2).constant(step_size).cumsum(0) +
      input_data.chip(encodings_traversal_iter_, 2).constant(TensorT(0.05))).ndtri();

    // Assign the categorical values
    input_data.chip(n_encodings_ + categorical_traversal_iter_, 2) = input_data.chip(n_encodings_ + categorical_traversal_iter_, 2).constant(TensorT(1));

    // Increase the traversal iterators
    encodings_traversal_iter_ += 1;
    if (encodings_traversal_iter_ >= n_encodings_) {
      encodings_traversal_iter_ = 0;
      categorical_traversal_iter_ += 1;
    }
    if (categorical_traversal_iter_ >= n_categorical_) {
      categorical_traversal_iter_ = 0;
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
  reconstruction the pixels using an Auto Encoder network where
  the labels of the images are disentangled from the style of the images
  using a concrete distribution and gaussian distribution, respectively

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
 */
void trainModel(const std::string& data_dir, const bool& make_model) {
  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false);

  // define the data simulator
  const std::size_t n_pixels = 784;
  const std::size_t encoding_size = 8;
  const std::size_t categorical_size = 10;
  const std::size_t n_hidden = 512;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  DataSimulatorExt<float> data_simulator;

  // read in the training data
  std::string training_data_filename = data_dir + "train-images.idx3-ubyte";
  std::string training_labels_filename = data_dir + "train-labels.idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, n_pixels);

  // read in the validation data
  std::string validation_data_filename = data_dir + "t10k-images.idx3-ubyte";
  std::string validation_labels_filename = data_dir + "t10k-labels.idx1-ubyte";
  data_simulator.readData(validation_data_filename, validation_labels_filename, false, validation_data_size, n_pixels);
  data_simulator.unitScaleData();
  data_simulator.n_encodings_ = encoding_size;
  data_simulator.n_categorical_ = categorical_size;

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_pixels; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the encoding nodes and add them to the input
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Gaussian_encoding_%012d-Sampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding_%012d-GumbelSampler", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding_%012d-InverseTau", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_pixels; ++i) {
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

  // Make the var nodes
  std::vector<std::string> encoding_nodes_logvar;
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "LogVar_%012d", i);
    std::string name(name_char);
    encoding_nodes_logvar.push_back(name);
  }

  // Make the alpha nodes
  std::vector<std::string> encoding_nodes_logalpha;
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "LogAlpha_%012d", i);
    std::string name(name_char);
    encoding_nodes_logalpha.push_back(name);
  }

  // define the model trainers and resources for the trainers
  std::vector<ModelInterpreterGpu<float>> model_interpreters;
  ModelResources model_resources = { ModelDevice(1, 1) };
  ModelInterpreterGpu<float> model_interpreter(model_resources);
  model_interpreters.push_back(model_interpreter);

  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(128);
  model_trainer.setNEpochsTraining(200001);
  model_trainer.setNEpochsValidation(25);
  model_trainer.setNEpochsEvaluation(0);
  model_trainer.setMemorySize(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3, loss_function_helper4;
  loss_function_helper1.output_nodes_ = output_nodes;
  //loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
  //loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0)) };
  loss_function_helper1.loss_functions_ = { std::make_shared<MAPELossOp<float>>(MAPELossOp<float>(1e-6, 1e-5)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MAPELossGradOp<float>>(MAPELossGradOp<float>(1e-6, 1e-5)) };
  loss_function_helpers.push_back(loss_function_helper1);
  loss_function_helper2.output_nodes_ = encoding_nodes_mu;
  loss_function_helper2.loss_functions_ = { std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper2.loss_function_grads_ = { std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper2);
  loss_function_helper3.output_nodes_ = encoding_nodes_logvar;
  loss_function_helper3.loss_functions_ = { std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper3.loss_function_grads_ = { std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper3);
  loss_function_helper4.output_nodes_ = encoding_nodes_logalpha;
  loss_function_helper4.loss_functions_ = { std::make_shared<KLDivergenceCatLossOp<float>>(KLDivergenceCatLossOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helper4.loss_function_grads_ = { std::make_shared<KLDivergenceCatLossGradOp<float>>(KLDivergenceCatLossGradOp<float>(1e-6, 0.0, 0.0)) };
  loss_function_helpers.push_back(loss_function_helper4);
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
    model_trainer.makeCVAE(model, n_pixels, categorical_size, encoding_size, n_hidden);
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "CVAE_model.binary";
    const std::string interpreter_filename = data_dir + "CVAE_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("CVAE1");
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreters[0]);
  }

  // Train the model
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, data_simulator,
    input_nodes, model_logger, model_interpreters.front());
}

void traverseLatentSpace(const std::string& data_dir, const bool& make_model) {

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, true, false, true);

  // define the data simulator
  const std::size_t n_pixels = 784;
  const std::size_t encoding_size = 8;
  const std::size_t categorical_size = 10;
  const std::size_t n_hidden = 512;
  DataSimulatorExt<float> data_simulator;
  data_simulator.n_encodings_ = encoding_size;
  data_simulator.n_categorical_ = categorical_size;

  // Make the input nodes
  std::vector<std::string> input_nodes;

  // Make the encoding nodes and add them to the input
  for (int i = 0; i < encoding_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Gaussian_encoding_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }
  for (int i = 0; i < categorical_size; ++i) {
    char name_char[512];
    sprintf(name_char, "Categorical_encoding-SoftMax-Out_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_pixels; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // define the model trainers and resources for the trainers
  ModelResources model_resources = { ModelDevice(0, 1) };
  ModelInterpreterGpu<float> model_interpreter(model_resources);

  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(8); // determines the number of samples across the latent dimension
  model_trainer.setNEpochsEvaluation(encoding_size * categorical_size); // determined by the number of latent dimensions to traverse
  model_trainer.setMemorySize(1);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(false, false, true);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<MSELossOp<float>>(MSELossOp<float>(1e-6, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>(1e-6, 1.0 )) };
  loss_function_helpers.push_back(loss_function_helper1);
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
  metric_function_helper1.metric_names_ = { "MAE" };
  metric_function_helpers.push_back(metric_function_helper1);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // build the decoder and update the weights from the trained model
  Model<float> model;
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    model_trainer.makeCVAEDecoder(model, n_pixels, categorical_size, encoding_size, n_hidden);
    std::cout << "Reading in the trained model weights..." << std::endl;
    const std::string model_filename = data_dir + "CVAE_model.binary";
    ModelFile<float> model_file;
    model_file.loadWeightValuesBinary(model_filename, model.weights_);

    // check that all weights were read in correctly
    for (auto& weight_map : model.getWeightsMap()) {
      if (weight_map.second->getInitWeight()) {
        std::cout << "Model " << model.getName() << " Weight " << weight_map.first << " has not be initialized." << std::endl;;
      }
    }
  }
  else {
    // read in the trained model
    std::cout << "Reading in the model..." << std::endl;
    const std::string model_filename = data_dir + "CVAEDecoder_model.binary";
    const std::string interpreter_filename = data_dir + "CVAEDecoder_interpreter.binary";
    ModelFile<float> model_file;
    model_file.loadModelBinary(model_filename, model);
    model.setId(1);
    model.setName("CVAEDecoder1");
    ModelInterpreterFileGpu<float> model_interpreter_file;
    model_interpreter_file.loadModelInterpreterBinary(interpreter_filename, model_interpreter);
  }

  // traverse the latent space (evaluation)
  Eigen::Tensor<float, 4> values = model_trainer.evaluateModel(model, data_simulator, input_nodes, model_logger, model_interpreter);
}

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
  if (train_model) trainModel(data_dir, make_model);
  else traverseLatentSpace(data_dir, make_model);

  return 0;
}