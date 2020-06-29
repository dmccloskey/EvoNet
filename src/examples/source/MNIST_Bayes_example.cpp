/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerExperimentalDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerExperimentalDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicatorExperimental.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/Parameters.h>

#include <SmartPeak/simulator/MNISTSimulator.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Extended classes
template<typename TensorT>
class ModelTrainerExt : public ModelTrainerExperimentalDefaultDevice<TensorT>
{
public:
  /*
  @brief Fully Connected Bayesian model with Xavier-like initialization

  Reference:
  Blundell 2015 Weight uncertainty in neural networks arXiv:1505.05424

  @param[in, out] model The network model
  @param[in] n_inputs The number of input pixels
  @param[in] n_outputs The number of output labels
  @param[in] n_hidden The length of the hidden layers
  @param[in] specify_layers Whether to give the `ModelInterpreter` "hints" as to the correct network structure during graph to tensor compilation
  */
  void makeFullyConnBayes(Model<TensorT>& model, const int& n_inputs = 784, const int& n_outputs = 10, const int& n_hidden_0 = 512, const int& n_hidden_1 = 512, const int& n_hidden_2 = 512, const bool& add_gaussian = false, const TensorT& logvar_1 = -1, const TensorT& logvar_2 = -4, const TensorT& pi = 0.5, const bool& specify_layers = false, const TensorT& learning_rate = 1e-3, const TensorT& gradient_clipping = 100) {
    model.setId(0);
    model.setName("FullyConnectedBayesClassifier");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, specify_layers);

    // Define the activation based on `add_feature_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());
    auto activation_linear = std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>());
    auto activation_linear_grad = std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the weight inits
    std::shared_ptr<WeightInitOp<TensorT>> weight_init_mu, weight_init_logvar;
    if (n_hidden_0 > 0) {
      weight_init_mu = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(n_inputs + n_hidden_0) / 2, 1));
      weight_init_logvar = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(-12 / n_hidden_0)));
    }
    else {
      weight_init_mu = std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(n_inputs + n_outputs) / 2, 1));
      weight_init_logvar = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(-12 / n_outputs)));
    }

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(learning_rate, 0.9, 0.999, 1e-8, gradient_clipping));
    auto solver_dummy_op = std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>());

    // Define the nodes
    std::vector<std::string> node_names_mu, node_names_logvar, node_names_encoding, node_names_input, node_names_prior, node_names_posterior;

    // Add the 1st FC layer
    if (n_hidden_0 > 0) {
      node_names_input = node_names;
      if (add_gaussian) {
        // Add the bayesian nodes
        node_names = model_builder.addFullyConnectedBayesian(model, "EN0", "EN0", node_names_input, n_hidden_0,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          weight_init_mu, solver_op, weight_init_logvar, solver_op, logvar_1, logvar_2, pi,
          node_names_logvar, node_names_posterior, node_names_prior, specify_layers);
        // Add the actual output nodes
        node_names_posterior = model_builder.addSinglyConnected(model, "EN0Posterior", "EN0Posterior", node_names_posterior, node_names_posterior.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_prior = model_builder.addSinglyConnected(model, "EN0Prior", "EN0Prior", node_names_prior, node_names_prior.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_logvar = model_builder.addSinglyConnected(model, "EN0LogVar", "EN0LogVar", node_names_logvar, node_names_logvar.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        // Specify the output node types manually
        for (const std::string& node_name : node_names_posterior)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_prior)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_logvar)
          model.nodes_.at(node_name)->setType(NodeType::output);
      }
      else {
        node_names = model_builder.addFullyConnected(model, "EN0", "EN0", node_names_input, n_hidden_0,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_0) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
      }
    }

    // Add the 2nd FC layer
    if (n_hidden_1 > 0) {
      node_names_input = node_names;
      if (add_gaussian) {
        // Add the bayesian nodes
        node_names = model_builder.addFullyConnectedBayesian(model, "EN1", "EN1", node_names_input, n_hidden_1,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          weight_init_mu, solver_op, weight_init_logvar, solver_op, logvar_1, logvar_2, pi,
          node_names_logvar, node_names_posterior, node_names_prior, specify_layers);
        // Add the actual output nodes
        node_names_posterior = model_builder.addSinglyConnected(model, "EN1Posterior", "EN1Posterior", node_names_posterior, node_names_posterior.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_prior = model_builder.addSinglyConnected(model, "EN1Prior", "EN1Prior", node_names_prior, node_names_prior.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_logvar = model_builder.addSinglyConnected(model, "EN1LogVar", "EN1LogVar", node_names_logvar, node_names_logvar.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        // Specify the output node types manually
        for (const std::string& node_name : node_names_posterior)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_prior)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_logvar)
          model.nodes_.at(node_name)->setType(NodeType::output);
      }
      else {
        node_names = model_builder.addFullyConnected(model, "EN1", "EN1", node_names_input, n_hidden_1,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_hidden_1) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
      }
    }

    // Add the output FC layer
    if (n_outputs > 0) {
      node_names_input = node_names;
      if (add_gaussian) {
        // Add the bayesian nodes
        node_names = model_builder.addFullyConnectedBayesian(model, "EN2", "EN2", node_names_input, n_outputs,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          weight_init_mu, solver_op, weight_init_logvar, solver_op, logvar_1, logvar_2, pi,
          node_names_logvar, node_names_posterior, node_names_prior, specify_layers);
        // Add the actual output nodes
        node_names_posterior = model_builder.addSinglyConnected(model, "EN2Posterior", "EN2Posterior", node_names_posterior, node_names_posterior.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_prior = model_builder.addSinglyConnected(model, "EN2Prior", "EN2Prior", node_names_prior, node_names_prior.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        node_names_logvar = model_builder.addSinglyConnected(model, "EN2LogVar", "EN2LogVar", node_names_logvar, node_names_logvar.size(),
          activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
          solver_dummy_op, 0.0f, 0.0f, false, true);
        // Specify the output node types manually
        for (const std::string& node_name : node_names_posterior)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_prior)
          model.nodes_.at(node_name)->setType(NodeType::output);
        for (const std::string& node_name : node_names_logvar)
          model.nodes_.at(node_name)->setType(NodeType::output);
      }
      else {
        node_names = model_builder.addFullyConnected(model, "EN2", "EN2", node_names_input, n_outputs,
          activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
          std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((TensorT)(node_names_input.size() + n_outputs) / 2, 1)),
          solver_op, 0.0f, 0.0f, false, specify_layers);
      }
    }

    // Add the actual output nodes
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      activation_linear, activation_linear_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      solver_dummy_op, 0.0f, 0.0f, false, true);

    // Specify the output node types manually
    for (const std::string& node_name : node_names)
      model.nodes_.at(node_name)->setType(NodeType::output);
    model.setInputAndOutputNodes();
  }
};

template<typename TensorT>
class DataSimulatorExt : public MNISTSimulator<TensorT>
{
public:
  bool add_gaussian_ = false;
  int n_hidden_0_ = 0;
  int n_hidden_1_ = 0;
  void simulateData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps, const bool& is_train)
  {
    // infer data dimensions based on the input tensors
    const int batch_size = input_data.dimension(0);
    const int memory_size = input_data.dimension(1);
    const int n_input_nodes = input_data.dimension(2);
    const int n_output_nodes = loss_output_data.dimension(2);
    const int n_metric_output_nodes = metric_output_data.dimension(2);

    // make the start and end sample indices
    Eigen::Tensor<int, 1> sample_indices(this->training_data.dimension(1));
    if (is_train)
      sample_indices = this->getTrainingIndices(batch_size, 1);
    else
      sample_indices = this->getValidationIndices(batch_size, 1);

    // pull out the training data and labels
    Eigen::Tensor<TensorT, 3> training_data(batch_size, memory_size, this->training_data.dimension(1));
    Eigen::Tensor<TensorT, 3> training_labels(batch_size, memory_size, this->training_labels.dimension(1));
    for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
      for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
        for (int nodes_iter = 0; nodes_iter < this->training_data.dimension(1); ++nodes_iter) {
          if (is_train) training_data(batch_iter, memory_iter, nodes_iter) = this->training_data(sample_indices[batch_iter], nodes_iter);
          else training_data(batch_iter, memory_iter, nodes_iter) = this->validation_data(sample_indices[batch_iter], nodes_iter);
        }
        for (int nodes_iter = 0; nodes_iter < this->training_labels.dimension(1); ++nodes_iter) {
          if (is_train) training_labels(batch_iter, memory_iter, nodes_iter) = (TensorT)this->training_labels(sample_indices[batch_iter], nodes_iter);
          else training_labels(batch_iter, memory_iter, nodes_iter) = (TensorT)this->validation_labels(sample_indices[batch_iter], nodes_iter);
        }
      }
    }

    // Assign the input data
    input_data.setZero();
    input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) })) = training_data;

    // Assign the input data
    loss_output_data.setConstant(TensorT(1)); // negative log likelihood expected value
    loss_output_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_labels.dimension(1) })) = training_labels;

    // Assign the input data
    metric_output_data.setZero(); // in order to compute the total magnitude of the logvar
    metric_output_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_labels.dimension(1) })) = training_labels;

    assert(memory_size == 1);
    if (add_gaussian_) {
      if (n_hidden_0_ > 0 && n_hidden_1_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 2 * this->training_data.dimension(1) * n_hidden_0_ + 2 * n_hidden_0_ * n_hidden_1_ + 2 * n_hidden_1_ * this->training_labels.dimension(1));
        assert(n_metric_output_nodes == this->training_labels.dimension(1) + this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * n_hidden_1_ + n_hidden_1_ * this->training_labels.dimension(1));
        assert(n_input_nodes == this->training_data.dimension(1) + this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * n_hidden_1_ + n_hidden_1_ * this->training_labels.dimension(1));

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples(batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * n_hidden_1_ + n_hidden_1_ * this->training_labels.dimension(1));
        if (is_train) gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * n_hidden_1_ + n_hidden_1_ * this->training_labels.dimension(1))
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * n_hidden_1_ + n_hidden_1_ * this->training_labels.dimension(1) }));
        else gaussian_samples.setZero();

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + this->training_data.dimension(1) * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ * n_hidden_1_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ * n_hidden_1_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ * this->training_labels.dimension(1) })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * n_hidden_1_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_1_ * this->training_labels.dimension(1) }));
      }
      else if (n_hidden_0_ > 0) {
        assert(n_output_nodes == this->training_labels.dimension(1) + 2 * this->training_data.dimension(1) * n_hidden_0_ + 2 * n_hidden_0_ * this->training_labels.dimension(1));
        assert(n_metric_output_nodes == this->training_labels.dimension(1) + this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * this->training_labels.dimension(1));
        assert(n_input_nodes == this->training_data.dimension(1) + this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * this->training_labels.dimension(1));

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples(batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * this->training_labels.dimension(1));
        if (is_train) gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * this->training_labels.dimension(1))
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ + n_hidden_0_ * this->training_labels.dimension(1) }));
        else gaussian_samples.setZero();

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * n_hidden_0_ }));
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) + this->training_data.dimension(1) * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ * this->training_labels.dimension(1) })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) * n_hidden_0_ }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, n_hidden_0_ * this->training_labels.dimension(1) }));
      }
      else {
        assert(n_output_nodes == this->training_labels.dimension(1) + 2 * this->training_data.dimension(1) * this->training_labels.dimension(1));
        assert(n_metric_output_nodes == this->training_labels.dimension(1) + this->training_data.dimension(1) * this->training_labels.dimension(1));
        assert(n_input_nodes == this->training_data.dimension(1) + this->training_data.dimension(1) * this->training_labels.dimension(1));

        // Gaussian sampler input/output data
        Eigen::Tensor<TensorT, 3> gaussian_samples(batch_size, memory_size, this->training_data.dimension(1) * this->training_labels.dimension(1));
        if (is_train) gaussian_samples = GaussianSampler<TensorT>(batch_size * memory_size, this->training_data.dimension(1) * this->training_labels.dimension(1))
          .reshape(Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * this->training_labels.dimension(1) }));
        else gaussian_samples.setZero();

        // Assign the input data
        input_data.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, this->training_data.dimension(1) }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * this->training_labels.dimension(1) })) = gaussian_samples.slice(
          Eigen::array<Eigen::Index, 3>({ 0, 0, 0 }), Eigen::array<Eigen::Index, 3>({ batch_size, memory_size, this->training_data.dimension(1) * this->training_labels.dimension(1) }));
      }
    }
    else {
      assert(n_output_nodes == this->training_labels.dimension(1));
      assert(n_metric_output_nodes == this->training_labels.dimension(1));
      assert(n_input_nodes == this->training_data.dimension(1));
    }
  }
  void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    simulateData(input_data, loss_output_data, metric_output_data, time_steps, true);
  }
  void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) override {
    simulateData(input_data, loss_output_data, metric_output_data, time_steps, false);
  }
};

template<typename TensorT>
class ModelReplicatorExt : public ModelReplicatorExperimental<TensorT>
{};

template<typename TensorT>
class PopulationTrainerExt : public PopulationTrainerExperimentalDefaultDevice<TensorT>
{};

/**
 @brief Image classification MNIST example whereby all pixels are
  linearized and read into the model.  The model then attempts to
  classify the image using a Bayesian fully connected architecture

  Data processing:
  - whole image pixels (linearized) 28x28 normalized to 0 to 1
  - classifier (1 hot vector from 0 to 9)
 */
template<class ...ParameterTypes>
void main_MNIST(const ParameterTypes& ...args) {
  auto parameters = std::make_tuple(args...);

  // define the population trainer parameters
  PopulationTrainerExt<float> population_trainer;
  setPopulationTrainerParameters(population_trainer, args...);

  // define the population logger
  PopulationLogger<float> population_logger(true, true);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  // define the data simulator
  const std::size_t input_size = 784;
  const std::size_t training_data_size = 60000; //60000;
  const std::size_t validation_data_size = 10000; //10000;
  const std::size_t output_size = 10;
  DataSimulatorExt<float> data_simulator;
  data_simulator.n_hidden_0_ = std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get();
  data_simulator.n_hidden_1_ = std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get();
  data_simulator.add_gaussian_ = std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get();

  // read in the training data
  std::string training_data_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "train-images.idx3-ubyte";
  std::string training_labels_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "train-labels.idx1-ubyte";
  data_simulator.readData(training_data_filename, training_labels_filename, true, training_data_size, input_size);

  // read in the validation data
  std::string validation_data_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "t10k-images.idx3-ubyte";
  std::string validation_labels_filename = std::get<EvoNetParameters::General::DataDir>(parameters).get() + "t10k-labels.idx1-ubyte";
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

  // Make the encoding nodes and add them to the input
  assert((
    std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0 && std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) || (
      std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() == 0 && std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() == 0)
  );
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get()) {
    if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0 && std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
      for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++j) {
          char name_char[512];
          sprintf(name_char, "EN0-Input_%012d-Gaussian_%012d-Sampler", i, j);
          std::string name(name_char);
          input_nodes.push_back(name);
        }
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
        for (int j = 0; j < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++j) {
          char name_char[512];
          sprintf(name_char, "EN1-EN0_%012d-Gaussian_%012d-Sampler", i, j);
          std::string name(name_char);
          input_nodes.push_back(name);
        }
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++i) {
        for (int j = 0; j < data_simulator.training_labels.dimension(1); ++j) {
          char name_char[512];
          sprintf(name_char, "EN2-EN1_%012d-Gaussian_%012d-Sampler", i, j);
          std::string name(name_char);
          input_nodes.push_back(name);
        }
      }
    }
    else {
      for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < data_simulator.training_labels.dimension(1); ++j) {
          char name_char[512];
          sprintf(name_char, "EN2-Input_%012d-Gaussian_%012d-Sampler", i, j);
          std::string name(name_char);
          input_nodes.push_back(name);
        }
      }
    }
  }

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < data_simulator.mnist_labels.size(); ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // Make the mu nodes and logvar nodes
  std::vector<std::string> encoding_nodes_en0posterior, encoding_nodes_en1posterior, encoding_nodes_en2posterior;
  std::vector<std::string> encoding_nodes_en0prior, encoding_nodes_en1prior, encoding_nodes_en2prior;
  std::vector<std::string> encoding_nodes_en0logvar, encoding_nodes_en1logvar, encoding_nodes_en2logvar;
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get()) {
    if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0 && std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
      for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++j) {
          char* name_char = new char[512];
          sprintf(name_char, "EN0Posterior_%012d", i * std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() + j);
          std::string name(name_char);
          encoding_nodes_en0posterior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN0Prior_%012d", i * std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() + j);
          name = name_char;
          encoding_nodes_en0prior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN0LogVar_%012d", i * std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() + j);
          name = name_char;
          encoding_nodes_en0logvar.push_back(name);
          delete[] name_char;
        }
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(); ++i) {
        for (int j = 0; j < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++j) {
          char* name_char = new char[512];
          sprintf(name_char, "EN1Posterior_%012d", i * std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() + j);
          std::string name(name_char);
          encoding_nodes_en1posterior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN1Prior_%012d", i * std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() + j);
          name = name_char;
          encoding_nodes_en1prior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN1LogVar_%012d", i * std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() + j);
          name = name_char;
          encoding_nodes_en1logvar.push_back(name);
          delete[] name_char;
        }
      }
      for (int i = 0; i < std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(); ++i) {
        for (int j = 0; j < data_simulator.training_labels.dimension(1); ++j) {
          char* name_char = new char[512];
          sprintf(name_char, "EN2Posterior_%012d", i * data_simulator.training_labels.dimension(1) + j);
          std::string name(name_char);
          encoding_nodes_en2posterior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN2Prior_%012d", i * data_simulator.training_labels.dimension(1) + j);
          name = name_char;
          encoding_nodes_en2prior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN2LogVar_%012d", i * data_simulator.training_labels.dimension(1) + j);
          name = name_char;
          encoding_nodes_en2logvar.push_back(name);
          delete[] name_char;
        }
      }
    }
    else {
      for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < data_simulator.training_labels.dimension(1); ++j) {
          char* name_char = new char[512];
          sprintf(name_char, "EN2Posterior_%012d", i * data_simulator.training_labels.dimension(1) + j);
          std::string name(name_char);
          encoding_nodes_en2posterior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN2Prior_%012d", i * data_simulator.training_labels.dimension(1) + j);
          name = name_char;
          encoding_nodes_en2prior.push_back(name);
          name_char = new char[512];
          sprintf(name_char, "EN2LogVar_%012d", i * data_simulator.training_labels.dimension(1) + j);
          name = name_char;
          encoding_nodes_en2logvar.push_back(name);
          delete[] name_char;
        }
      }
    }
  }

  // define the model interpreters
  std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
  setModelInterpreterParameters(model_interpreters, args...);

  // define the model trainer
  ModelTrainerExt<float> model_trainer;
  setModelTrainerParameters(model_trainer, args...);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-24, 1.0)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-24, 1.0)) };
  loss_function_helpers.push_back(loss_function_helper1);
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get()) {
    if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0) {
      loss_function_helper2.output_nodes_ = encoding_nodes_en0posterior;
      loss_function_helper2.loss_functions_ = { std::make_shared<NegativeLogLikelihoodLossOp<float>>(NegativeLogLikelihoodLossOp<float>(1e-6, -1 / model_trainer.getBatchSize())) };
      loss_function_helper2.loss_function_grads_ = { std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>(1e-6, -1 / model_trainer.getBatchSize())) };
      loss_function_helpers.push_back(loss_function_helper2);
      loss_function_helper3.output_nodes_ = encoding_nodes_en0prior;
      loss_function_helper3.loss_functions_ = { std::make_shared<NegativeLogLikelihoodLossOp<float>>(NegativeLogLikelihoodLossOp<float>(1e-6, 1 / model_trainer.getBatchSize())) };
      loss_function_helper3.loss_function_grads_ = { std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize())) };
      loss_function_helpers.push_back(loss_function_helper3);
    }
    if (std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
      loss_function_helper2.output_nodes_ = encoding_nodes_en1posterior;
      loss_function_helper2.loss_functions_ = { std::make_shared<NegativeLogLikelihoodLossOp<float>>(NegativeLogLikelihoodLossOp<float>(1e-6, -1 / model_trainer.getBatchSize())) };
      loss_function_helper2.loss_function_grads_ = { std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>(1e-6, -1 / model_trainer.getBatchSize())) };
      loss_function_helpers.push_back(loss_function_helper2);
      loss_function_helper3.output_nodes_ = encoding_nodes_en1prior;
      loss_function_helper3.loss_functions_ = { std::make_shared<NegativeLogLikelihoodLossOp<float>>(NegativeLogLikelihoodLossOp<float>(1e-6, 1 / model_trainer.getBatchSize())) };
      loss_function_helper3.loss_function_grads_ = { std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize())) };
      loss_function_helpers.push_back(loss_function_helper3);
    }
    loss_function_helper2.output_nodes_ = encoding_nodes_en2posterior;
    loss_function_helper2.loss_functions_ = { std::make_shared<NegativeLogLikelihoodLossOp<float>>(NegativeLogLikelihoodLossOp<float>(1e-6, -1 / model_trainer.getBatchSize())) };
    loss_function_helper2.loss_function_grads_ = { std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>(1e-6, -1 / model_trainer.getBatchSize())) };
    loss_function_helpers.push_back(loss_function_helper2);
    loss_function_helper3.output_nodes_ = encoding_nodes_en2prior;
    loss_function_helper3.loss_functions_ = { std::make_shared<NegativeLogLikelihoodLossOp<float>>(NegativeLogLikelihoodLossOp<float>(1e-6, 1 / model_trainer.getBatchSize())) };
    loss_function_helper3.loss_function_grads_ = { std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>(1e-6, 1 / model_trainer.getBatchSize())) };
    loss_function_helpers.push_back(loss_function_helper3);
  }
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1, metric_function_helper2;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<AccuracyMCMicroOp<float>>(AccuracyMCMicroOp<float>()), std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>()) };
  metric_function_helper1.metric_names_ = { "AccuracyMCMicro", "PrecisionMCMicro" };
  metric_function_helpers.push_back(metric_function_helper1);
  if (std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get()) {
    if (std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get() > 0) {
      metric_function_helper1.output_nodes_ = encoding_nodes_en0logvar;
      metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
      metric_function_helper1.metric_names_ = { "MAE_EN0LogVar" };
      metric_function_helpers.push_back(metric_function_helper1);
    }
    if (std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get() > 0) {
      metric_function_helper1.output_nodes_ = encoding_nodes_en1logvar;
      metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
      metric_function_helper1.metric_names_ = { "MAE_EN1LogVar" };
      metric_function_helpers.push_back(metric_function_helper1);
    }
    metric_function_helper1.output_nodes_ = encoding_nodes_en2logvar;
    metric_function_helper1.metric_functions_ = { std::make_shared<MAEOp<float>>(MAEOp<float>()) };
    metric_function_helper1.metric_names_ = { "MAE_EN2LogVar" };
    metric_function_helpers.push_back(metric_function_helper1);
  }
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the model replicator for growth mode
  ModelReplicatorExt<float> model_replicator;
  setModelReplicatorParameters(model_replicator, args...);

  // define the initial population
  Model<float> model;
  if (std::get<EvoNetParameters::Main::MakeModel>(parameters).get()) {
    std::cout << "Making the model..." << std::endl;
    model_trainer.makeFullyConnBayes(model, input_size, output_size, std::get<EvoNetParameters::ModelTrainer::NHidden0>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::NHidden1>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::NHidden2>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::AddGaussian>(parameters).get(),
      -1, -4, 0.5, true, std::get<EvoNetParameters::ModelTrainer::LearningRate>(parameters).get(), std::get<EvoNetParameters::ModelTrainer::GradientClipping>(parameters).get());  // Baseline
    model.setId(0);
  }
  else {
    ModelFile<float> model_file;
    ModelInterpreterFileDefaultDevice<float> model_interpreter_file;
    loadModelFromParameters(model, model_interpreters.at(0), model_file, model_interpreter_file, args...);
  }
  model.setName(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get()); //So that all output will be written to a specific directory

  // Run the training, evaluation, or evolution
  runTrainEvalEvoFromParameters<float>(model, model_interpreters, model_trainer, population_trainer, model_replicator, data_simulator, model_logger, population_logger, input_nodes, args...);
}

/// MNIST_CovNet_example 0 C:/Users/dmccloskey/Documents/GitHub/mnist/Parameters.csv
int main(int argc, char** argv)
{
  // Parse the user commands
  int id_int = -1;
  std::string parameters_filename = "";
  parseCommandLineArguments(argc, argv, id_int, parameters_filename);

  // Set the parameter names and defaults
  EvoNetParameters::General::ID id("id", -1);
  EvoNetParameters::General::DataDir data_dir("data_dir", std::string(""));
  EvoNetParameters::General::OutputDir output_dir("output_dir", std::string(""));
  EvoNetParameters::Main::DeviceId device_id("device_id", 0);
  EvoNetParameters::Main::ModelName model_name("model_name", "");
  EvoNetParameters::Main::MakeModel make_model("make_model", true);
  EvoNetParameters::Main::LoadModelCsv load_model_csv("load_model_csv", false);
  EvoNetParameters::Main::LoadModelBinary load_model_binary("load_model_binary", false);
  EvoNetParameters::Main::TrainModel train_model("train_model", true);
  EvoNetParameters::Main::EvolveModel evolve_model("evolve_model", false);
  EvoNetParameters::Main::EvaluateModel evaluate_model("evaluate_model", false);
  EvoNetParameters::Main::EvaluateModels evaluate_models("evaluate_models", false);
  EvoNetParameters::Examples::ModelType model_type("model_type", "Solution");
  EvoNetParameters::Examples::SimulationType simulation_type("simulation_type", "");
  EvoNetParameters::PopulationTrainer::PopulationName population_name("population_name", "");
  EvoNetParameters::PopulationTrainer::NGenerations n_generations("n_generations", 1);
  EvoNetParameters::PopulationTrainer::NInterpreters n_interpreters("n_interpreters", 1);
  EvoNetParameters::PopulationTrainer::PruneModelNum prune_model_num("prune_model_num", 10);
  EvoNetParameters::PopulationTrainer::RemoveIsolatedNodes remove_isolated_nodes("remove_isolated_nodes", true);
  EvoNetParameters::PopulationTrainer::CheckCompleteModelInputToOutput check_complete_model_input_to_output("check_complete_model_input_to_output", true);
  EvoNetParameters::PopulationTrainer::PopulationSize population_size("population_size", 128);
  EvoNetParameters::PopulationTrainer::NTop n_top("n_top", 8);
  EvoNetParameters::PopulationTrainer::NRandom n_random("n_random", 8);
  EvoNetParameters::PopulationTrainer::NReplicatesPerModel n_replicates_per_model("n_replicates_per_model", 1);
  EvoNetParameters::PopulationTrainer::ResetModelCopyWeights reset_model_copy_weights("reset_model_copy_weights", true);
  EvoNetParameters::PopulationTrainer::ResetModelTemplateWeights reset_model_template_weights("reset_model_template_weights", true);
  EvoNetParameters::PopulationTrainer::Logging population_logging("population_logging", true);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeFixed set_population_size_fixed("set_population_size_fixed", false);
  EvoNetParameters::PopulationTrainer::SetPopulationSizeDoubling set_population_size_doubling("set_population_size_doubling", true);
  EvoNetParameters::PopulationTrainer::SetTrainingStepsByModelSize set_training_steps_by_model_size("set_training_steps_by_model_size", false);
  EvoNetParameters::ModelTrainer::BatchSize batch_size("batch_size", 32);
  EvoNetParameters::ModelTrainer::MemorySize memory_size("memory_size", 1);
  EvoNetParameters::ModelTrainer::NEpochsTraining n_epochs_training("n_epochs_training", 1000);
  EvoNetParameters::ModelTrainer::NEpochsValidation n_epochs_validation("n_epochs_validation", 25);
  EvoNetParameters::ModelTrainer::NEpochsEvaluation n_epochs_evaluation("n_epochs_evaluation", 10);
  EvoNetParameters::ModelTrainer::NTBTTSteps n_tbtt_steps("n_tbtt_steps", 64);
  EvoNetParameters::ModelTrainer::NTETTSteps n_tett_steps("n_tett_steps", 64);
  EvoNetParameters::ModelTrainer::Verbosity verbosity("verbosity", 1);
  EvoNetParameters::ModelTrainer::LoggingTraining logging_training("logging_training", true);
  EvoNetParameters::ModelTrainer::LoggingValidation logging_validation("logging_validation", false);
  EvoNetParameters::ModelTrainer::LoggingEvaluation logging_evaluation("logging_evaluation", true);
  EvoNetParameters::ModelTrainer::FindCycles find_cycles("find_cycles", true);
  EvoNetParameters::ModelTrainer::FastInterpreter fast_interpreter("fast_interpreter", true);
  EvoNetParameters::ModelTrainer::PreserveOoO preserve_ooo("preserve_ooo", true);
  EvoNetParameters::ModelTrainer::InterpretModel interpret_model("interpret_model", true);
  EvoNetParameters::ModelTrainer::ResetInterpreter reset_interpreter("reset_interpreter", true);
  EvoNetParameters::ModelTrainer::ResetModel reset_model("reset_model", false);
  EvoNetParameters::ModelTrainer::NHidden0 n_hidden_0("n_hidden_0", 128);
  EvoNetParameters::ModelTrainer::NHidden1 n_hidden_1("n_hidden_1", 128);
  EvoNetParameters::ModelTrainer::NHidden2 n_hidden_2("n_hidden_2", 0);
  EvoNetParameters::ModelTrainer::AddGaussian add_gaussian("add_gaussian", true);
  EvoNetParameters::ModelTrainer::AddMixedGaussian add_mixed_gaussian("add_mixed_gaussian", false);
  EvoNetParameters::ModelTrainer::LearningRate learning_rate("learning_rate", 1e-3);
  EvoNetParameters::ModelTrainer::GradientClipping gradient_clipping("gradient_clipping", 10);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsLB n_node_down_additions_lb("n_node_down_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsLB n_node_right_additions_lb("n_node_right_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesLB n_node_down_copies_lb("n_node_down_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesLB n_node_right_copies_lb("n_node_right_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsLB n_link_additons_lb("n_link_additons_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesLB n_link_copies_lb("n_link_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsLB n_node_deletions_lb("n_node_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsLB n_link_deletions_lb("n_link_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesLB n_node_activation_changes_lb("n_node_activation_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesLB n_node_integration_changes_lb("n_node_integration_changes_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsLB n_module_additions_lb("n_module_additions_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesLB n_module_copies_lb("n_module_copies_lb", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsLB n_module_deletions_lb("n_module_deletions_lb", 0);
  EvoNetParameters::ModelReplicator::NNodeDownAdditionsUB n_node_down_additions_ub("n_node_down_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightAdditionsUB n_node_right_additions_ub("n_node_right_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDownCopiesUB n_node_down_copies_ub("n_node_down_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeRightCopiesUB n_node_right_copies_ub("n_node_right_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkAdditionsUB n_link_additons_ub("n_link_additons_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkCopiesUB n_link_copies_ub("n_link_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeDeletionsUB n_node_deletions_ub("n_node_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NLinkDeletionsUB n_link_deletions_ub("n_link_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeActivationChangesUB n_node_activation_changes_ub("n_node_activation_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NNodeIntegrationChangesUB n_node_integration_changes_ub("n_node_integration_changes_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleAdditionsUB n_module_additions_ub("n_module_additions_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleCopiesUB n_module_copies_ub("n_module_copies_ub", 0);
  EvoNetParameters::ModelReplicator::NModuleDeletionsUB n_module_deletions_ub("n_module_deletions_ub", 0);
  EvoNetParameters::ModelReplicator::SetModificationRateFixed set_modification_rate_fixed("set_modification_rate_fixed", false);
  EvoNetParameters::ModelReplicator::SetModificationRateByPrevError set_modification_rate_by_prev_error("set_modification_rate_by_prev_error", false);
  auto parameters = std::make_tuple(id, data_dir, output_dir,
    device_id, model_name, make_model, load_model_csv, load_model_binary, train_model, evolve_model, evaluate_model, evaluate_models,
    model_type, simulation_type,
    population_name, n_generations, n_interpreters, prune_model_num, remove_isolated_nodes, check_complete_model_input_to_output, population_size, n_top, n_random, n_replicates_per_model, reset_model_copy_weights, reset_model_template_weights, population_logging, set_population_size_fixed, set_population_size_doubling, set_training_steps_by_model_size,
    batch_size, memory_size, n_epochs_training, n_epochs_validation, n_epochs_evaluation, n_tbtt_steps, n_tett_steps, verbosity, logging_training, logging_validation, logging_evaluation, find_cycles, fast_interpreter, preserve_ooo, interpret_model, reset_model, reset_interpreter,
    n_hidden_0, n_hidden_1, n_hidden_2, add_gaussian, add_mixed_gaussian, learning_rate, gradient_clipping,
    n_node_down_additions_lb, n_node_right_additions_lb, n_node_down_copies_lb, n_node_right_copies_lb, n_link_additons_lb, n_link_copies_lb, n_node_deletions_lb, n_link_deletions_lb, n_node_activation_changes_lb, n_node_integration_changes_lb, n_module_additions_lb, n_module_copies_lb, n_module_deletions_lb, n_node_down_additions_ub, n_node_right_additions_ub, n_node_down_copies_ub, n_node_right_copies_ub, n_link_additons_ub, n_link_copies_ub, n_node_deletions_ub, n_link_deletions_ub, n_node_activation_changes_ub, n_node_integration_changes_ub, n_module_additions_ub, n_module_copies_ub, n_module_deletions_ub, set_modification_rate_fixed, set_modification_rate_by_prev_error);

  // Read in the parameters
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = SmartPeak::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);

  // Run the application
  SmartPeak::apply([](auto&& ...args) { main_MNIST(args ...); }, parameters);
  return 0;
}