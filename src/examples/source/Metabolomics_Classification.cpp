/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/simulator/BiochemicalDataSimulator.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Other extended classes
template<typename TensorT>
class MetDataSimClassification : public BiochemicalDataSimulator<TensorT>
{
public:
  std::vector<std::string> labels_training_;
  std::vector<std::string> labels_validation_;
  void makeTrainingDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_training, const std::vector<std::string>& labels_training,
    const int& n_epochs, const int& batch_size, const int& memory_size,
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) override {

    // infer the input sizes
    const int input_nodes = data_training.dimension(0);
    assert(n_input_nodes == input_nodes);
    assert(n_loss_output_nodes == labels_training_.size());
    assert(n_metric_output_nodes == 2* labels_training_.size()); // accuracy and precision
    assert(data_training.dimension(0) == features.size());
    assert(data_training.dimension(1) == labels_training.size());

    // initialize the Tensors
    this->input_data_training_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_training_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_training_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_training_.resize(batch_size, memory_size, n_epochs);

    // expand the training data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_training.dimension(1))), 1);
    const int over_expanded = data_training.dimension(1)*expansion_factor - batch_size * n_epochs;
    assert(batch_size * memory_size * n_epochs == data_training.dimension(1)*expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_training_expanded(data_training.dimension(0), data_training.dimension(1)*expansion_factor);
    Eigen::Tensor<std::string, 2> labels_training_expanded(data_training.dimension(1)*expansion_factor, 1);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = {0, i*data_training.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_training.dimension(0), data_training.dimension(1) };
      data_training_expanded.slice(offset1, span1) = data_training;

      // Slices for the labels
      for (int j = 0; j < data_training.dimension(1); ++j) {
        labels_training_expanded(i*data_training.dimension(1) + j, 0) = labels_training.at(j);
      }
      //Eigen::array<Eigen::Index, 2> offset2 = { i*data_training.dimension(1), 0 };
      //Eigen::array<Eigen::Index, 2> span2 = { data_training.dimension(1), 1 };
      //Eigen::TensorMap<Eigen::Tensor<std::string, 2>> labels_2d(labels_training.data(), data_training.dimension(1), 1);
      //labels_training_expanded.slice(offset2, span2) = labels_2d;
    }

    // assign the input tensors
    auto data_training_expanded_4d = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), data_training.dimension(1)*expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_training.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({1,2,0,3}));
    this->input_data_training_ = data_training_expanded_4d;

    // Check that values of the data and input tensors are correctly aligned
    Eigen::Tensor<TensorT, 1> data_training_head = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), 1 })
    ).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    Eigen::Tensor<TensorT, 1> data_training_tail = data_training_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), data_training.dimension(1)*expansion_factor - over_expanded })
    ).slice(Eigen::array<Eigen::Index, 2>({ 0, batch_size * memory_size * n_epochs - 1 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(0), 1 })
    ).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    Eigen::Tensor<TensorT, 1> input_training_head = this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ 1, 1, data_training.dimension(0), 1 })
    ).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    Eigen::Tensor<TensorT, 1> input_training_tail = this->input_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
      Eigen::array<Eigen::Index, 4>({ 1, 1, data_training.dimension(0), 1 })
    ).reshape(Eigen::array<Eigen::Index, 1>({ data_training.dimension(0) }));
    for (int i = 0; i < data_training.dimension(0); ++i) {
      assert(data_training_head(i) == input_training_head(i));
      assert(data_training_tail(i) == input_training_tail(i));
    }

    // make the one-hot encodings       
    Eigen::Tensor<TensorT, 2> one_hot_vec = OneHotEncoder<std::string, TensorT>(labels_training_expanded, this->labels_training_);
    //Eigen::Tensor<TensorT, 2> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

    // assign the loss tensors
    auto one_hot_vec_4d = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(1)*expansion_factor - over_expanded, one_hot_vec.dimension(1) })
    ).reshape(Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_epochs, int(labels_training_.size()) })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 0,1,3,2 }));
    this->loss_output_data_training_ = one_hot_vec_4d;

    // Check that values of the labels and output tensors are correctly aligned
    Eigen::Tensor<TensorT, 1> labels_training_head = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ 1, int(labels_training_.size()) })
    ).reshape(Eigen::array<Eigen::Index, 1>({ int(labels_training_.size()) }));
    Eigen::Tensor<TensorT, 1> labels_training_tail = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_training.dimension(1)*expansion_factor - over_expanded, one_hot_vec.dimension(1) })
    ).slice(Eigen::array<Eigen::Index, 2>({ batch_size * memory_size * n_epochs - 1, 0 }),
      Eigen::array<Eigen::Index, 2>({ 1, int(labels_training_.size()) })
    ).reshape(Eigen::array<Eigen::Index, 1>({ int(labels_training_.size()) }));
    Eigen::Tensor<TensorT, 1> loss_training_head = this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ 1, 1, int(labels_training_.size()), 1 })
    ).reshape(Eigen::array<Eigen::Index, 1>({ int(labels_training_.size()) }));
    Eigen::Tensor<TensorT, 1> loss_training_tail = this->loss_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ batch_size - 1, memory_size - 1, 0, n_epochs - 1 }),
      Eigen::array<Eigen::Index, 4>({ 1, 1, int(labels_training_.size()), 1 })
    ).reshape(Eigen::array<Eigen::Index, 1>({ int(labels_training_.size()) }));
    for (int i = 0; i < int(labels_training_.size()); ++i) {
      assert(labels_training_head(i) == loss_training_head(i));
      assert(labels_training_tail(i) == loss_training_tail(i));
    }

    // assign the metric tensors
    this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_training_.size()), n_epochs })) = one_hot_vec_4d;
    this->metric_output_data_training_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, int(labels_training_.size()), 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_training_.size()), n_epochs })) = one_hot_vec_4d;
  }
  void makeValidationDataForCache(const std::vector<std::string>& features, const Eigen::Tensor<TensorT, 2>& data_validation, const std::vector<std::string>& labels_validation,
    const int& n_epochs, const int& batch_size, const int& memory_size,
    const int& n_input_nodes, const int& n_loss_output_nodes, const int& n_metric_output_nodes) override {

    // infer the input sizes
    const int input_nodes = data_validation.dimension(0);
    assert(n_input_nodes == input_nodes);
    assert(n_loss_output_nodes == labels_validation_.size());
    assert(n_metric_output_nodes == 2 * labels_validation_.size()); // accuracy and precision
    assert(data_validation.dimension(0) == features.size());
    assert(data_validation.dimension(1) == labels_validation.size());

    // initialize the Tensors
    this->input_data_validation_.resize(batch_size, memory_size, n_input_nodes, n_epochs);
    this->loss_output_data_validation_.resize(batch_size, memory_size, n_loss_output_nodes, n_epochs);
    this->metric_output_data_validation_.resize(batch_size, memory_size, n_metric_output_nodes, n_epochs);
    this->time_steps_validation_.resize(batch_size, memory_size, n_epochs);

    // expand the validation data to fit into the requested input size
    const int expansion_factor = maxFunc(std::ceil(TensorT(batch_size * n_epochs) / TensorT(data_validation.dimension(1))), 1);
    const int over_expanded = data_validation.dimension(1)*expansion_factor - batch_size * n_epochs;
    assert(batch_size * memory_size * n_epochs == data_validation.dimension(1)*expansion_factor - over_expanded);
    Eigen::Tensor<TensorT, 2> data_validation_expanded(data_validation.dimension(0), data_validation.dimension(1)*expansion_factor);
    Eigen::Tensor<std::string, 2> labels_validation_expanded(data_validation.dimension(1)*expansion_factor, 1);
    for (int i = 0; i < expansion_factor; ++i) {
      // Slices for the data
      Eigen::array<Eigen::Index, 2> offset1 = { 0, i*data_validation.dimension(1) };
      Eigen::array<Eigen::Index, 2> span1 = { data_validation.dimension(0), data_validation.dimension(1) };
      data_validation_expanded.slice(offset1, span1) = data_validation;

      // Slices for the labels
      for (int j = 0; j < data_validation.dimension(1); ++j) {
        labels_validation_expanded(i*data_validation.dimension(1) + j, 0) = labels_validation.at(j);
      }
      //Eigen::array<Eigen::Index, 2> offset2 = { i*data_validation.dimension(1), 0 };
      //Eigen::array<Eigen::Index, 2> span2 = { data_validation.dimension(1), 1 };
      //Eigen::TensorMap<Eigen::Tensor<std::string, 2>> labels_2d(labels_validation.data(), data_validation.dimension(1), 1);
      //labels_validation_expanded.slice(offset2, span2) = labels_2d;
    }

    // assign the input tensors
    auto data_validation_expanded_4d = data_validation_expanded.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_validation.dimension(0), data_validation.dimension(1)*expansion_factor - over_expanded })
    ).reshape(Eigen::array<Eigen::Index, 4>({ data_validation.dimension(0), batch_size, memory_size, n_epochs })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 1,2,0,3 }));
    this->input_data_validation_ = data_validation_expanded_4d;

    // make the one-hot encodings       
    Eigen::Tensor<TensorT, 2> one_hot_vec = OneHotEncoder<std::string, TensorT>(labels_validation_expanded, this->labels_validation_);
    //Eigen::Tensor<TensorT, 2> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

    // assign the loss tensors
    auto one_hot_vec_4d = one_hot_vec.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }),
      Eigen::array<Eigen::Index, 2>({ data_validation.dimension(1)*expansion_factor - over_expanded, one_hot_vec.dimension(1) })
    ).reshape(Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, n_epochs, int(labels_validation_.size()) })
    ).shuffle(Eigen::array<Eigen::Index, 4>({ 0,1,3,2 }));
    this->loss_output_data_validation_ = one_hot_vec_4d;

    // assign the metric tensors
    this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_validation_.size()), n_epochs })) = one_hot_vec_4d;
    this->metric_output_data_validation_.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, int(labels_validation_.size()), 0 }),
      Eigen::array<Eigen::Index, 4>({ batch_size, memory_size, int(labels_validation_.size()), n_epochs })) = one_hot_vec_4d;
  }
};

template<typename TensorT>
class ModelTrainerExt : public ModelTrainerDefaultDevice<TensorT>
{
public:
  /*
  @brief Fully connected classifier
  */
  void makeModelFCClass(Model<TensorT>& model, const int& n_inputs, const int& n_outputs, const bool& linear_scale_input, const bool& log_transform_input, const bool& standardize_input,
    const int& n_hidden_0 = 32, const int& n_hidden_1 = 0, const int& n_hidden_2 = 0) {
    model.setId(0);
    model.setName("Classifier");
    ModelBuilder<TensorT> model_builder;

    // Add the inputs
    std::vector<std::string> node_names = model_builder.addInputNodes(model, "Input", "Input", n_inputs, true);

    // Data online pre-processing steps
    this->addDataPreproccessingSteps(model, node_names, linear_scale_input, log_transform_input, standardize_input);

    // Define the activation based on `add_feature_norm`
    auto activation = std::make_shared<LeakyReLUOp<TensorT>>(LeakyReLUOp<TensorT>());
    auto activation_grad = std::make_shared<LeakyReLUGradOp<TensorT>>(LeakyReLUGradOp<TensorT>());

    // Define the node integration
    auto integration_op = std::make_shared<SumOp<TensorT>>(SumOp<TensorT>());
    auto integration_error_op = std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>());
    auto integration_weight_grad_op = std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>());

    // Define the solver
    auto solver_op = std::make_shared<AdamOp<TensorT>>(AdamOp<TensorT>(1e-4, 0.9, 0.999, 1e-8, 10));

    // Add the hidden layers
    if (n_hidden_0 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC0", "FC0", node_names, n_hidden_0,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_0) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, true);
    }
    if (n_hidden_1 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC1", "FC1", node_names, n_hidden_1,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_1) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, true);
    }
    if (n_hidden_2 > 0) {
      node_names = model_builder.addFullyConnected(model, "FC2", "FC2", node_names, n_hidden_2,
        activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
        std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>((int)(node_names.size() + n_hidden_2) / 2, 1)),
        solver_op, 0.0f, 0.0f, false, true);
    }

    // Add the final output layer
    node_names = model_builder.addFullyConnected(model, "FC-Output", "FC-Output", node_names, n_outputs,
      activation, activation_grad, integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<RandWeightInitOp<TensorT>>(RandWeightInitOp<TensorT>(node_names.size(), 2)),
      solver_op, 0.0f, 0.0f, false, true);

    // Add the dummy output layer
    node_names = model_builder.addSinglyConnected(model, "Output", "Output", node_names, n_outputs,
      std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()),
      std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
      integration_op, integration_error_op, integration_weight_grad_op,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
      std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0f, 0.0f, false, true);

    // Manually define the output nodes
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
        std::make_shared<LogOp<TensorT>>(LogOp<TensorT>()),
        std::make_shared<LogGradOp<TensorT>>(LogGradOp<TensorT>()),
        std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()),
        std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()),
        std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)),
        std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, true);
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
    ModelInterpreterDefaultDevice<TensorT>& model_interpreter,
    const std::vector<float>& model_errors) {
    if (n_epochs % 1000 == 0) { // store on n_epochs == 0
    //if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;

      //// Save weights to .csv
      //data.storeModelCsv(model.getName() + "_" + std::to_string(n_epochs) + "_nodes.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_links.csv",
      //	model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model, false, false, true);

      // Save to binary
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
  void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger,
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
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, false);
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
    model_logger.writeLogs(model, n_epochs, log_train_headers, log_test_headers, log_train_values, log_test_values, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
};

/// Script to run the classification network
void main_classification(const std::string& data_dir, const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
  const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
  const bool& make_model = true, const bool& train_model = true, const int& norm_method = 0,
  const bool& simulate_MARs = true, const bool& sample_concs = true, const bool& use_fold_change = false, const std::string& fold_change_ref = "Evo04")
{
  // global local variables
  const int n_epochs = 100;// 100000;
  const int batch_size = 64;
  const int memory_size = 1;
  const bool fill_sampling = false;
  const bool fill_mean = false;
  const bool fill_zero = true;
  const int n_reps_per_sample = n_epochs * batch_size / 4;

  // define the data simulator
  BiochemicalReactionModel<float> reaction_model;
  MetDataSimClassification<float> metabolomics_data;
  std::string model_name = "0_Metabolomics";

  // Read in the training data
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
  const int n_reaction_ids_training = reaction_model.reaction_ids_.size();
  const int n_labels_training = reaction_model.labels_.size();
  const int n_component_group_names_training = reaction_model.component_group_names_.size();
  metabolomics_data.labels_training_ = reaction_model.labels_;

  // Make the training data caches
  std::map<std::string, int> sample_group_name_to_reps;
  std::pair<int, int> max_reps_n_labels = reaction_model.getMaxReplicatesAndNLabels(sample_group_name_to_reps, reaction_model.sample_group_names_, reaction_model.component_group_names_);
  if (sample_concs) {
    for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;
    std::vector<std::string> metabo_labels;
    metabo_labels.reserve(n_reps_per_sample * sample_group_name_to_reps.size());
    Eigen::Tensor<float, 2> metabo_data(int(reaction_model.component_group_names_.size()), n_reps_per_sample * sample_group_name_to_reps.size());
    reaction_model.getMetDataAsTensors(metabo_data, metabo_labels,
      reaction_model.sample_group_names_, reaction_model.component_group_names_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
      true, false, true, false, false, false, false, use_fold_change, fold_change_ref, 10);
    metabolomics_data.makeTrainingDataForCache(reaction_model.component_group_names_, metabo_data, metabo_labels, n_epochs, batch_size, memory_size,
      reaction_model.component_group_names_.size(), reaction_model.labels_.size(), 2 * reaction_model.labels_.size());
  }
  else if (simulate_MARs) {
    for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;
    std::vector<std::string> metabo_labels;
    metabo_labels.reserve(n_reps_per_sample * sample_group_name_to_reps.size());
    Eigen::Tensor<float, 2> metabo_data(int(reaction_model.reaction_ids_.size()), n_reps_per_sample * sample_group_name_to_reps.size());
    reaction_model.getMetDataAsTensors(metabo_data, metabo_labels,
      reaction_model.sample_group_names_, reaction_model.reaction_ids_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
      false, true, true, false, false, false, false, use_fold_change, fold_change_ref, 10);
    metabolomics_data.makeTrainingDataForCache(reaction_model.reaction_ids_, metabo_data, metabo_labels, n_epochs, batch_size, memory_size,
      reaction_model.reaction_ids_.size(), reaction_model.labels_.size(), 2 * reaction_model.labels_.size());
  }
  else {
    std::vector<std::string> metabo_labels;
    metabo_labels.reserve(max_reps_n_labels.second);
    Eigen::Tensor<float, 2> metabo_data(int(reaction_model.component_group_names_.size()), max_reps_n_labels.second);
    reaction_model.getMetDataAsTensors(metabo_data, metabo_labels,
      reaction_model.sample_group_names_, reaction_model.component_group_names_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
      true, false, false, true, fill_sampling, fill_mean, fill_zero, use_fold_change, fold_change_ref, 10);
    metabolomics_data.makeTrainingDataForCache(reaction_model.component_group_names_, metabo_data, metabo_labels, n_epochs, batch_size, memory_size,
      reaction_model.component_group_names_.size(), reaction_model.labels_.size(), 2 * reaction_model.labels_.size());
  }

  // Read in the validation data
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
  const int n_reaction_ids_validation = reaction_model.reaction_ids_.size();
  const int n_labels_validation = reaction_model.labels_.size();
  const int n_component_group_names_validation = reaction_model.component_group_names_.size();
  metabolomics_data.labels_validation_ = reaction_model.labels_;

  // Make the validation data caches
  sample_group_name_to_reps.clear();
  max_reps_n_labels = reaction_model.getMaxReplicatesAndNLabels(sample_group_name_to_reps, reaction_model.sample_group_names_, reaction_model.component_group_names_);
  if (sample_concs) {
    for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;
    std::vector<std::string> metabo_labels;
    metabo_labels.reserve(n_reps_per_sample * sample_group_name_to_reps.size());
    Eigen::Tensor<float, 2> metabo_data(int(reaction_model.component_group_names_.size()), n_reps_per_sample * sample_group_name_to_reps.size());
    reaction_model.getMetDataAsTensors(metabo_data, metabo_labels,
      reaction_model.sample_group_names_, reaction_model.component_group_names_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
      true, false, true, false, false, false, false, use_fold_change, fold_change_ref, 10);
    metabolomics_data.makeValidationDataForCache(reaction_model.component_group_names_, metabo_data, metabo_labels, n_epochs, batch_size, memory_size,
      reaction_model.component_group_names_.size(), reaction_model.labels_.size(), 2 * reaction_model.labels_.size());
  }
  else if (simulate_MARs) {
    for (auto& sample_group_name_to_rep : sample_group_name_to_reps) sample_group_name_to_rep.second = n_reps_per_sample;
    std::vector<std::string> metabo_labels;
    metabo_labels.reserve(n_reps_per_sample * sample_group_name_to_reps.size());
    Eigen::Tensor<float, 2> metabo_data(int(reaction_model.reaction_ids_.size()), n_reps_per_sample * sample_group_name_to_reps.size());
    reaction_model.getMetDataAsTensors(metabo_data, metabo_labels,
      reaction_model.sample_group_names_, reaction_model.reaction_ids_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
      false, true, true, false, false, false, false, use_fold_change, fold_change_ref, 10);
    metabolomics_data.makeValidationDataForCache(reaction_model.reaction_ids_, metabo_data, metabo_labels, n_epochs, batch_size, memory_size,
      reaction_model.reaction_ids_.size(), reaction_model.labels_.size(), 2 * reaction_model.labels_.size());
  }
  else {
    std::vector<std::string> metabo_labels;
    metabo_labels.reserve(max_reps_n_labels.second);
    Eigen::Tensor<float, 2> metabo_data(int(reaction_model.component_group_names_.size()), max_reps_n_labels.second);
    reaction_model.getMetDataAsTensors(metabo_data, metabo_labels,
      reaction_model.sample_group_names_, reaction_model.component_group_names_, reaction_model.sample_group_name_to_label_, sample_group_name_to_reps,
      true, false, false, true, fill_sampling, fill_mean, fill_zero, use_fold_change, fold_change_ref, 10);
    metabolomics_data.makeValidationDataForCache(reaction_model.component_group_names_, metabo_data, metabo_labels, n_epochs, batch_size, memory_size,
      reaction_model.component_group_names_.size(), reaction_model.labels_.size(), 2 * reaction_model.labels_.size());
  }

  // Checks for the training and validation data
  assert(n_reaction_ids_training == n_reaction_ids_validation);
  assert(n_labels_training == n_labels_validation);
  assert(n_component_group_names_training == n_component_group_names_validation);

  // define the model input/output nodes
  int n_input_nodes;
  if (simulate_MARs) n_input_nodes = reaction_model.reaction_ids_.size();
  else n_input_nodes = reaction_model.component_group_names_.size();
  const int n_output_nodes = reaction_model.labels_.size();

  // define the input nodes
  std::vector<std::string> input_nodes;
  for (int i = 0; i < n_input_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Input_%012d", i);
    std::string name(name_char);
    input_nodes.push_back(name);
  }

  // define the output nodes
  std::vector<std::string> output_nodes;
  for (int i = 0; i < n_output_nodes; ++i) {
    char name_char[512];
    sprintf(name_char, "Output_%012d", i);
    std::string name(name_char);
    output_nodes.push_back(name);
  }

  // define the model trainers and resources for the trainers
  ModelResources model_resources = { ModelDevice(1, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(batch_size);
  model_trainer.setMemorySize(memory_size);
  model_trainer.setNEpochsTraining(n_epochs);
  model_trainer.setNEpochsValidation(0);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);
  model_trainer.setLossFunctions({ std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-8, 1)) });
  model_trainer.setLossFunctionGrads({ std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-8, 1)) });
  model_trainer.setLossOutputNodes({ output_nodes });
  model_trainer.setMetricFunctions({ std::make_shared<AccuracyMCMicroOp<float>>(AccuracyMCMicroOp<float>()), std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>())
    });
  model_trainer.setMetricOutputNodes({ output_nodes, output_nodes });
  model_trainer.setMetricNames({ "AccuracyMCMicro", "PrecisionMCMicro" });

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  Model<float> model;
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    bool linear_scale_input = false;
    bool log_transform_input = false;
    bool standardize_input = false;
    if (norm_method == 0) {// normalization type 0 (No normalization)
    }
    else if (norm_method == 1) {// normalization type 1 (Projection)
      linear_scale_input = true;
    }
    else if (norm_method == 2) {// normalization type 2 (Standardization + Projection)
      linear_scale_input = true;
      standardize_input = true;
    }
    else if (norm_method == 3) {// normalization type 3 (Log transformation + Projection)
      linear_scale_input = true;
      log_transform_input = true;
    }
    else if (norm_method == 4) {// normalization type 4 (Log transformation + Standardization + Projection)
      linear_scale_input = true;
      log_transform_input = true;
      standardize_input = true;
    }
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, true, 64, 64, 0); // normalization type 4 (Log transformation + Standardization + Projection)
    model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, linear_scale_input, log_transform_input, standardize_input, 8, 0, 0);
  }
  else {
    // TODO
  }
  model.setName(data_dir + "Classifier"); //So that all output will be written to a specific directory

  // Train the model
  std::cout << "Training the model..." << std::endl;
  std::pair<std::vector<float>, std::vector<float>> model_errors = model_trainer.trainModel(model, metabolomics_data,
    input_nodes, model_logger, model_interpreter);
}

template<typename TensorT>
void calculateInputLayer0Correlation() {
  // Read in the weights
  Model<TensorT> model;

  // Calculate the average per node weight magnitude for the first layer
  const int n_fc_0 = 64;
  const int n_input = 92;
  Eigen::Tensor<TensorT, 1> weight_ave_values(n_input);
  for (int i = 0; i < n_input; ++i) {
    TensorT weight_sum = 0;
    for (int j = 0; j < n_fc_0; ++j) {
      char weight_name_char[512];
      sprintf(weight_name_char, "Input_%012d_to_FC%012d", i, j);
      std::string weight_name(weight_name_char);
      TensorT weight_value = model.getWeightsMap().at(weight_name)->getWeight();
      weight_sum += weight_value;
    }
    weight_ave_values(i) = weight_sum / n_fc_0;
  }

  // Generate a large sample of input

  // Calculate the Pearson Correlation
}

// Main
int main(int argc, char** argv)
{
  // Set the data directories
  std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";

  // Initialize the defaults
  //std::string data_dir = "";
  std::string biochem_rxns_filename = data_dir + "iJO1366.csv";
  std::string metabo_data_filename_train = data_dir + "ALEsKOs01_Metabolomics_train.csv"; // IndustrialStrains0103_
  std::string meta_data_filename_train = data_dir + "ALEsKOs01_MetaData_train.csv";
  std::string metabo_data_filename_test = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  std::string meta_data_filename_test = data_dir + "ALEsKOs01_MetaData_test.csv";
  bool make_model = true;
  bool train_model = true;
  int norm_method = 0;
  bool simulate_MARs = false;
  bool sample_concs = true;
  bool use_fold_change = false;
  std::string fold_change_ref = "Evo04";

  // Parse the input
  std::cout << "Parsing the user input..." << std::endl;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    biochem_rxns_filename = data_dir + argv[2];
  }
  if (argc >= 4) {
    metabo_data_filename_train = data_dir + argv[3];
  }
  if (argc >= 5) {
    meta_data_filename_train = data_dir + argv[4];
  }
  if (argc >= 6) {
    metabo_data_filename_test = data_dir + argv[5];
  }
  if (argc >= 7) {
    meta_data_filename_test = data_dir + argv[6];
  }
  if (argc >= 8) {
    make_model = (argv[7] == std::string("true")) ? true : false;
  }
  if (argc >= 9) {
    train_model = (argv[8] == std::string("true")) ? true : false;
  }
  if (argc >= 10) {
    try {
      norm_method = (std::stoi(argv[9]) >= 0 && std::stoi(argv[9]) <= 4) ? std::stoi(argv[9]) : 0;
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
  }
  if (argc >= 11) {
    simulate_MARs = (argv[10] == std::string("true")) ? true : false;
  }
  if (argc >= 12) {
    sample_concs = (argv[11] == std::string("true")) ? true : false;
  }
  if (argc >= 13) {
    use_fold_change = (argv[12] == std::string("true")) ? true : false;
  }
  if (argc >= 14) {
    fold_change_ref = argv[13];
  }

  // Cout the parsed input
  std::cout << "data_dir: " << data_dir << std::endl;
  std::cout << "biochem_rxns_filename: " << biochem_rxns_filename << std::endl;
  std::cout << "metabo_data_filename_train: " << metabo_data_filename_train << std::endl;
  std::cout << "meta_data_filename_train: " << meta_data_filename_train << std::endl;
  std::cout << "metabo_data_filename_test: " << metabo_data_filename_test << std::endl;
  std::cout << "meta_data_filename_test: " << meta_data_filename_test << std::endl;
  std::cout << "make_model: " << make_model << std::endl;
  std::cout << "train_model: " << train_model << std::endl;
  std::cout << "norm_method: " << norm_method << std::endl;
  std::cout << "simulate_MARs: " << simulate_MARs << std::endl;
  std::cout << "sample_concs: " << sample_concs << std::endl;
  std::cout << "use_fold_change: " << use_fold_change << std::endl;
  std::cout << "fold_change_ref: " << fold_change_ref << std::endl;

  // Run the classification
  main_classification(data_dir, biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train,
    metabo_data_filename_test, meta_data_filename_test,
    make_model, train_model, norm_method,
    simulate_MARs, sample_concs, use_fold_change, fold_change_ref
  );
  return 0;
}