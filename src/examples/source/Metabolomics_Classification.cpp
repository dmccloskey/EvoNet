/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelTrainerDefaultDevice.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/simulator/MetabolomicsClassificationDataSimulator.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Other extended classes
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

/// Script to run the classification network
void main_classification(const std::string& data_dir, const std::string& biochem_rxns_filename,
  const std::string& metabo_data_filename_train, const std::string& meta_data_filename_train,
  const std::string& metabo_data_filename_test, const std::string& meta_data_filename_test,
  const bool& make_model, const bool& train_model,
  const bool& use_concentrations, const bool& use_MARs,
  const bool& sample_values, const bool& iter_values,
  const bool& fill_sampling, const bool& fill_mean, const bool& fill_zero,
  const bool& apply_fold_change, const std::string& fold_change_ref, const float& fold_change_log_base,
  const bool& offline_linear_scale_input, const bool& offline_log_transform_input, const bool& offline_standardize_input,
  const bool& online_linear_scale_input, const bool& online_log_transform_input, const bool& online_standardize_input,
  const int& device_id)
{
  // global local variables
  const int n_epochs = 20000;
  const int batch_size = 64;
  const int memory_size = 1;
  //const int n_reps_per_sample = 10000;

  // prior to using shuffle when making the data caches
  const int n_labels = 7; // IndustrialStrains0103
  const int n_reps_per_sample = n_epochs * batch_size / n_labels;

  //std::string model_name = "MetClass_" + std::to_string(use_concentrations) + "-" + std::to_string(use_MARs) + "-" + std::to_string(sample_values) + "-" + std::to_string(iter_values) + "-"
  //  + std::to_string(fill_sampling) + "-" + std::to_string(fill_mean) + "-" + std::to_string(fill_zero) + "-" + std::to_string(apply_fold_change) + "-" + std::to_string(fold_change_log_base) + "-"
  //  + std::to_string(offline_linear_scale_input) + "-" + std::to_string(offline_log_transform_input) + "-" + std::to_string(offline_standardize_input) + "-"
  //  + std::to_string(online_linear_scale_input) + "-" + std::to_string(online_log_transform_input) + "-" + std::to_string(online_standardize_input);
  std::string model_name = "Classifier";

  // define the data simulator
  std::cout << "Making the training and validation data..." << std::endl;
  MetabolomicsClassificationDataSimulator<float> metabolomics_data;
  int n_reaction_ids_training, n_labels_training, n_component_group_names_training;
  int n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation;
  metabolomics_data.readAndProcessMetabolomicsTrainingAndValidationData(
    n_reaction_ids_training, n_labels_training, n_component_group_names_training, n_reaction_ids_validation, n_labels_validation, n_component_group_names_validation,
    biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero, apply_fold_change, fold_change_ref, fold_change_log_base,
    offline_linear_scale_input, offline_log_transform_input, offline_standardize_input, online_linear_scale_input, online_log_transform_input, online_standardize_input,
    n_reps_per_sample, true, false, n_epochs, batch_size, memory_size);

  // define the model input/output nodes
  int n_input_nodes;
  if (use_MARs) n_input_nodes = n_reaction_ids_training;
  else n_input_nodes = n_component_group_names_training;
  const int n_output_nodes = n_labels_training;

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
  ModelResources model_resources = { ModelDevice(device_id, 1) };
  ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
  ModelTrainerExt<float> model_trainer;
  model_trainer.setBatchSize(batch_size);
  model_trainer.setMemorySize(memory_size);
  model_trainer.setNEpochsTraining(n_epochs * 5); // Iterate through the stored data 5 times
  model_trainer.setNEpochsValidation(0);
  model_trainer.setVerbosityLevel(1);
  model_trainer.setLogging(true, false, false);
  model_trainer.setFindCycles(false);
  model_trainer.setFastInterpreter(true);
  model_trainer.setPreserveOoO(true);

  std::vector<LossFunctionHelper<float>> loss_function_helpers;
  LossFunctionHelper<float> loss_function_helper1, loss_function_helper2, loss_function_helper3;
  loss_function_helper1.output_nodes_ = output_nodes;
  loss_function_helper1.loss_functions_ = { std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-8, 1)) };
  loss_function_helper1.loss_function_grads_ = { std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-8, 1)) };
  model_trainer.setLossFunctionHelpers(loss_function_helpers);

  std::vector<MetricFunctionHelper<float>> metric_function_helpers;
  MetricFunctionHelper<float> metric_function_helper1;
  metric_function_helper1.output_nodes_ = output_nodes;
  metric_function_helper1.metric_functions_ = { std::make_shared<AccuracyMCMicroOp<float>>(AccuracyMCMicroOp<float>()), std::make_shared<PrecisionMCMicroOp<float>>(PrecisionMCMicroOp<float>()) };
  metric_function_helper1.metric_names_ = { "AccuracyMCMicro", "PrecisionMCMicro" };
  metric_function_helpers.push_back(metric_function_helper1);
  model_trainer.setMetricFunctionHelpers(metric_function_helpers);

  // define the model logger
  ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

  Model<float> model;
  if (make_model) {
    std::cout << "Making the model..." << std::endl;
    //model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, true, true, true, 64, 64, 0); // normalization type 4 (Log transformation + Standardization + Projection)
    model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes, online_linear_scale_input, online_log_transform_input, online_standardize_input, 8, 0, 0);
  }
  else {
    // TODO
  }
  model.setName(data_dir + model_name); //So that all output will be written to a specific directory

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
  //std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";

  // Initialize the defaults
  std::string data_dir = "";
  std::string biochem_rxns_filename = data_dir + "iJO1366.csv";
  std::string metabo_data_filename_train = data_dir + "ALEsKOs01_Metabolomics_train.csv"; // IndustrialStrains0103_
  std::string meta_data_filename_train = data_dir + "ALEsKOs01_MetaData_train.csv";
  std::string metabo_data_filename_test = data_dir + "ALEsKOs01_Metabolomics_test.csv";
  std::string meta_data_filename_test = data_dir + "ALEsKOs01_MetaData_test.csv";
  bool make_model = true;
  bool train_model = true;
  bool use_concentrations = true;
  bool use_MARs = false;
  bool sample_values = true;
  bool iter_values = false;
  bool fill_sampling = false;
  bool fill_mean = false;
  bool fill_zero = false;
  bool apply_fold_change = false;
  std::string fold_change_ref = "Evo04";
  float fold_change_log_base = 10;
  bool offline_linear_scale_input = true;
  bool offline_log_transform_input = false;
  bool offline_standardize_input = false;
  bool online_linear_scale_input = false;
  bool online_log_transform_input = false;
  bool online_standardize_input = false;
  int device_id = 1;

  // Parse the input
  std::cout << "Parsing the user input..." << std::endl;
  if (argc >= 2) {
    data_dir = argv[1];
  }
  if (argc >= 3) {
    biochem_rxns_filename = argv[2];
  }
  if (argc >= 4) {
    metabo_data_filename_train = argv[3];
  }
  if (argc >= 5) {
    meta_data_filename_train = argv[4];
  }
  if (argc >= 6) {
    metabo_data_filename_test = argv[5];
  }
  if (argc >= 7) {
    meta_data_filename_test = argv[6];
  }
  if (argc >= 8) {
    make_model = (argv[7] == std::string("true")) ? true : false;
  }
  if (argc >= 9) {
    train_model = (argv[8] == std::string("true")) ? true : false;
  }
  if (argc >= 10) {
    use_concentrations = (argv[9] == std::string("true")) ? true : false;
  }
  if (argc >= 11) {
    use_MARs = (argv[10] == std::string("true")) ? true : false;
  }
  if (argc >= 12) {
    sample_values = (argv[11] == std::string("true")) ? true : false;
  }
  if (argc >= 13) {
    iter_values = (argv[12] == std::string("true")) ? true : false;
  }
  if (argc >= 14) {
    fill_sampling = (argv[13] == std::string("true")) ? true : false;
  }
  if (argc >= 15) {
    fill_mean = (argv[14] == std::string("true")) ? true : false;
  }
  if (argc >= 16) {
    fill_zero = (argv[15] == std::string("true")) ? true : false;
  }
  if (argc >= 17) {
    apply_fold_change = (argv[16] == std::string("true")) ? true : false;
  }
  if (argc >= 18) {
    fold_change_ref = argv[17];
  }
  if (argc >= 19) {
    try {
      fold_change_log_base = std::stof(argv[18]);
    }
    catch (std::exception& e) {
      std::cout << e.what() << std::endl;
    }
  }
  if (argc >= 20) {
    offline_linear_scale_input = (argv[19] == std::string("true")) ? true : false;
  }
  if (argc >= 21) {
    offline_log_transform_input = (argv[20] == std::string("true")) ? true : false;
  }
  if (argc >= 22) {
    offline_standardize_input = (argv[21] == std::string("true")) ? true : false;
  }
  if (argc >= 23) {
    online_linear_scale_input = (argv[22] == std::string("true")) ? true : false;
  }
  if (argc >= 24) {
    online_log_transform_input = (argv[23] == std::string("true")) ? true : false;
  }
  if (argc >= 25) {
    online_standardize_input = (argv[24] == std::string("true")) ? true : false;
  }
  if (argc >= 26) {
    try {
      device_id = std::stoi(argv[25]);
    }
    catch (std::exception& e) {
      std::cout << e.what() << std::endl;
    }
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
  std::cout << "use_concentrations: " << use_concentrations << std::endl;
  std::cout << "use_MARs: " << use_MARs << std::endl;
  std::cout << "sample_values: " << sample_values << std::endl;
  std::cout << "iter_values: " << iter_values << std::endl;
  std::cout << "fill_sampling: " << fill_sampling << std::endl;
  std::cout << "fill_mean: " << fill_mean << std::endl;
  std::cout << "fill_zero: " << fill_zero << std::endl;
  std::cout << "apply_fold_change: " << apply_fold_change << std::endl;
  std::cout << "fold_change_ref: " << fold_change_ref << std::endl;
  std::cout << "fold_change_log_base: " << fold_change_log_base << std::endl;
  std::cout << "offline_linear_scale_input: " << offline_linear_scale_input << std::endl;
  std::cout << "offline_log_transform_input: " << offline_log_transform_input << std::endl;
  std::cout << "offline_standardize_input: " << offline_standardize_input << std::endl;
  std::cout << "online_linear_scale_input: " << online_linear_scale_input << std::endl;
  std::cout << "online_log_transform_input: " << online_log_transform_input << std::endl;
  std::cout << "online_standardize_input: " << online_standardize_input << std::endl;
  std::cout << "device_id: " << device_id << std::endl;

  // Run the classification
  main_classification(data_dir, biochem_rxns_filename, metabo_data_filename_train, meta_data_filename_train, metabo_data_filename_test, meta_data_filename_test,
    make_model, train_model,
    use_concentrations, use_MARs, sample_values, iter_values, fill_sampling, fill_mean, fill_zero,
    apply_fold_change, fold_change_ref, fold_change_log_base,
    offline_linear_scale_input, offline_log_transform_input, offline_standardize_input,
    online_linear_scale_input, online_log_transform_input, online_standardize_input,
    device_id
  );
  return 0;
}