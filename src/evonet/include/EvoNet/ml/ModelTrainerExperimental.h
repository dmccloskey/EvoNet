/**TODO:  Add copyright*/

#ifndef EVONET_MODELTRAINEREXPERIMENTAL_H
#define EVONET_MODELTRAINEREXPERIMENTAL_H

// .h
#include <EvoNet/ml/ModelTrainer.h>

// .cpp

namespace EvoNet
{
  /**
    @brief Experimental features of `ModelTrainer`
  */
	template<typename TensorT, typename InterpreterT>
  class ModelTrainerExperimental: public ModelTrainer<TensorT, InterpreterT>
  {
public:
    ModelTrainerExperimental() = default; ///< Default constructor
    ~ModelTrainerExperimental() = default; ///< Default destructor

    /// Overrides used in all examples
    void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override;
    void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override;
    void validationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override;
    void evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes) override;
    void evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const Eigen::Tensor<TensorT, 1>& model_metrics) override;
  };	
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainerExperimental<TensorT, InterpreterT>::trainingModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error)
  { // Left blank intentionally to prevent writing of files during population training
  }
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainerExperimental<TensorT, InterpreterT>::trainingModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test)
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1000 == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
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
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainerExperimental<TensorT, InterpreterT>::validationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error)
  {  // Left blank intentionally to prevent writing of files during population validation
  }
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainerExperimental<TensorT, InterpreterT>::evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes)
  { // Left blank intentionally to prevent writing of files during population evaluation
  }
  template<typename TensorT, typename InterpreterT>
  inline void ModelTrainerExperimental<TensorT, InterpreterT>::evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const Eigen::Tensor<TensorT, 1>& model_metrics)
  {
    // Set the defaults
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(false);
    model_logger.setLogNodeInputsEpoch(false);
    model_logger.setLogNodeOutputsEpoch(false);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_logger.initLogs(model);
    }

    // Per n epoch logging
    if (n_epochs % 1 == 0) { // FIXME
      model_logger.setLogExpectedEpoch(true);
      model_logger.setLogNodeInputsEpoch(true);
      model_logger.setLogNodeOutputsEpoch(true);
      model_interpreter.getModelResults(model, true, false, false, true);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_headers;
    std::vector<TensorT> log_values;
    int metric_iter = 0;
    for (const std::string& metric_name : this->getMetricNamesLinearized()) {
      log_headers.push_back(metric_name);
      log_values.push_back(model_metrics(metric_iter));
      ++metric_iter;
    }
    model_logger.writeLogs(model, n_epochs, log_headers, {}, log_values, {}, output_nodes, expected_values, {}, output_nodes, {}, input_nodes, {});
  }
}
#endif //EVONET_MODELTRAINEREXPERIMENTAL_H