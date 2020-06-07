/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINEREXPERIMENTAL_H
#define SMARTPEAK_MODELTRAINEREXPERIMENTAL_H

// .h
#include <SmartPeak/ml/ModelTrainer.h>

// .cpp

namespace SmartPeak
{
  template<typename TensorT>
  struct LossFunctionHelper
  {
    std::vector<std::string> output_nodes_; ///< output node names
    std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> loss_functions_; ///< loss functions to apply to the node names
    std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> loss_function_grads_; ///< corresponding loss function grads to apply to the node names
  };

  template<typename TensorT>
  struct MetricFunctionHelper
  {
    std::vector<std::string> output_nodes_; ///< output node names
    std::vector<std::shared_ptr<MetricFunctionOp<TensorT>>> metric_functions_; ///< metric functions to apply to the node names
    std::vector<std::string> metric_names_; ///< corresponding metric function names given for each metric function
  };

  /**
    @brief Class to train a network model
  */
	template<typename TensorT, typename InterpreterT>
  class ModelTrainerExperimental: public ModelTrainer<TensorT, InterpreterT>
  {
public:
    ModelTrainerExperimental() = default; ///< Default constructor
    ~ModelTrainerExperimental() = default; ///< Default destructor

    /// Overrides used in all examples
    void adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, const std::vector<TensorT>& model_errors);
    void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override;
    void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override;
    void validationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error) override;
    void validationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override;
    void evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes) override;
    void evaluationModelLogger(const int& n_epochs, Model<TensorT>& model, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const Eigen::Tensor<TensorT, 1>& model_metrics) override;
  };	
}
#endif //SMARTPEAK_MODELTRAINEREXPERIMENTAL_H