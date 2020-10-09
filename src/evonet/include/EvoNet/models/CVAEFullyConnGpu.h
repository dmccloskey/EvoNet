/**TODO:  Add copyright*/

#ifndef EVONET_CVAEFULLYCONNGPU_H
#define EVONET_CVAEFULLYCONNGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <EvoNet/models/CVAEFullyConn.h>
#include <EvoNet/ml/ModelInterpreterGpu.h>

// .cpp
#include <EvoNet/io/ModelInterpreterFileGpu.h>
#include <EvoNet/io/ModelFile.h>

namespace EvoNet
{
  /**
    @brief TODO
  */
  template<typename TensorT>
  class CVAEFullyConnGpu : public CVAEFullyConn<TensorT, ModelInterpreterGpu<TensorT>>
  {
  public:
    CVAEFullyConnGpu() = default; ///< Default constructor
    ~CVAEFullyConnGpu() = default; ///< Default destructor    

    /// Overrides used in all examples
    void adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors) override;
    void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override;
    void validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override;
  };
  template<typename TensorT>
  inline void CVAEFullyConnGpu<TensorT>::adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors)
  {
    // Check point the model every 1000 epochs
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false, false);
      // save the model weights (Not needed if binary is working fine)
      //WeightFile<float> weight_data;
      //weight_data.storeWeightValuesCsv(model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model.weights_);

      // save the model and tensors to binary
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);

      // Increase the KL divergence beta and capacity
      TensorT beta = this->beta_;
      TensorT capacity_c = this->capacity_c_;
      TensorT capacity_d = this->capacity_d_;
      if (this->KL_divergence_warmup_) {
        TensorT scale_factor1 = (n_epochs - 100 > 0) ? n_epochs - 100 : 1;
        beta /= 2.5e4 * scale_factor1;
        if (beta > this->beta_) beta = this->beta_;
        TensorT scale_factor2 = (n_epochs - 1.0e4 > 0) ? n_epochs - 1.0e4 : 1;
        capacity_c /= 1.5e4 * scale_factor2;
        if (capacity_c > this->capacity_c_) capacity_c = this->capacity_c_;
        capacity_d /= 1.5e4 * scale_factor2;
        if (capacity_d > this->capacity_d_) capacity_d = this->capacity_d_;
      }
      this->getLossFunctionHelpers().at(1).loss_functions_.at(0) = std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(2).loss_functions_.at(0) = std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(3).loss_functions_.at(0) = std::make_shared<KLDivergenceCatLossOp<float>>(KLDivergenceCatLossOp<float>(1e-6, beta, capacity_d));
      this->getLossFunctionHelpers().at(1).loss_function_grads_.at(0) = std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(2).loss_function_grads_.at(0) = std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, beta, capacity_c));
      this->getLossFunctionHelpers().at(3).loss_function_grads_.at(0) = std::make_shared<KLDivergenceCatLossGradOp<float>>(KLDivergenceCatLossGradOp<float>(1e-6, beta, capacity_d));
    }
  }
  template <typename TensorT>
  inline void CVAEFullyConnGpu<TensorT>::trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) {
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
    if (n_epochs % 1000 == 0) { // FIXME
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
  template<typename TensorT>
  inline void CVAEFullyConnGpu<TensorT>::validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test)
  {
    // Per n epoch logging
    model_logger.setLogTimeEpoch(true);
    model_logger.setLogTrainValMetricEpoch(true);
    model_logger.setLogExpectedEpoch(true);
    model_logger.setLogNodeInputsEpoch(true);
    model_logger.setLogNodeOutputsEpoch(true);
    model_interpreter.getModelResults(model, true, false, false, true);

    // initialize all logs
    if (n_epochs == 0) {
      model_logger.initLogs(model);
    }

    // Create the metric headers and data arrays
    std::vector<std::string> log_train_headers = { "Test_Error" };
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
}
#endif
#endif //EVONET_CVAEFULLYCONNGPU_H