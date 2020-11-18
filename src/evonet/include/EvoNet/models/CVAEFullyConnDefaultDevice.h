/**TODO:  Add copyright*/

#ifndef EVONET_CVAEFULLYCONNDEFAULTDEVICE_H
#define EVONET_CVAEFULLYCONNDEFAULTDEVICE_H

// .h
#include <EvoNet/models/CVAEFullyConn.h>
#include <EvoNet/ml/ModelInterpreterDefaultDevice.h>

// .cpp
#include <EvoNet/io/ModelInterpreterFileDefaultDevice.h>
#include <EvoNet/io/ModelFile.h>

namespace EvoNet
{
	/**
		@brief TODO
	*/
	template<typename TensorT>
	class CVAEFullyConnDefaultDevice : public CVAEFullyConn<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	public:
    CVAEFullyConnDefaultDevice() = default; ///< Default constructor
		~CVAEFullyConnDefaultDevice() = default; ///< Default destructor    
                                                        
    /// Overrides used in all examples
    void adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors) override;
    void trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override;
    void validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) override;
	};
  template<typename TensorT>
  inline void CVAEFullyConnDefaultDevice<TensorT>::adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors)
  {
    // Check point the model every 1000 epochs
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      model_interpreter.getModelResults(model, false, true, false, false);
      //// save the model weights (Not needed if binary is working fine)
      //WeightFile<float> weight_data;
      //weight_data.storeWeightValuesCsv(model.getName() + "_" + std::to_string(n_epochs) + "_weights.csv", model.weights_);

      // save the model and tensors to binary
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }

    // copy the loss function helpers
    auto lossFunctionHelpers = this->getLossFunctionHelpers();

    // Increase the KL divergence beta and capacity
    TensorT beta_c = this->beta_c_;
    TensorT beta_d = this->beta_d_;
    TensorT capacity_c = this->capacity_c_;
    TensorT capacity_d = this->capacity_d_;
    if (this->KL_divergence_warmup_) {
      TensorT scale_factor1 = (n_epochs - 100 > 0) ? n_epochs - 100 : 1;
      beta_c /= (2.5e4 / scale_factor1);
      if (beta_c > this->beta_c_) beta_c = this->beta_c_;
      beta_d /= (2.5e4 / scale_factor1);
      if (beta_d > this->beta_d_) beta_d = this->beta_d_;
      TensorT scale_factor2 = (n_epochs - 1.0e4 > 0) ? n_epochs - 1.0e4 : 1;
      capacity_c /= (1.5e4 / scale_factor2);
      if (capacity_c > this->capacity_c_) capacity_c = this->capacity_c_;
      capacity_d /= (1.5e4 * scale_factor2);
      if (capacity_d > this->capacity_d_) capacity_d = this->capacity_d_;
    }
    lossFunctionHelpers.at(1).loss_functions_.at(0) = std::make_shared<KLDivergenceMuLossOp<float>>(KLDivergenceMuLossOp<float>(1e-6, beta_c, capacity_c));
    lossFunctionHelpers.at(2).loss_functions_.at(0) = std::make_shared<KLDivergenceLogVarLossOp<float>>(KLDivergenceLogVarLossOp<float>(1e-6, beta_c, capacity_c));
    lossFunctionHelpers.at(3).loss_functions_.at(0) = std::make_shared<KLDivergenceCatLossOp<float>>(KLDivergenceCatLossOp<float>(1e-6, beta_d, capacity_d));
    lossFunctionHelpers.at(1).loss_function_grads_.at(0) = std::make_shared<KLDivergenceMuLossGradOp<float>>(KLDivergenceMuLossGradOp<float>(1e-6, beta_c, capacity_c));
    lossFunctionHelpers.at(2).loss_function_grads_.at(0) = std::make_shared<KLDivergenceLogVarLossGradOp<float>>(KLDivergenceLogVarLossGradOp<float>(1e-6, beta_c, capacity_c));
    lossFunctionHelpers.at(3).loss_function_grads_.at(0) = std::make_shared<KLDivergenceCatLossGradOp<float>>(KLDivergenceCatLossGradOp<float>(1e-6, beta_d, capacity_d));

    // Modulate the level of supervision
    if (this->getLossFunctionHelpers().size() >= 5) {
      std::random_device rd;
      std::uniform_int_distribution<int> distribution(1, 100);
      std::mt19937 engine(rd());
      int value = distribution(engine);
      TensorT supervision = 1.0;
      if (value > this->supervision_percent_) supervision = 0.0;
      if (this->supervision_warmup_) supervision = (n_epochs - 2.5e4 > 0) ? supervision : 1.0;
      lossFunctionHelpers.at(4).loss_functions_.at(0) = std::make_shared<CrossEntropyWithLogitsLossOp<float>>(CrossEntropyWithLogitsLossOp<float>(1e-6, supervision * this->classification_loss_weight_));
      lossFunctionHelpers.at(4).loss_function_grads_.at(0) = std::make_shared<CrossEntropyWithLogitsLossGradOp<float>>(CrossEntropyWithLogitsLossGradOp<float>(1e-6, supervision * this->classification_loss_weight_));
    }

    // Update the loss function helpers
    this->setLossFunctionHelpers(lossFunctionHelpers);
  }
  template <typename TensorT>
  inline void CVAEFullyConnDefaultDevice<TensorT>::trainingModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test) {
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
  inline void CVAEFullyConnDefaultDevice<TensorT>::validationModelLogger(const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 3>& expected_values, const std::vector<std::string>& output_nodes, const std::vector<std::string>& input_nodes, const TensorT& model_error_train, const TensorT& model_error_test, const Eigen::Tensor<TensorT, 1>& model_metrics_train, const Eigen::Tensor<TensorT, 1>& model_metrics_test)
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
#endif //EVONET_CVAEFULLYCONNDEFAULTDEVICE_H