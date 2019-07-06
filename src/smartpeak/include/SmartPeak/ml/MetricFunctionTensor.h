/**TODO:  Add copyright*/

#ifndef SMARTPEAK_METRICFUNCTIONTENSOR_H
#define SMARTPEAK_METRICFUNCTIONTENSOR_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
	/**
	@brief Base class for all model metric functions.

  NOTE: Unlike LossFunctions that return the results on a per batch basis,
    model metric functions return a single value across all batch results
	*/
	template<typename TensorT, typename DeviceT>
	class MetricFunctionTensorOp
	{
	public:
		MetricFunctionTensorOp() = default;
		virtual ~MetricFunctionTensorOp() = default;
		virtual std::string getName() = 0;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
	};

  /**
    @brief ClassificationAccuracy metric function.
       The class returns the average classification accuracy across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification accuracy = (TP + TN)/(TP + TN + FP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class ClassificationAccuracyTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    ClassificationAccuracyTensorOp() = default;
    ClassificationAccuracyTensorOp(const TensorT& classification_threshold) : classification_threshold_(classification_threshold) {};
		std::string getName() override { return "ClassificationAccuracyTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);

      // Calculate the soft max
      auto predicted_chip = predicted_tensor.chip(time_step, 1); // 4 dims
      auto exps = (predicted_chip.chip(0, 3) - predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 3>({ 1, layer_size, 1 }))).exp(); // 3 dims
      auto stable_softmax = exps.chip(0, 2) / exps.sum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));  // 2 dims

      // calculate the confusion matrix
      auto tp = (stable_softmax >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(0.9))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto tn = (stable_softmax < expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor < expected_tensor.constant(TensorT(0.1))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fp = (stable_softmax >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor < expected_tensor.constant(TensorT(0.1))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fn = (stable_softmax < expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(0.9))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      //// DEBUG
      //std::cout << "Stable softmax: " << stable_softmax << std::endl;
      //std::cout << "Expected: " << expected_tensor << std::endl;
      //std::cout << "TP: " << tp << std::endl;
      //std::cout << "TN: " << tn << std::endl;
      //std::cout << "FP: " << fp << std::endl;
      //std::cout << "FN: " << fn << std::endl;

      // calculate the accuracy     
      auto accuracy = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum());
      error_tensor.chip(time_step, 0).device(device) += accuracy; //Not needed: / accuracy.constant(TensorT(batch_size));
		};
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief PredictionBias metric function.

    Where Prediction bias = average of predictions - average of labels in the data set
  */
  template<typename TensorT, typename DeviceT>
  class PredictionBiasTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
		using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
		std::string getName() { return "PredictionBiasTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
		};
  };

  /**
    @brief F1 score metric function.
      The class returns the average F1 score across all batches

    Where F1 score = 2*precision*recall/(precision + recall)
	    and precision = TP/(TP + FP)
	    and recall = TP/(TP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class F1ScoreTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
		std::string getName() { return "F1ScoreTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
		};
  };

  /**
    @brief AUROC metric function.

    Where ROC point per batch = sensitivity/FPR
	    and sensitivity = recall = TP/(TP + FN)
	    and FPR = FP/(FP + TN)
    And AUROC = area under the curve of sensitivity vs. FPR
  */
  template<typename TensorT, typename DeviceT>
  class AUROCTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
		std::string getName() { return "AUROCTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
		};
  };

	/**
		@brief Matthews correlation coefficient (binary 2 class problems) metric function.
      The class retuns the average metthews correlation coefficient across all batches

    Where MCC = TP*TN-FP*FN/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
	*/
	template<typename TensorT, typename DeviceT>
	class MCCOp : public MetricFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
		std::string getName() { return "MCCOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
		};
	};

  /**
    @brief MAE Mean Absolute Error metric function.

    Where MAE = 1/N * Sum[ abs(xi-xhat) ]
  */
  template<typename TensorT, typename DeviceT>
  class MAETensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() { return "MAETensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);

      error_tensor.chip(time_step, 0).device(device) += ((expected_tensor - predicted_chip).pow(TensorT(2)).pow(TensorT(0.5)) / expected_tensor.constant(TensorT(layer_size) * TensorT(batch_size))).sum();
    };
  };
}
#endif //SMARTPEAK_METRICFUNCTIONTENSOR_H