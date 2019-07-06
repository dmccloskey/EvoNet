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

    Where classification accuracy = (TP + TN)/(TP + TN + FP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class ClassificationAccuracyTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    MetricFunctionTensorOp() = default;
    MetricFunctionTensorOp(const TensorT& classification_threshold_) = default;
		std::string getName() { return "ClassificationAccuracyTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);

      // Calculate the soft max
      auto predicted_chip = predicted_tensor.chip(time_step, 1); // 4 dims
      auto exps = (predicted_chip.chip(0, 3) - predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 3>({ 1, layer_size, 1 }))).exp(); // 3 dims
      auto stable_softmax = exps.chip(0, 2) / exps.sum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));  // 2 dims

      // calculate the true positives and true negatives
      auto tp = ()

      // calculate the true negatives rate
			error_tensor.chip(time_step, 0).device(device) += (((expected_tensor - predicted_chip).pow((TensorT)2).sqrt()).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
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
			error_tensor.chip(time_step, 0).device(device) += (((expected_tensor - (predicted_chip).pow((TensorT)2)) * expected_tensor.constant((TensorT)0.5)).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9)); // modified to simplify the derivative
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
			
			error_tensor.chip(time_step, 0).device(device) += ((-(
				expected_tensor * (predicted_chip + expected_tensor.constant(this->eps_)).log() + // check if .clip((TensorT)1e-6,(TensorT)1) should be used instead
				(expected_tensor.constant((TensorT)1) - expected_tensor) * (expected_tensor.constant((TensorT)1) - (predicted_chip - expected_tensor.constant(this->eps_))).log())).sum(Eigen::array<int, 1>({ 1 }))
				* error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
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
		void setN(const TensorT& n) { n_ = n; }
		std::string getName() { return "AUROCTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> error_tensor(error, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);

			error_tensor.chip(time_step, 0).device(device) += ((-expected_tensor * (predicted_chip.clip((TensorT)1e-6,(TensorT)1).log())) * expected_tensor.constant((TensorT)1 / (TensorT)layer_size)).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_);
		};
	private:
		TensorT n_ = (TensorT)1; ///< the number of total classifiers
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
			
			error_tensor.chip(time_step, 0).device(device) += ((-expected_tensor.constant((TensorT)0.5) + expected_tensor.constant((TensorT)0.5)*predicted_chip.pow((TensorT)2)).sum(Eigen::array<int, 1>({ 1 }))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
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

      error_tensor.chip(time_step, 0).device(device) += (((expected_tensor - predicted_chip).pow((TensorT)2) * expected_tensor.constant((TensorT)0.5) / expected_tensor.constant((TensorT)layer_size)).sum(Eigen::array<int, 1>({ 1 }))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9), TensorT(1e9));
    };
  };
}
#endif //SMARTPEAK_METRICFUNCTIONTENSOR_H