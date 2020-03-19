/**TODO:  Add copyright*/

#ifndef SMARTPEAK_METRICFUNCTIONTENSOR_H
#define SMARTPEAK_METRICFUNCTIONTENSOR_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

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
    MetricFunctionTensorOp(std::string& reduction_func) : reduction_func_(reduction_func) {}; ///< Options are Sum, Mean, Var
		virtual ~MetricFunctionTensorOp() = default;
		virtual std::string getName() = 0;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const = 0;
    void setReductionFunc(std::string& reduction_func) { reduction_func_ = reduction_func; }
    std::string getReductionFunc() { return reduction_func_; }
  protected:
    TensorT threshold_positive_ = 0.9;
    TensorT threshold_negative_ = 0.1;
    std::string reduction_func_ = "Sum";
	};

  /**
    @brief Accuracy metric function for binary classification.
       The class returns the average classification accuracy across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification accuracy = (TP + TN)/(TP + TN + FP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class AccuracyBCTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    AccuracyBCTensorOp() = default;
    AccuracyBCTensorOp(const TensorT& classification_threshold) : classification_threshold_(classification_threshold) {};
		std::string getName() override { return "AccuracyBCTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // calculate the confusion matrix
      auto predicted_chip = predicted_tensor.chip(time_step, 1); 
      auto tp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto tn = (predicted_chip < expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fn = (predicted_chip < expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      // calculate the accuracy
      auto accuracy = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum());
      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += accuracy;
		};
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief Accuracy metric function for multiclass classification.
       The class returns the micro average classification accuracy across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification accuracy = (TP + TN)/(TP + TN + FP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class AccuracyMCMicroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() override { return "AccuracyMCMicroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size*layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      // find the maximum value for each batch
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> max_tensor(tmp_data, batch_size, layer_size);
      max_tensor.device(device) = predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));

      // calculate the confusion matrix
      auto tp = (predicted_chip.chip(0, 2) >= (max_tensor - max_tensor.constant(TensorT(1e-6))) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto tn = (predicted_chip.chip(0, 2) < max_tensor && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fp = (predicted_chip.chip(0, 2) >= (max_tensor - max_tensor.constant(TensorT(1e-6))) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fn = (predicted_chip.chip(0, 2) < max_tensor && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      // calculate the accuracy     
      auto accuracy = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum());
      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += accuracy;

      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif

      //// DEBUG
      //std::cout << "TP: " << tp << std::endl;
      //std::cout << "TN: " << tn << std::endl;
      //std::cout << "FP: " << fp << std::endl;
      //std::cout << "FN: " << fn << std::endl;
      //std::cout << "Accuracy: " << accuracy << std::endl;
    };
  };

  /**
    @brief Accuracy metric function for multiclass classification.
       The class returns the macro average classification accuracy across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification accuracy = (TP + TN)/(TP + TN + FP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class AccuracyMCMacroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() override { return "AccuracyMCMacroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // find the maximum value for each batch
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      // TODO... 
      // Sum on the per batch level and then average e.g. / accuracy.constant(TensorT(batch_size));
    };
  };

  /**
    @brief Precision metric function for binary classification.
       The class returns the average classification precision across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification precision = TP/(TP + FP)
  */
  template<typename TensorT, typename DeviceT>
  class PrecisionBCTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    PrecisionBCTensorOp() = default;
    PrecisionBCTensorOp(const TensorT& classification_threshold) : classification_threshold_(classification_threshold) {};
    std::string getName() override { return "PrecisionBCTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // calculate the confusion matrix
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto tp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      // calculate the precision     
      auto precision = tp.sum() / (tp.sum() + fp.sum());
      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += precision;
    };
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief Precision metric function for multiclass classification.
       The class returns the micro average classification precision across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification precision = TP/(TP + FP)
  */
  template<typename TensorT, typename DeviceT>
  class PrecisionMCMicroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() override { return "PrecisionMCMicroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // find the maximum value for each batch
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto max_tensor = predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));

      // calculate the confusion matrix
      auto tp = (predicted_chip.chip(0, 2) >= (max_tensor - max_tensor.constant(TensorT(1e-6))) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fp = (predicted_chip.chip(0, 2) >= (max_tensor - max_tensor.constant(TensorT(1e-6))) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      // calculate the precision     
      auto precision = tp.sum() / (tp.sum() + fp.sum());
      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += precision;

      //// DEBUG
      //std::cout << "predicted_chip.chip(0, 1): " << predicted_chip << std::endl;
      //std::cout << "max_tensor: " << max_tensor << std::endl;
      //std::cout << "expected_tensor: " << expected_tensor << std::endl;
      //std::cout << "TP: " << tp << std::endl;
      //std::cout << "FP: " << fp << std::endl;
      //std::cout << "precision: " << precision << std::endl;
    };
  };

  /**
    @brief Precision metric function for multiclass classification.
       The class returns the macro average classification precision across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification precision = TP/(TP + FP)
  */
  template<typename TensorT, typename DeviceT>
  class PrecisionMCMacroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() override { return "PrecisionMCMacroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // find the maximum value for each batch
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      // TODO... 
      // Sum on the per batch level and then average e.g. / precision.constant(TensorT(batch_size));
    };
  };

  /**
    @brief Recall metric function for binary classification.
       The class returns the average classification recall across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification recall = TP /(TP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class RecallBCTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    RecallBCTensorOp() = default;
    RecallBCTensorOp(const TensorT& classification_threshold) : classification_threshold_(classification_threshold) {};
    std::string getName() override { return "RecallBCTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // calculate the confusion matrix
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto tp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fn = (predicted_chip < expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      // calculate the recall     
      auto recall = tp.sum() / (tp.sum() + fn.sum());
      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += recall;
    };
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief Recall metric function for multiclass classification.
       The class returns the micro average classification recall across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification recall = TP /(TP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class RecallMCMicroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() override { return "RecallMCMicroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // find the maximum value for each batch
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto max_tensor = predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));

      // calculate the confusion matrix
      auto tp = (predicted_chip.chip(0, 2) >= (max_tensor - max_tensor.constant(TensorT(1e-6))) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      auto fn = (predicted_chip.chip(0, 2) < max_tensor && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      // calculate the recall     
      auto recall = tp.sum() / (tp.sum() + fn.sum());
      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += recall;
    };
  };

  /**
    @brief Recall metric function for multiclass classification.
       The class returns the macro average classification recall across all batches
       where an expected true value > 0.9 and an expected false value < 0.9

    Where classification recall = TP /(TP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class RecallMCMacroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() override { return "RecallMCMacroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      // find the maximum value for each batch
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      // TODO... 
      // Sum on the per batch level and then average e.g. / recall.constant(TensorT(batch_size));
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
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
      // TODO...
		};
  };

  /**
    @brief F1 score metric function for binary classification.
      The class returns the average F1 score across all batches

    Where F1 score = 2*precision*recall/(precision + recall)
	    and precision = TP/(TP + FP)
	    and recall = TP/(TP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class F1ScoreBCTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    F1ScoreBCTensorOp() = default;
    F1ScoreBCTensorOp(const TensorT& classification_threshold) : classification_threshold_(classification_threshold) {};
		std::string getName() { return "F1ScoreBCTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      //// calculate the confusion matrix
      //auto predicted_chip = predicted_tensor.chip(time_step, 1);
      //auto tp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      //auto fp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      //auto fn = (predicted_chip < expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      //// calculate the F1 score
      //auto precision = tp.sum()/(tp.sum() + fp.sum());
      //auto recall = tp.sum() / (tp.sum() + fn.sum());
      //auto f1score = precision.constant(TensorT(2))*precision*recall / (precision + recall);
      //error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += f1score;
		};
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief F1 score metric function for multiclass classification.
      The class returns the micro average F1 score across all batches

    Where F1 score = 2*precision*recall/(precision + recall)
      and precision = TP/(TP + FP)
      and recall = TP/(TP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class F1ScoreMCMicroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() { return "F1ScoreMCMicroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

//      // allocate temporary memory
//      TensorT* tmp_data;
//      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
//        tmp_data = new TensorT[batch_size*layer_size];
//      }
//#if COMPILE_WITH_CUDA
//      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
//        size_t bytes = batch_size * layer_size * sizeof(TensorT);
//        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
//      }
//#endif
//      // find the maximum value for each batch
//      auto predicted_chip = predicted_tensor.chip(time_step, 1);
//      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> max_tensor(tmp_data, batch_size, layer_size);
//      max_tensor = predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));
//
//      // calculate the confusion matrix
//      auto tp = (predicted_chip.chip(0, 2) >= (max_tensor - max_tensor.constant(TensorT(1e-6))) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
//      auto fp = (predicted_chip.chip(0, 2) >= (max_tensor - max_tensor.constant(TensorT(1e-6))) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
//      auto fn = (predicted_chip.chip(0, 2) < max_tensor && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
//
//      // calculate the F1 score
//      auto precision = tp.sum() / (tp.sum() + fp.sum());
//      auto recall = tp.sum() / (tp.sum() + fn.sum());
//      auto f1score = precision.constant(TensorT(2))*precision*recall / (precision + recall);
//      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += f1score;
//
//      // deallocate temporary memory
//      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
//        delete[] tmp_data;
//      }
//#if COMPILE_WITH_CUDA
//      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
//        assert(cudaFree(tmp_data) == cudaSuccess);
//      }
//#endif

      //// DEBUG
      //std::cout << "precision: " << precision << std::endl;
      //std::cout << "recall: " << recall << std::endl;
      //std::cout << "f1score: " << f1score << std::endl;
    };
  };

  /**
    @brief F1 score metric function for multiclass classification.
      The class returns the macro average F1 score across all batches

    Where F1 score = 2*precision*recall/(precision + recall)
      and precision = TP/(TP + FP)
      and recall = TP/(TP + FN)
  */
  template<typename TensorT, typename DeviceT>
  class F1ScoreMCMacroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() { return "F1ScoreMCMacroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);

      //// calculate the confusion matrix
      //auto predicted_chip = predicted_tensor.chip(time_step, 1);
      //auto tp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      //auto fp = (predicted_chip >= expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor < expected_tensor.constant(TensorT(this->threshold_negative_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));
      //auto fn = (predicted_chip < expected_tensor.constant(TensorT(this->classification_threshold_)) && expected_tensor > expected_tensor.constant(TensorT(this->threshold_positive_))).select(expected_tensor.constant(TensorT(1)), expected_tensor.constant(TensorT(0)));

      //// calculate the F1 score [TODO: update as this is not correct...]
      //auto precision = tp.sum() / (tp.sum() + fp.sum());
      //auto recall = tp.sum() / (tp.sum() + fn.sum());
      //auto f1score = precision.constant(TensorT(2))*precision*recall / (precision + recall);
      //error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += f1score;
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
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
		};
  };

	/**
		@brief Matthews correlation coefficient (binary 2 class problems) metric function.
      The class retuns the average metthews correlation coefficient across all batches

    Where MCC = TP*TN-FP*FN/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
	*/
	template<typename TensorT, typename DeviceT>
	class MCCBCTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
		std::string getName() { return "MCCBCTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
      // TODO...
		};
	};

  /**
    @brief Matthews correlation coefficient metric function for multiclass classification.
      The class retuns the micro average metthews correlation coefficient across all batches

    Where MCC = TP*TN-FP*FN/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  */
  template<typename TensorT, typename DeviceT>
  class MCCMCMicroTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using MetricFunctionTensorOp<TensorT, DeviceT>::MetricFunctionTensorOp;
    std::string getName() { return "MCCMCMicroTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      // TODO...
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
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);

      error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += ((expected_tensor - predicted_chip).pow(TensorT(2)).pow(TensorT(0.5)) / expected_tensor.constant(TensorT(layer_size) * TensorT(batch_size))).sum();
    };
  };

  /**
    @brief CosineSimilarity metric function.

    Where CosineSimilarity = A*B/(||A||*||B||)

    Note: need to divide by the batch size if the average value over all batches is needed
  */
  template<typename TensorT, typename DeviceT>
  class CosineSimilarityTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    CosineSimilarityTensorOp() = default;
    CosineSimilarityTensorOp(std::string& reduction_func) : MetricFunctionTensorOp(reduction_func) {};
    std::string getName() { return "CosineSimilarityTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto dot_prod = (predicted_chip * expected_tensor).sum(Eigen::array<Eigen::Index, 1>({1})); // dim 1 batch_size
      auto predicted_unit = (predicted_chip.pow(TensorT(2)).sum(Eigen::array<Eigen::Index, 1>({ 1 }))).pow(TensorT(0.5)); // dim 1 batch_size
      auto expected_unit = (expected_tensor.pow(TensorT(2)).sum(Eigen::array<Eigen::Index, 1>({ 1 }))).pow(TensorT(0.5)); // dim 1 batch_size

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * 1];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * 1 * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> cosine_similarity(tmp_data, batch_size, 1);
      cosine_similarity.device(device) = dot_prod / (predicted_unit * expected_unit);
      if (this->reduction_func_ == "Sum")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += cosine_similarity.sum();
      else if (this->reduction_func_ == "Mean")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += (cosine_similarity / cosine_similarity.constant(TensorT(batch_size))).sum();
      else if (this->reduction_func_ == "Var") {
        auto mean = (cosine_similarity / cosine_similarity.constant(TensorT(batch_size))).sum(Eigen::array<int, 1>({ 0 })).broadcast(Eigen::array<int, 1>({ batch_size }));
        auto var = ((mean - cosine_similarity.chip(0, 1)).pow(TensorT(2)) / mean.constant(TensorT(batch_size) - 1)).sum();
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += var;
      }

      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    };
  };

  /**
    @brief PearsonR metric function.

    Where PearsonR = Rxy = Sum(i=1 to n)[(xi-xhat)(yi-yhat)]/(sqrt(Sum(i=1 to n)[(xi-xhat)^2]) * sqrt(Sum(i=1 to n)[(yi-yhat)^2]))

    Note: need to divide by the batch size if the average value over all batches is needed
  */
  template<typename TensorT, typename DeviceT>
  class PearsonRTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    PearsonRTensorOp() = default;
    PearsonRTensorOp(std::string& reduction_func) : MetricFunctionTensorOp(reduction_func) {};
    std::string getName() { return "PearsonRTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> expected_tensor(expected, batch_size, layer_size, 1, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto cov = ((predicted_chip.chip(0, 2) - predicted_chip.mean(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, layer_size, 1 }))) *
        (expected_tensor.chip(0, 2) - expected_tensor.mean(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, layer_size, 1 })))
        ).sum(Eigen::array<Eigen::Index, 1>({ 1 })); // Dim 1 batch_size
      auto predicted_stdev = ((predicted_chip.chip(0, 2) - predicted_chip.mean(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, layer_size, 1 }))
        ).pow(TensorT(2)).sum(Eigen::array<Eigen::Index, 1>({ 1 })).pow(TensorT(0.5))); // Dim 1 batch_size
      auto expected_stdev = ((expected_tensor.chip(0, 2) - expected_tensor.mean(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, layer_size, 1 }))
        ).pow(TensorT(2)).sum(Eigen::array<Eigen::Index, 1>({ 1 })).pow(TensorT(0.5))); // Dim 1 batch_size

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * 1];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * 1 * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> PearsonR(tmp_data, batch_size, 1);
      PearsonR.device(device) = cov / (predicted_stdev * expected_stdev);
      if (this->reduction_func_ == "Sum")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += PearsonR.sum();
      else if (this->reduction_func_ == "Mean")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += (PearsonR / PearsonR.constant(TensorT(batch_size))).sum();
      else if (this->reduction_func_ == "Var") {
        auto mean = (PearsonR / PearsonR.constant(TensorT(batch_size))).sum(Eigen::array<int, 1>({ 0 })).broadcast(Eigen::array<int, 1>({ batch_size }));
        auto var = ((mean - PearsonR.chip(0, 1)).pow(TensorT(2)) / mean.constant(TensorT(batch_size) - 1)).sum();
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += var;
      }

      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    };
  };

  /**
    @brief EuclideanDist metric function.

    NOTE: useful for data in the range of (-inf, inf)
  */
  template<typename TensorT, typename DeviceT>
  class EuclideanDistTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    EuclideanDistTensorOp() = default;
    EuclideanDistTensorOp(std::string& reduction_func) : MetricFunctionTensorOp(reduction_func) {};
    std::string getName() { return "EuclideanDistTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * 1];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * 1 * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> euclidean_dist(tmp_data, batch_size, 1);
      euclidean_dist.device(device) = ((expected_tensor - predicted_chip).pow(TensorT(2))).sum(Eigen::array<int, 1>({ 1 })).sqrt();
      if (this->reduction_func_ == "Sum")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += euclidean_dist.sum();
      else if (this->reduction_func_ == "Mean")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum();
      else if (this->reduction_func_ == "Var") {
        auto mean = (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum(Eigen::array<int, 1>({ 0 })).broadcast(Eigen::array<int, 1>({ batch_size }));
        auto var = ((mean - euclidean_dist.chip(0, 1)).pow(TensorT(2)) / mean.constant(TensorT(batch_size) - 1)).sum();
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += var;
      }

      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    };
  };

  /**
    @brief ManhattanDist metric function.

    NOTE: useful for data in the range of (-inf, inf)
  */
  template<typename TensorT, typename DeviceT>
  class ManhattanDistTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    ManhattanDistTensorOp() = default;
    ManhattanDistTensorOp(std::string& reduction_func) : MetricFunctionTensorOp(reduction_func) {};
    std::string getName() { return "ManhattanDistTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * 1];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * 1 * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> euclidean_dist(tmp_data, batch_size, 1);
      euclidean_dist.device(device) = ((expected_tensor - predicted_chip).pow(TensorT(2)).sqrt()).sum(Eigen::array<int, 1>({ 1 }));
      if (this->reduction_func_ == "Sum")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += euclidean_dist.sum();
      else if (this->reduction_func_ == "Mean")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum();
      else if (this->reduction_func_ == "Var") {
        auto mean = (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum(Eigen::array<int, 1>({ 0 })).broadcast(Eigen::array<int, 1>({ batch_size }));
        auto var = ((mean - euclidean_dist.chip(0, 1)).pow(TensorT(2)) / mean.constant(TensorT(batch_size) - 1)).sum();
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += var;
      }

      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    };
  };

  /**
    @brief JeffreysAndMatusitaDist metric function.

    NOTE: only useful for data in the range of [0, inf)
  */
  template<typename TensorT, typename DeviceT>
  class JeffreysAndMatusitaDistTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    JeffreysAndMatusitaDistTensorOp() = default;
    JeffreysAndMatusitaDistTensorOp(std::string& reduction_func) : MetricFunctionTensorOp(reduction_func) {};
    std::string getName() { return "JeffreysAndMatusitaDistTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * 1];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * 1 * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> euclidean_dist(tmp_data, batch_size, 1);
      euclidean_dist.device(device) = ((expected_tensor.cwiseMax(expected_tensor.constant(TensorT(0))).sqrt() -
        predicted_chip.cwiseMax(predicted_chip.constant(TensorT(0))).sqrt()).pow(TensorT(2))).sum(Eigen::array<int, 1>({ 1 })).sqrt();
      if (this->reduction_func_ == "Sum")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += euclidean_dist.sum();
      else if (this->reduction_func_ == "Mean")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum();
      else if (this->reduction_func_ == "Var") {
        auto mean = (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum(Eigen::array<int, 1>({ 0 })).broadcast(Eigen::array<int, 1>({ batch_size }));
        auto var = ((mean - euclidean_dist.chip(0, 1)).pow(TensorT(2)) / mean.constant(TensorT(batch_size) - 1)).sum();
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += var;
      }

      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    };
  };

  /**
    @brief Logarithmic Distance metric function.

    NOTE: only useful for data in the range of [0, inf)
  */
  template<typename TensorT, typename DeviceT>
  class LogarithmicDistTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    LogarithmicDistTensorOp() = default;
    LogarithmicDistTensorOp(std::string& reduction_func) : MetricFunctionTensorOp(reduction_func) {};
    std::string getName() { return "LogarithmicDistTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> expected_tensor(expected, batch_size, layer_size, 1, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto diff = expected_tensor - predicted_chip;
      auto min_offset = diff.chip(0, 2) - diff.minimum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, layer_size, 1 })) + diff.chip(0, 2).constant(TensorT(1));

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * 1];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * 1 * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> euclidean_dist(tmp_data, batch_size, 1);
      euclidean_dist.device(device) = min_offset.log().sum(Eigen::array<int, 1>({ 1 }));
      if (this->reduction_func_ == "Sum")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += euclidean_dist.sum();
      else if (this->reduction_func_ == "Mean")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum();
      else if (this->reduction_func_ == "Var") {
        auto mean = (euclidean_dist / euclidean_dist.constant(TensorT(batch_size))).sum(Eigen::array<int, 1>({ 0 })).broadcast(Eigen::array<int, 1>({ batch_size }));
        auto var = ((mean - euclidean_dist.chip(0, 1)).pow(TensorT(2)) / mean.constant(TensorT(batch_size) - 1)).sum();
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += var;
      }
      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    };
  };


  /**
    @brief PercentDifference metric function.

    NOTE: useful for data in the range of (-inf, inf)
  */
  template<typename TensorT, typename DeviceT>
  class PercentDifferenceTensorOp : public MetricFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    PercentDifferenceTensorOp() = default;
    PercentDifferenceTensorOp(std::string& reduction_func) : MetricFunctionTensorOp(reduction_func) {};
    std::string getName() { return "PercentDifferenceTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, n_metrics, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto perc_diff_selected = (expected_tensor == expected_tensor.constant(TensorT(0))).select(expected_tensor.constant(TensorT(0)), ((expected_tensor - predicted_chip) / expected_tensor).pow(TensorT(2)).sqrt() );

      // allocate temporary memory
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * 1];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * 1 * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> perce_diff(tmp_data, batch_size, 1);
      perce_diff.device(device) = perc_diff_selected.sum(Eigen::array<int, 1>({ 1 }));
      if (this->reduction_func_ == "Sum")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += perce_diff.sum();
      else if (this->reduction_func_ == "Mean")
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += (perce_diff / perce_diff.constant(TensorT(batch_size))).sum();
      else if (this->reduction_func_ == "Var") {
        auto mean = (perce_diff / perce_diff.constant(TensorT(batch_size))).sum(Eigen::array<int, 1>({ 0 })).broadcast(Eigen::array<int, 1>({ batch_size }));
        auto var = ((mean - perce_diff.chip(0, 1)).pow(TensorT(2)) / mean.constant(TensorT(batch_size) - 1)).sum();
        error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += var;
      }

      // deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    };
  };
}
#endif //SMARTPEAK_METRICFUNCTIONTENSOR_H