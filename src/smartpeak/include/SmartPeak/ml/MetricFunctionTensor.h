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
		virtual ~MetricFunctionTensorOp() = default;
		virtual std::string getName() = 0;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size,
      const int& n_metrics, const int& time_step, const int& metric_index, DeviceT& device) const = 0;
  protected:
    TensorT threshold_positive_ = 0.9;
    TensorT threshold_negative_ = 0.1;
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

      //// calculate the accuracy     
      //auto accuracy = TODO
      //error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += accuracy;
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

      //// calculate the precision     
      //auto precision = TODO
      //error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += precision;
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

      //// calculate the recall     
      //auto recall = TODO
      //error_tensor.chip(metric_index, 0).chip(time_step, 0).device(device) += recall;
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

      //// calculate the F1 score [TODO: update as this is not correct]
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
}
#endif //SMARTPEAK_METRICFUNCTIONTENSOR_H