/**TODO:  Add copyright*/

#ifndef SMARTPEAK_METRICFUNCTION_H
#define SMARTPEAK_METRICFUNCTION_H

namespace SmartPeak
{
	/**
	@brief Base class for all model metric functions
	*/
	template<typename TensorT>
	class MetricFunctionOp
	{
	public:
		MetricFunctionOp() = default;
		virtual ~MetricFunctionOp() = default;
		virtual std::string getName() = 0;
		virtual std::vector<TensorT> getParameters() const = 0;
	};

  /**
    @brief Classification accuracy function.
  */
  template<typename TensorT>
  class ClassificationAccuracyOp : public MetricFunctionOp<TensorT>
  {
  public: 
    ClassificationAccuracyOp() = default;
    ClassificationAccuracyOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {}
		std::string getName() {	return "ClassificationAccuracyOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->classification_threshold_}); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief Prediction biass function.
  */
  template<typename TensorT>
  class PredictionBiasOp : public MetricFunctionOp<TensorT>
  {
  public:
		using MetricFunctionOp<TensorT>::MetricFunctionOp;
		std::string getName() { return "PredictionBiasOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief F1 score function.
  */
  template<typename TensorT>
  class F1ScoreOp : public MetricFunctionOp<TensorT>
  {
  public:
    F1ScoreOp() = default;
    F1ScoreOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {};
    std::string getName() { return "F1ScoreOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->classification_threshold_ }); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief Area under the ROC (AUROC) function.
  */
  template<typename TensorT>
  class AUROCOp : public MetricFunctionOp<TensorT>
  {
  public:
    AUROCOp() = default;
    AUROCOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {}
    std::string getName() { return "AUROCOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->classification_threshold_ }); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief Mathews correlation coefficient (MCC) function.
  */
  template<typename TensorT>
  class MCCOp : public MetricFunctionOp<TensorT>
  {
  public:
    MCCOp() = default;
    MCCOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {}
    std::string getName() { return "MCCOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->classification_threshold_ }); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief MAE Mean Absolute Error function.
  */
  template<typename TensorT>
  class MAEOp : public MetricFunctionOp<TensorT>
  {
  public:
		using MetricFunctionOp<TensorT>::MetricFunctionOp;
		std::string getName() { return "MAEOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };
}
#endif //SMARTPEAK_METRICFUNCTION_H