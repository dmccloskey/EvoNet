/**TODO:  Add copyright*/

#ifndef SMARTPEAK_METRICFUNCTION_H
#define SMARTPEAK_METRICFUNCTION_H

namespace SmartPeak
{
	/**
	@brief Base class for all model metric functions

  Abbreviations used in classes:
    - BC: binary classification
    - MC: multiclass classification
    - ML: multilabel classification
    - H: hierarchical classification
    - micro: micro averaging
    - macro: macro averaging
	*/
	template<typename TensorT>
	class MetricFunctionOp
	{
	public:
		MetricFunctionOp() = default;
		virtual ~MetricFunctionOp() = default;
		virtual std::string getName() = 0;
		virtual std::vector<TensorT> getParameters() const = 0;
    std::string getReductionFunc() { return reduction_func_; }
    void setReductionFunc(const std::string& reduction_func) { reduction_func_ = reduction_func; }
  protected:
    std::string reduction_func_ = "Sum";
	};

  /**
    @brief Classification accuracy function for binary classification problems.
  */
  template<typename TensorT>
  class AccuracyBCOp : public MetricFunctionOp<TensorT>
  {
  public: 
    AccuracyBCOp() = default;
    AccuracyBCOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {}
		std::string getName() {	return "AccuracyBCOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->classification_threshold_}); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5; ///< greater than or equal to is true, less than is false
  };

  /**
    @brief Classification accuracy function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class AccuracyMCMicroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "AccuracyMCMicroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
  };

  /**
    @brief Classification accuracy function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class AccuracyMCMacroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "AccuracyMCMacroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
  };

  /**
    @brief Classification precision function for binary classification problems.
  */
  template<typename TensorT>
  class PrecisionBCOp : public MetricFunctionOp<TensorT>
  {
  public:
    PrecisionBCOp() = default;
    PrecisionBCOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {}
    std::string getName() { return "PrecisionBCOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->classification_threshold_ }); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5; ///< greater than or equal to is true, less than is false
  };

  /**
    @brief Classification precision function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class PrecisionMCMicroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "PrecisionMCMicroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
  };

  /**
    @brief Classification precision function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class PrecisionMCMacroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "PrecisionMCMacroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
  };

  /**
    @brief Classification recall function for binary classification problems.
  */
  template<typename TensorT>
  class RecallBCOp : public MetricFunctionOp<TensorT>
  {
  public:
    RecallBCOp() = default;
    RecallBCOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {}
    std::string getName() { return "RecallBCOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->classification_threshold_ }); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5; ///< greater than or equal to is true, less than is false
  };

  /**
    @brief Classification recall function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class RecallMCMicroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "RecallMCMicroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
  };

  /**
    @brief Classification recall function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class RecallMCMacroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "RecallMCMacroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
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
    @brief F1 score function for binary classification problems.
  */
  template<typename TensorT>
  class F1ScoreBCOp : public MetricFunctionOp<TensorT>
  {
  public:
    F1ScoreBCOp() = default;
    F1ScoreBCOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {};
    std::string getName() { return "F1ScoreBCOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->classification_threshold_ }); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief F1 score function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class F1ScoreMCMicroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "F1ScoreMCMicroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
  };

  /**
    @brief F1 score function for multiclass classification problems using macro averaging.
  */
  template<typename TensorT>
  class F1ScoreMCMacroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "F1ScoreMCMacroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
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
    @brief Mathews correlation coefficient (MCC) function for binary classification.
  */
  template<typename TensorT>
  class MCCBCOp : public MetricFunctionOp<TensorT>
  {
  public:
    MCCBCOp() = default;
    MCCBCOp(const TensorT& classification_threshold) :classification_threshold_(classification_threshold) {}
    std::string getName() { return "MCCBCOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->classification_threshold_ }); }
    TensorT getClassificationThreshold() const { return this->classification_threshold_; }
  protected:
    TensorT classification_threshold_ = 0.5;
  };

  /**
    @brief Mathews correlation coefficient (MCC) function for multiclass classification problems using micro averaging.
  */
  template<typename TensorT>
  class MCCMCMicroOp : public MetricFunctionOp<TensorT>
  {
  public:
    using MetricFunctionOp<TensorT>::MetricFunctionOp;
    std::string getName() { return "MCCMCMicroOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ }); }
  };

  /**
    @brief MAE Mean Absolute Error function.
  */
  template<typename TensorT>
  class MAEOp : public MetricFunctionOp<TensorT>
  {
  public:
    MAEOp() = default;
    ~MAEOp() = default;
    MAEOp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
		std::string getName() { return "MAEOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief CosineSimilarity function.
  */
  template<typename TensorT>
  class CosineSimilarityOp : public MetricFunctionOp<TensorT>
  {
  public:
    CosineSimilarityOp() = default;
    ~CosineSimilarityOp() = default;
    CosineSimilarityOp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
    std::string getName() { return "CosineSimilarityOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief PearsonR function.
  */
  template<typename TensorT>
  class PearsonROp : public MetricFunctionOp<TensorT>
  {
  public:
    PearsonROp() = default;
    ~PearsonROp() = default;
    PearsonROp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
    std::string getName() { return "PearsonROp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief EuclideanDist function.
  */
  template<typename TensorT>
  class EuclideanDistOp : public MetricFunctionOp<TensorT>
  {
  public:
    EuclideanDistOp() = default;
    ~EuclideanDistOp() = default;
    EuclideanDistOp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
    std::string getName() { return "EuclideanDistOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief ManhattanDist function.
  */
  template<typename TensorT>
  class ManhattanDistOp : public MetricFunctionOp<TensorT>
  {
  public:
    ManhattanDistOp() = default;
    ~ManhattanDistOp() = default;
    ManhattanDistOp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
    std::string getName() { return "ManhattanDistOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief JeffreysAndMatusitaDist function.
  */
  template<typename TensorT>
  class JeffreysAndMatusitaDistOp : public MetricFunctionOp<TensorT>
  {
  public:
    JeffreysAndMatusitaDistOp() = default;
    ~JeffreysAndMatusitaDistOp() = default;
    JeffreysAndMatusitaDistOp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
    std::string getName() { return "JeffreysAndMatusitaDistOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief LogarithmicDist function.
  */
  template<typename TensorT>
  class LogarithmicDistOp : public MetricFunctionOp<TensorT>
  {
  public:
    LogarithmicDistOp() = default;
    ~LogarithmicDistOp() = default;
    LogarithmicDistOp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
    std::string getName() { return "LogarithmicDistOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };

  /**
    @brief PercentDifference function.
  */
  template<typename TensorT>
  class PercentDifferenceOp : public MetricFunctionOp<TensorT>
  {
  public:
    PercentDifferenceOp() = default;
    ~PercentDifferenceOp() = default;
    PercentDifferenceOp(const std::string& reduction_func){ this->setReductionFunc(reduction_func); }; ///< Options are Sum, Mean, Var
    std::string getName() { return "PercentDifferenceOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>(); }
  };
}
#endif //SMARTPEAK_METRICFUNCTION_H