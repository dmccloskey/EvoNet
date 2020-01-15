/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LOSSFUNCTION_H
#define SMARTPEAK_LOSSFUNCTION_H

#include <SmartPeak/core/Preprocessing.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>

namespace SmartPeak
{
	/**
	@brief Base class loss function.
	*/
	template<typename TensorT>
	class LossFunctionOp
	{
	public:
		LossFunctionOp() = default;
		LossFunctionOp(const TensorT& eps, const TensorT& scale) : eps_(eps), scale_(scale) {};
		~LossFunctionOp() = default;
		virtual std::string getName() = 0;
		virtual std::vector<TensorT> getParameters() const = 0;
	protected:
		TensorT eps_ = TensorT(1e-6);
		TensorT scale_ = TensorT(1);
	};

	/**
	@brief Base class loss function gradient.
	*/
	template<typename TensorT>
	class LossFunctionGradOp
	{
	public:
		LossFunctionGradOp() = default;
		LossFunctionGradOp(const TensorT& eps, const TensorT& scale) : eps_(eps), scale_(scale) {};
		~LossFunctionGradOp() = default;
		virtual std::string getName() = 0;
		virtual std::vector<TensorT> getParameters() const = 0;
	protected:
		TensorT eps_ = (TensorT)1e-6;
		TensorT scale_ = (TensorT)1.0;
	};

  /**
    @brief ManhattanDistance loss function.
  */
  template<typename TensorT>
  class ManhattanDistanceLossOp : public LossFunctionOp<TensorT>
  {
public: 
		using LossFunctionOp<TensorT>::LossFunctionOp;
		std::string getName() {	return "ManhattanDistanceLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief ManhattanDistance loss function gradient.
  */
  template<typename TensorT>
  class ManhattanDistanceLossGradOp : public LossFunctionGradOp<TensorT>
  {
public: 
		using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
		std::string getName() { return "ManhattanDistanceLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename TensorT>
  class L2NormLossOp : public LossFunctionOp<TensorT>
  {
public:
		using LossFunctionOp<TensorT>::LossFunctionOp;
		std::string getName() { return "L2NormLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename TensorT>
  class L2NormLossGradOp : public LossFunctionGradOp<TensorT>
  {
public:
		using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
		std::string getName() { return "L2NormLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief Binary Cross Entropy loss function.
  */
  template<typename TensorT>
  class BCELossOp : public LossFunctionOp<TensorT>
  {
public:
		using LossFunctionOp<TensorT>::LossFunctionOp;
		std::string getName() { return "BCELossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief Binary Cross Entropy loss function gradient.

	The derivative of -(z * log(x) + (1 - z)*log(1-x)) is the following
		= (1-z)/(1-x) - z/x
		= -(x-z)/((x-1)*x)
  */
  template<typename TensorT>
  class BCELossGradOp : public LossFunctionGradOp<TensorT>
  {
public:
		using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
		std::string getName() { return "BCELossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief NegativeLogLikelihood loss function.

		NOTES: implemented as the following:
		def CrossEntropy(yHat, y):
			if y == 1:
				return -log(yHat)
			else:
				return -log(1 - yHat)
  */
  template<typename TensorT>
  class NegativeLogLikelihoodLossOp : public LossFunctionOp<TensorT>
  {
public:
		using LossFunctionOp<TensorT>::LossFunctionOp;
		std::string getName() { return "NegativeLogLikelihoodLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename TensorT>
  class NegativeLogLikelihoodLossGradOp : public LossFunctionGradOp<TensorT>
  {
public:
		using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
		std::string getName() { return "NegativeLogLikelihoodLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename TensorT>
  class MSELossOp : public LossFunctionOp<TensorT>
  {
public:
		using LossFunctionOp<TensorT>::LossFunctionOp;
		std::string getName() { return "MSELossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename TensorT>
  class MSELossGradOp : public LossFunctionGradOp<TensorT>
  {
public:
		using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
		std::string getName() { return "MSELossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
  };

  /**
    @brief MAE Mean Absolute Error loss function.
  */
  template<typename TensorT>
  class MAELossOp : public LossFunctionOp<TensorT>
  {
  public:
    using LossFunctionOp<TensorT>::LossFunctionOp;
    std::string getName() { return "MAELossOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MAE Mean Abasolute Error loss function gradient.
  */
  template<typename TensorT>
  class MAELossGradOp : public LossFunctionGradOp<TensorT>
  {
  public:
    using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
    std::string getName() { return "MAELossGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };
  
  /**
    @brief MRSE Mean Roote Squared Error loss function.
  */
  template<typename TensorT>
  class MRSELossOp : public LossFunctionOp<TensorT>
  {
  public:
    using LossFunctionOp<TensorT>::LossFunctionOp;
    std::string getName() { return "MRSELossOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MRSE Mean Root Squared Error loss function gradient.
  */
  template<typename TensorT>
  class MRSELossGradOp : public LossFunctionGradOp<TensorT>
  {
  public:
    using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
    std::string getName() { return "MRSELossGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MLE Mean Logarithmic Error loss function.
  */
  template<typename TensorT>
  class MLELossOp : public LossFunctionOp<TensorT>
  {
  public:
    using LossFunctionOp<TensorT>::LossFunctionOp;
    std::string getName() { return "MLELossOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MLE Mean Logarithmic Error loss function gradient.
  */
  template<typename TensorT>
  class MLELossGradOp : public LossFunctionGradOp<TensorT>
  {
  public:
    using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
    std::string getName() { return "MLELossGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

	/**
		@brief KLDivergenceMu loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename TensorT>
	class KLDivergenceMuLossOp : public LossFunctionOp<TensorT>
	{
	public:
		KLDivergenceMuLossOp() = default;
		KLDivergenceMuLossOp(const TensorT & eps, const TensorT & scale, const TensorT& capacity) : LossFunctionOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceMuLossOp() = default;
		std::string getName() { return "KLDivergenceMuLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_, this->capacity_}); }
	private:
		TensorT capacity_ = TensorT(0);
	};

	/**
		@brief KLDivergenceMu  loss function gradient.
	*/
	template<typename TensorT>
	class KLDivergenceMuLossGradOp : public LossFunctionGradOp<TensorT>
	{
	public:
		KLDivergenceMuLossGradOp() = default;
		KLDivergenceMuLossGradOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionGradOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceMuLossGradOp() = default;
		std::string getName() { return "KLDivergenceMuLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_, this->capacity_ }); }
	private:
		TensorT capacity_ = TensorT(0);
	};

	/**
		@brief KLDivergenceLogVar loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename TensorT>
	class KLDivergenceLogVarLossOp : public LossFunctionOp<TensorT>
	{
	public:
		KLDivergenceLogVarLossOp() = default;
		KLDivergenceLogVarLossOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceLogVarLossOp() = default;
		std::string getName() { return "KLDivergenceLogVarLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_, this->capacity_ }); }
	private:
		TensorT capacity_ = TensorT(0);
	};

	/**
		@brief KLDivergenceLogVar  loss function gradient.
	*/
	template<typename TensorT>
	class KLDivergenceLogVarLossGradOp : public LossFunctionGradOp<TensorT>
	{
	public:
		KLDivergenceLogVarLossGradOp() = default;
		KLDivergenceLogVarLossGradOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionGradOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceLogVarLossGradOp() = default;
		std::string getName() { return "KLDivergenceLogVarLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_, this->capacity_ }); }
	private:
		TensorT capacity_ = TensorT(0);
	};

	/**
	@brief BCEWithLogits loss function.

	Binary Cross Entropy with integrated sigmoid layer
	z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
	= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
	= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
	= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
	= (1 - z) * x + log(1 + exp(-x))
	= x - x * z + log(1 + exp(-x))


	References:
	https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss

	PyTorch implementation:
	max_val = (-input).clamp(min=0)
	loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

	TensorFlow implementation:
	max(x, 0) - x * z + log(1 + exp(-abs(x)))
	*/
	template<typename TensorT>
	class BCEWithLogitsLossOp : public LossFunctionOp<TensorT>
	{
	public:
		using LossFunctionOp<TensorT>::LossFunctionOp;
		std::string getName() { return "BCEWithLogitsLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
	};

	/**
	@brief BCEWithLogits loss function gradient.

	Starting from the following BCEWithLogits formula
	x - x * z + log(1 + exp(-x))

	The derivative with respect to x can be formulated as
	1 - z + 1/(1 + exp(-x))*(-exp(-x))
	= -((z - 1)*exp(x) + z)/(exp(x) + 1)
	*/
	template<typename TensorT>
	class BCEWithLogitsLossGradOp : public LossFunctionGradOp<TensorT>
	{
	public:
		using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
		std::string getName() { return "BCEWithLogitsLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
	};

	/**
		@brief CrossEntropyWithLogits loss function.
	*/
	template<typename TensorT>
	class CrossEntropyWithLogitsLossOp : public LossFunctionOp<TensorT>
	{
	public:
		using LossFunctionOp<TensorT>::LossFunctionOp;
		std::string getName() { return "CrossEntropyWithLogitsLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
	};

	/**
		@brief CrossEntropyWithLogits loss function gradient.
	*/
	template<typename TensorT>
	class CrossEntropyWithLogitsLossGradOp : public LossFunctionGradOp<TensorT>
	{
	public:
		using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
		std::string getName() { return "CrossEntropyWithLogitsLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->eps_, this->scale_}); }
	};

  /**
    @brief MSE Mean Squared Error loss function for when a value is not within a specified range.
  */
  template<typename TensorT>
  class MSERangeUBLossOp : public LossFunctionOp<TensorT>
  {
  public:
    using LossFunctionOp<TensorT>::LossFunctionOp;
    std::string getName() { return "MSERangeUBLossOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MSE Mean Squared Error loss function gradient for when a value is not within a specified range.
  */
  template<typename TensorT>
  class MSERangeUBLossGradOp : public LossFunctionGradOp<TensorT>
  {
  public:
    using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
    std::string getName() { return "MSERangeUBLossGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MSE Mean Squared Error loss function for when a value is not within a specified range.
  */
  template<typename TensorT>
  class MSERangeLBLossOp : public LossFunctionOp<TensorT>
  {
  public:
    using LossFunctionOp<TensorT>::LossFunctionOp;
    std::string getName() { return "MSERangeLBLossOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MSE Mean Squared Error loss function gradient for when a value is not within a specified range.
  */
  template<typename TensorT>
  class MSERangeLBLossGradOp : public LossFunctionGradOp<TensorT>
  {
  public:
    using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
    std::string getName() { return "MSERangeLBLossGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief KLDivergenceCat loss function.

  References
    Maddison 2017 The concrete distribution
    Jang 2017 Categorical reparameterization with Gumbel-softmax
    Dupont 2018 Learning disentangled joint continuous and discrete representations

    KLD = alpha * log(alpha) + log(n) where n is the number of categories
  */
  template<typename TensorT>
  class KLDivergenceCatLossOp : public LossFunctionOp<TensorT>
  {
  public:
		KLDivergenceCatLossOp() = default;
		KLDivergenceCatLossOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceCatLossOp() = default;
    std::string getName() { return "KLDivergenceCatLossOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_, this->capacity_ }); }
	private:
		TensorT capacity_ = TensorT(0);
  };

  /**
    @brief KLDivergenceCat  loss function gradient.
  */
  template<typename TensorT>
  class KLDivergenceCatLossGradOp : public LossFunctionGradOp<TensorT>
  {
  public:
		KLDivergenceCatLossGradOp() = default;
		KLDivergenceCatLossGradOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionGradOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceCatLossGradOp() = default;
    std::string getName() { return "KLDivergenceCatLossGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_, this->capacity_ }); }
	private:
		TensorT capacity_ = TensorT(0);
  };

  /**
    @brief MAPELoss loss function
  */
  template<typename TensorT>
  class MAPELossOp : public LossFunctionOp<TensorT>
  {
  public:
    using LossFunctionOp<TensorT>::LossFunctionOp;
    std::string getName() { return "MAPELossOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MAPELoss  loss function gradient.
  */
  template<typename TensorT>
  class MAPELossGradOp : public LossFunctionGradOp<TensorT>
  {
  public:
    using LossFunctionGradOp<TensorT>::LossFunctionGradOp;
    std::string getName() { return "MAPELossGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->eps_, this->scale_ }); }
  };

	/**
		@brief Hinge loss function.  

		Typically used for classification

		NOTES: implemented as the following:
		def Hinge(yHat, y):
			return np.max(0, 1 - yHat * y)
	*/
}
#endif //SMARTPEAK_LOSSFUNCTION_H