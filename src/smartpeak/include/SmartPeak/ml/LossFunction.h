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
	template<typename T>
	class LossFunctionOp
	{
	public:
		LossFunctionOp() = default;
		LossFunctionOp(const T& eps, const T& scale) : eps_(eps), scale_(scale) {};
		~LossFunctionOp() = default;
		virtual std::string getName() = 0;
		virtual std::vector<T> getParameters() const = 0;
	protected:
		T eps_ = 1e-6;
		T scale_ = 1.0;
	};

	/**
	@brief Base class loss function gradient.
	*/
	template<typename T>
	class LossFunctionGradOp
	{
	public:
		LossFunctionGradOp() = default;
		LossFunctionGradOp(const T& eps, const T& scale) : eps_(eps), scale_(scale) {};
		~LossFunctionGradOp() = default;
		virtual std::string getName() = 0;
		virtual std::vector<T> getParameters() const = 0;
	protected:
		T eps_ = 1e-6;
		T scale_ = 1.0;
	};

  /**
    @brief EuclideanDistance loss function.
  */
  template<typename T>
  class EuclideanDistanceOp : public LossFunctionOp<T>
  {
public: 
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() {	return "EuclideanDistanceOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief EuclideanDistance loss function gradient.
  */
  template<typename T>
  class EuclideanDistanceGradOp : public LossFunctionGradOp<T>
  {
public: 
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "EuclideanDistanceGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename T>
  class L2NormOp : public LossFunctionOp<T>
  {
public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "L2NormOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename T>
  class L2NormGradOp : public LossFunctionGradOp<T>
  {
public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "L2NormGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief Binary Cross Entropy loss function.
  */
  template<typename T>
  class BCEOp : public LossFunctionOp<T>
  {
public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "BCEOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief Binary Cross Entropy loss function gradient.

	The derivative of -(z * log(x) + (1 - z)*log(1-x)) is the following
		= (1-z)/(1-x) - z/x
		= -(x-z)/((x-1)*x)
  */
  template<typename T>
  class BCEGradOp : public LossFunctionGradOp<T>
  {
public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "BCEGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
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
  template<typename T>
  class NegativeLogLikelihoodOp : public LossFunctionOp<T>
  {
public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "NegativeLogLikelihoodOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename T>
  class NegativeLogLikelihoodGradOp : public LossFunctionGradOp<T>
  {
public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "NegativeLogLikelihoodGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename T>
  class MSEOp : public LossFunctionOp<T>
  {
public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "MSEOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename T>
  class MSEGradOp : public LossFunctionGradOp<T>
  {
public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "MSEGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
  };

	/**
		@brief KLDivergenceMu loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename T>
	class KLDivergenceMuOp : public LossFunctionOp<T>
	{
	public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "KLDivergenceMuOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
	};

	/**
		@brief KLDivergenceMu  loss function gradient.
	*/
	template<typename T>
	class KLDivergenceMuGradOp : public LossFunctionGradOp<T>
	{
	public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "KLDivergenceMuGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
	};

	/**
		@brief KLDivergenceLogVar loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename T>
	class KLDivergenceLogVarOp : public LossFunctionOp<T>
	{
	public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "KLDivergenceLogVarOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
	};

	/**
		@brief KLDivergenceLogVar  loss function gradient.
	*/
	template<typename T>
	class KLDivergenceLogVarGradOp : public LossFunctionGradOp<T>
	{
	public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "KLDivergenceLogVarGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
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
	template<typename T>
	class BCEWithLogitsOp : public LossFunctionOp<T>
	{
	public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "BCEWithLogitsOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
	};

	/**
	@brief BCEWithLogits loss function gradient.

	Starting from the following BCEWithLogits formula
	x - x * z + log(1 + exp(-x))

	The derivative with respect to x can be formulated as
	1 - z + 1/(1 + exp(-x))*(-exp(-x))
	= -((z - 1)*exp(x) + z)/(exp(x) + 1)
	*/
	template<typename T>
	class BCEWithLogitsGradOp : public LossFunctionGradOp<T>
	{
	public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "BCEWithLogitsGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
	};

	/**
		@brief CrossEntropyWithLogits loss function.
	*/
	template<typename T>
	class CrossEntropyWithLogitsOp : public LossFunctionOp<T>
	{
	public:
		using LossFunctionOp<T>::LossFunctionOp;
		std::string getName() { return "CrossEntropyWithLogitsOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
	};

	/**
		@brief CrossEntropyWithLogits loss function gradient.
	*/
	template<typename T>
	class CrossEntropyWithLogitsGradOp : public LossFunctionGradOp<T>
	{
	public:
		using LossFunctionGradOp<T>::LossFunctionGradOp;
		std::string getName() { return "CrossEntropyWithLogitsGradOp"; };
		std::vector<T> getParameters() const { return std::vector<T>({this->eps_, this->scale_}); }
	};

  /**
    @brief MSE Mean Squared Error loss function for when a value is not within a specified range.
  */
  template<typename T>
  class MSERangeUBOp : public LossFunctionOp<T>
  {
  public:
    using LossFunctionOp<T>::LossFunctionOp;
    std::string getName() { return "MSERangeUBOp"; };
    std::vector<T> getParameters() const { return std::vector<T>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MSE Mean Squared Error loss function gradient for when a value is not within a specified range.
  */
  template<typename T>
  class MSERangeUBGradOp : public LossFunctionGradOp<T>
  {
  public:
    using LossFunctionGradOp<T>::LossFunctionGradOp;
    std::string getName() { return "MSERangeUBGradOp"; };
    std::vector<T> getParameters() const { return std::vector<T>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MSE Mean Squared Error loss function for when a value is not within a specified range.
  */
  template<typename T>
  class MSERangeLBOp : public LossFunctionOp<T>
  {
  public:
    using LossFunctionOp<T>::LossFunctionOp;
    std::string getName() { return "MSERangeLBOp"; };
    std::vector<T> getParameters() const { return std::vector<T>({ this->eps_, this->scale_ }); }
  };

  /**
    @brief MSE Mean Squared Error loss function gradient for when a value is not within a specified range.
  */
  template<typename T>
  class MSERangeLBGradOp : public LossFunctionGradOp<T>
  {
  public:
    using LossFunctionGradOp<T>::LossFunctionGradOp;
    std::string getName() { return "MSERangeLBGradOp"; };
    std::vector<T> getParameters() const { return std::vector<T>({ this->eps_, this->scale_ }); }
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