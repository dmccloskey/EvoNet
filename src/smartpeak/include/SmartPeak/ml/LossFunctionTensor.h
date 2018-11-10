/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LOSSFUNCTIONTENSOR_H
#define SMARTPEAK_LOSSFUNCTIONTENSOR_H

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
	template<typename TensorT, typename DeviceT>
	class LossFunctionTensorOp
	{
	public:
		LossFunctionTensorOp() = default;
		LossFunctionTensorOp(const TensorT& eps) : eps_(eps) {};
		~LossFunctionTensorOp() = default;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
	protected:
		TensorT eps_ = 1e-6;
	};

	/**
	@brief Base class loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class LossFunctionGradTensorOp
	{
	public:
		LossFunctionGradTensorOp() = default;
		LossFunctionGradTensorOp(const TensorT& eps) : eps_(eps) {};
		~LossFunctionGradTensorOp() = default;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
	protected:
		TensorT eps_ = 1e-6;
	};

  /**
    @brief EuclideanDistance loss function.
  */
  template<typename TensorT, typename DeviceT>
  class EuclideanDistanceTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public: 
    EuclideanDistanceTensorOp(){}; 
    ~EuclideanDistanceTensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			Eigen::array<int, 1> dims({ 1 });			
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - predicted_chip).pow(2).sqrt()).sum(dims);// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief EuclideanDistance loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class EuclideanDistanceGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public: 
    EuclideanDistanceGradTensorOp(){}; 
    ~EuclideanDistanceGradTensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - predicted_chip) / ((expected_tensor - predicted_chip).pow(2).sqrt()));// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public: 
    L2NormTensorOp(){}; 
    ~L2NormTensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			Eigen::array<int, 1> dims({ 1 });
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - (predicted_chip).pow(2)) * expected_tensor.constant(0.5)).sum(dims);// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9)); // modified to simplify the derivative
		};
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public: 
    L2NormGradTensorOp(){}; 
    ~L2NormGradTensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += (expected_tensor - predicted_chip);// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9)); // modified to exclude the 0.5
		};
  };

  /**
    @brief Binary Cross Entropy loss function.
  */
  template<typename TensorT, typename DeviceT>
  class BCETensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public: 
    BCETensorOp(){}; 
    ~BCETensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			Eigen::array<int, 1> dims({ 1 });
			
			error_tensor.chip(time_step, 1).device(device) += (-(
				expected_tensor * (predicted_chip + expected_tensor.constant(this->eps_)).log() +
				(expected_tensor.constant(ones) - expected_tensor) * (expected_tensor.constant(ones) - (predicted_chip - expected_tensor.constant(this->eps_))).log())).sum(dims);// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief Binary Cross Entropy loss function gradient.

	The derivative of -(z * log(x) + (1 - z)*log(1-x)) is the following
		= (1-z)/(1-x) - z/x
		= -(x-z)/((x-1)*x)
  */
  template<typename TensorT, typename DeviceT>
  class BCEGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public: 
    BCEGradTensorOp(){}; 
    ~BCEGradTensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			//error_tensor.chip(time_step, 1).device(device) += (-(predicted_chip - expected_tensor) / ((predicted_chip.unaryExpr(OffsetTensorOp<TensorT>(- this->eps_ - ones))) * predicted_chip)).unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
			error_tensor.chip(time_step, 1).device(device) += (-(predicted_chip - expected_tensor) / ((predicted_chip - expected_tensor.constant(this->eps_) - expected_tensor.constant(ones)) * predicted_chip));// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));

		};
  };

  /**
    @brief NegativeLogLikelihood loss function.

		NOTES: implemented as the following:
		def CrossEntropy(yHat, y):
			if y == 1:
				error_tensor.chip(time_step, 1).device(device) += -log(yHat)
			else:
				error_tensor.chip(time_step, 1).device(device) += -log(1 - yHat)
  */
  template<typename TensorT, typename DeviceT>
  class NegativeLogLikelihoodTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public: 
    NegativeLogLikelihoodTensorOp() = default;
		NegativeLogLikelihoodTensorOp(const TensorT& n) { setN(n); };
		void setN(const TensorT& n) { n_ = n; }
    ~NegativeLogLikelihoodTensorOp() = default;
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			Eigen::array<int, 1> dims({ 1 });

			//error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (predicted_chip.unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1)).log())) * expected_tensor.constant(1 / batch_size)).sum(dims);
			error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (predicted_chip.log())) * expected_tensor.constant(1 / batch_size)).sum(dims);
		};
	private:
		TensorT n_ = 1.0; ///< the number of total classifiers
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class NegativeLogLikelihoodGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public: 
    NegativeLogLikelihoodGradTensorOp(){};
		NegativeLogLikelihoodGradTensorOp(const TensorT& n) { setN(n); };
		void setN(const TensorT& n) { n_ = n; }
    ~NegativeLogLikelihoodGradTensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += (expected_tensor / (predicted_chip + expected_tensor.constant(this->eps_)) / expected_tensor.constant(batch_size));// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
	private:
		TensorT n_ = 1.0; ///< the number of total classifiers
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSETensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public: 
    MSETensorOp(){}; 
    ~MSETensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const	{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			//Eigen::array<int, 1> dims({ 1 });
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - predicted_chip).pow(2) * expected_tensor.constant(0.5) / expected_tensor.constant(layer_size)).sum(Eigen::array<int, 1>({ 1 }));// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSEGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public: 
    MSEGradTensorOp(){}; 
    ~MSEGradTensorOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - predicted_chip) / expected_tensor.constant(layer_size)); // .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

	/**
		@brief KLDivergenceMu loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceMuTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceMuTensorOp() {};
		~KLDivergenceMuTensorOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			Eigen::array<int, 1> dims({ 1 });
			
			error_tensor.chip(time_step, 1).device(device) += (-expected_tensor.constant(0.5) + expected_tensor.constant(0.5)*predicted_chip.pow(2)).sum(dims);// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
	};

	/**
		@brief KLDivergenceMu  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceMuGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceMuGradTensorOp() {};
		~KLDivergenceMuGradTensorOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += expected_tensor.constant(2) * predicted_chip;
		};
	};

	/**
		@brief KLDivergenceLogVar loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceLogVarTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceLogVarTensorOp() {};
		~KLDivergenceLogVarTensorOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			Eigen::array<int, 1> dims({ 1 });
			
			error_tensor.chip(time_step, 1).device(device) += (-expected_tensor.constant(0.5) - expected_tensor.constant(0.5)*predicted_chip + expected_tensor.constant(0.5)*predicted_chip.exp()).sum(dims);// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
	};

	/**
		@brief KLDivergenceLogVar  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceLogVarGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceLogVarGradTensorOp() {};
		~KLDivergenceLogVarGradTensorOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += (-expected_tensor.constant(0.5) + expected_tensor.constant(0.5)*predicted_chip.exp());// .unaryExpr(ClipTensorOp<TensorT>(1e-6, -1e9, 1e9));
		};
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
	template<typename TensorT, typename DeviceT>
	class BCEWithLogitsTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		BCEWithLogitsTensorOp() {};
		~BCEWithLogitsTensorOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			Eigen::array<int, 1> dims({ 1 });
			
			// Step 1
			auto max_values = (-predicted_chip).cwiseMax(expected_tensor.constant(0));
			// Step 2
			error_tensor.chip(time_step, 1).device(device) += (predicted_chip - predicted_chip * expected_tensor + max_values + ((-max_values).exp() + (-predicted_chip - max_values).exp()).log()).sum(dims);
		};
	};

	/**
	@brief BCEWithLogits loss function gradient.

	Starting from the following BCEWithLogits formula
	x - x * z + log(1 + exp(-x))

	The derivative with respect to x can be formulated as
	1 - z + 1/(1 + exp(-x))*(-exp(-x))
	= -((z - 1)*exp(x) + z)/(exp(x) + 1)
	*/
	template<typename TensorT, typename DeviceT>
	class BCEWithLogitsGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		BCEWithLogitsGradTensorOp() {};
		~BCEWithLogitsGradTensorOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			//error_tensor.chip(time_step, 1).device(device) += -((expected_tensor - expected_tensor.constant(1))*predicted_chip.exp().unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1e9)) + expected_tensor) / (predicted_chip.exp().unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1e9)) + expected_tensor.constant(1));
			error_tensor.chip(time_step, 1).device(device) += -((expected_tensor - expected_tensor.constant(1))*predicted_chip.exp() + expected_tensor)/(predicted_chip.exp() + expected_tensor.constant(1));
		};
	};

	/**
		@brief Hinge loss function.  

		Typically used for classification

		NOTES: implemented as the following:
		def Hinge(yHat, y):
			error_tensor.chip(time_step, 1).device(device) += np.max(0, 1 - yHat * y)
	*/
}
#endif //SMARTPEAK_LOSSFUNCTIONTENSOR_H