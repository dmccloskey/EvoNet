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
		LossFunctionTensorOp(const TensorT& eps, const TensorT& scale) : eps_(eps), scale_(scale) {};
		~LossFunctionTensorOp() = default;
		virtual std::string getName() = 0;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
	protected:
		TensorT eps_ = 1e-6;
		TensorT scale_ = 1.0;
	};

	/**
	@brief Base class loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class LossFunctionGradTensorOp
	{
	public:
		LossFunctionGradTensorOp() = default;
		LossFunctionGradTensorOp(const TensorT& eps, const TensorT& scale) : eps_(eps), scale_(scale) {};
		~LossFunctionGradTensorOp() = default;
		virtual std::string getName() = 0;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
	protected:
		TensorT eps_ = 1e-6;
		TensorT scale_ = 1.0;
	};

  /**
    @brief EuclideanDistance loss function.
  */
  template<typename TensorT, typename DeviceT>
  class EuclideanDistanceTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public: 
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "EuclideanDistanceTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);	
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip).pow((TensorT)2).sqrt()).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
  };

  /**
    @brief EuclideanDistance loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class EuclideanDistanceGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "EuclideanDistanceGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - predicted_chip) / ((expected_tensor - predicted_chip - expected_tensor.constant(this->eps_)).pow((TensorT)2).sqrt()))*error_tensor.chip(time_step, 1).constant(this->scale_);
		};
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "L2NormTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - (predicted_chip).pow((TensorT)2)) * expected_tensor.constant((TensorT)0.5)).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9)); // modified to simplify the derivative
		};
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "L2NormGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - predicted_chip) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9)); // modified to exclude the 0.5
		};
  };

  /**
    @brief Binary Cross Entropy loss function.
  */
  template<typename TensorT, typename DeviceT>
  class BCETensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "BCETensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((-(
				expected_tensor * (predicted_chip + expected_tensor.constant(this->eps_)).log() + // check if .clip((TensorT)1e-6,(TensorT)1) should be used instead
				(expected_tensor.constant((TensorT)1) - expected_tensor) * (expected_tensor.constant((TensorT)1) - (predicted_chip - expected_tensor.constant(this->eps_))).log())).sum(Eigen::array<int, 1>({ 1 }))
				* error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
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
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "BCEGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
      // NOTE: change `(predicted_chip - expected_tensor)` to `-(predicted_chip - expected_tensor)`
			error_tensor.chip(time_step, 1).device(device) += ((-(predicted_chip - expected_tensor) / (((predicted_chip - expected_tensor.constant((TensorT)1)) * predicted_chip) + expected_tensor.constant(this->eps_)))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));

		};
  };

  /**
    @brief NegativeLogLikelihood loss function.
  */
  template<typename TensorT, typename DeviceT>
  class NegativeLogLikelihoodTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		void setN(const TensorT& n) { n_ = n; }
		std::string getName() { return "NegativeLogLikelihoodTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);

			//error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (predicted_chip.unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1)).log())) * expected_tensor.constant((TensorT)1 / layer_size)).sum(Eigen::array<int, 1>({ 1 }));
			error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (predicted_chip.clip((TensorT)1e-6,(TensorT)1).log())) * expected_tensor.constant((TensorT)1 / (TensorT)layer_size)).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_);
		};
	private:
		TensorT n_ = (TensorT)1; ///< the number of total classifiers
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class NegativeLogLikelihoodGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		void setN(const TensorT& n) { n_ = n; }
		std::string getName() { return "NegativeLogLikelihoodGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			// NOTE: added - so that the gradient is -
			error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor / (predicted_chip + expected_tensor.constant(this->eps_)) / expected_tensor.constant((TensorT)layer_size))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
	private:
		TensorT n_ = (TensorT)1.0; ///< the number of total classifiers
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSETensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "MSETensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const	{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip).pow((TensorT)2) * expected_tensor.constant((TensorT)0.5) / expected_tensor.constant((TensorT)layer_size)).sum(Eigen::array<int, 1>({ 1 }))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSEGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "MSEGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip) / expected_tensor.constant((TensorT)layer_size))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
  };

	/**
		@brief KLDivergenceMu loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow((TensorT)2) - logvar.exp())
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceMuTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "KLDivergenceMuTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor.constant((TensorT)0.5) + expected_tensor.constant((TensorT)0.5)*predicted_chip.pow((TensorT)2)).sum(Eigen::array<int, 1>({ 1 }))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
	};

	/**
		@brief KLDivergenceMu  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceMuGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "KLDivergenceMuGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			// NOTE: changed to -= to ensure a negative gradient
			error_tensor.chip(time_step, 1).device(device) -= expected_tensor.constant((TensorT)2) * predicted_chip * error_tensor.chip(time_step, 1).constant(this->scale_);
		};
	};

	/**
		@brief KLDivergenceLogVar loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow((TensorT)2) - logvar.exp())
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceLogVarTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "KLDivergenceLogVarTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);

			error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor.constant((TensorT)0.5) - expected_tensor.constant((TensorT)0.5)*predicted_chip + (expected_tensor.constant((TensorT)0.5)*predicted_chip).exp()).sum(Eigen::array<int, 1>({ 1 }))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
	};

	/**
		@brief KLDivergenceLogVar  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceLogVarGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "KLDivergenceLogVarGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			// NOTE: changed to -= to ensure a negative gradient
			error_tensor.chip(time_step, 1).device(device) -= ((-expected_tensor.constant((TensorT)0.5) + (expected_tensor.constant((TensorT)0.5)*predicted_chip).exp())
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
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
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "BCEWithLogitsTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);			
			// Step 1
			auto max_values = (-predicted_chip).cwiseMax(expected_tensor.constant((TensorT)0));
			// Step 2 //NOTE: removed .clip(-1e9, 20.72) before .exp()
			error_tensor.chip(time_step, 1).device(device) += (predicted_chip - predicted_chip * expected_tensor + max_values + ((-max_values).exp() + (-predicted_chip - max_values).exp()).log()).sum(Eigen::array<int, 1>({ 1 }))*error_tensor.chip(time_step, 1).constant(this->scale_);
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
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "BCEWithLogitsGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			//error_tensor.chip(time_step, 1).device(device) += -((expected_tensor - expected_tensor.constant(1))*predicted_chip.exp().unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1e9)) + expected_tensor) / (predicted_chip.exp().unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1e9)) + expected_tensor.constant(1));
			// NOTE: removed -(( to (( to ensure negative gradients
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - expected_tensor.constant((TensorT)1))*predicted_chip.exp() + expected_tensor)/(predicted_chip.exp() + expected_tensor.constant((TensorT)1)) * error_tensor.chip(time_step, 1).constant(this->scale_);
		};
	};

	/**
		@brief Softmax + Cross Entropy loss function.

		NOTES: implemented as the following:

		def stable_softmax(X):
			exps = np.exp(X - np.max(X))
			return exps / np.sum(exps)

		def cross_entropy(p,y):
			"""
			p is the output from softmax layer (num_examples x num_classes)
			y is labels (num_examples x 1)
			"""
			m = y.shape[0]
			log_likelihood = -np.log(p[range(m),y])
			loss = np.sum(log_likelihood) / m
			return loss
	*/
	template<typename TensorT, typename DeviceT>
	class CrossEntropyWithLogitsTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "CrossEntropyWithLogitsTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 5>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1, 1);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1); // 4 dims
			auto exps = (predicted_chip.chip(0, 3) - predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 3>({1, layer_size, 1}))).exp(); // 3 dims
			auto stable_softmax = exps.chip(0, 2) / exps.sum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));  // 2 dims

			//error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (stable_softmax.unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1)).log())) * expected_tensor.constant((TensorT)1 / layer_size)).sum(Eigen::array<int, 1>({ 1 }));
			error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (stable_softmax.clip((TensorT)1e-6,(TensorT)1).log())) * expected_tensor.constant((TensorT)1 / (TensorT)layer_size)).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_);
		};
	};

	/**
		@brief Softmax + Cross Entropy loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class CrossEntropyWithLogitsGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "CrossEntropyWithLogitsGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			// NOTE: changed from prediced_chip - expected to expected - predicted_chip
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip) / expected_tensor.constant((TensorT)layer_size)) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
		};
	};

  /**
    @brief MSERangeUB Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeUBTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MSERangeUBTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse = ((expected_tensor - predicted_chip).pow((TensorT)2) * expected_tensor.constant((TensorT)0.5) / expected_tensor.constant((TensorT)layer_size));        
      auto in_range = predicted_chip > expected_tensor;
      auto result = in_range.select(mse, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += (result.sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
    };
  };

  /**
    @brief MSERangeUB Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeUBGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MSERangeUBGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse_grad = (((expected_tensor - predicted_chip) / expected_tensor.constant((TensorT)layer_size))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
      auto in_range = predicted_chip > expected_tensor;
      auto result = in_range.select(mse_grad, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += result;
    };
  };

  /**
    @brief MSERangeLB Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeLBTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MSERangeLBTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse = ((expected_tensor - predicted_chip).pow((TensorT)2) * expected_tensor.constant((TensorT)0.5) / expected_tensor.constant((TensorT)layer_size));
      auto in_range = predicted_chip < expected_tensor;
      auto result = in_range.select(mse, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += (result.sum(Eigen::array<int, 1>({ 1 }))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
    };
  };

  /**
    @brief MSERangeLB Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeLBGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MSERangeLBGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap < Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse_grad = (((expected_tensor - predicted_chip) / expected_tensor.constant((TensorT)layer_size))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(TensorT(-1e9),TensorT(1e9));
      auto in_range = predicted_chip < expected_tensor;
      auto result = in_range.select(mse_grad, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += result;
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