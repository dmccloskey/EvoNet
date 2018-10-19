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
	template<typename TensorT, typename DeviceT>
	class LossFunctionOp
	{
	public:
		LossFunctionOp() = default;
		LossFunctionOp(const TensorT& eps) : eps_(eps) {};
		~LossFunctionOp() = default;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const = 0;
	protected:
		TensorT eps_ = 1e-6;
	};

	/**
	@brief Base class loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class LossFunctionGradOp
	{
	public:
		LossFunctionGradOp() = default;
		LossFunctionGradOp(const TensorT& eps) : eps_(eps) {};
		~LossFunctionGradOp() = default;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const = 0;
	protected:
		TensorT eps_ = 1e-6;
	};

  /**
    @brief EuclideanDistance loss function.
  */
  template<typename TensorT, typename DeviceT>
  class EuclideanDistanceOp : public LossFunctionOp<TensorT, DeviceT>
  {
public: 
    EuclideanDistanceOp(){}; 
    ~EuclideanDistanceOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += ((expected_chip - predicted_chip).pow(2).sqrt()).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief EuclideanDistance loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class EuclideanDistanceGradOp : public LossFunctionGradOp<TensorT, DeviceT>
  {
public: 
    EuclideanDistanceGradOp(){}; 
    ~EuclideanDistanceGradOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += ((expected_chip - predicted_chip) / ((expected_chip - predicted_chip).pow(2).sqrt())).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormOp : public LossFunctionOp<TensorT, DeviceT>
  {
public: 
    L2NormOp(){}; 
    ~L2NormOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += ((expected_chip - (predicted_chip).pow(2)).unaryExpr(ScaleOp<TensorT>(0.5))).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9)); // modified to simplify the derivative
		};
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormGradOp : public LossFunctionGradOp<TensorT, DeviceT>
  {
public: 
    L2NormGradOp(){}; 
    ~L2NormGradOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += (expected_chip - predicted_chip).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9)); // modified to exclude the 0.5
		};
  };

  /**
    @brief Binary Cross Entropy loss function.
  */
  template<typename TensorT, typename DeviceT>
  class BCEOp : public LossFunctionOp<TensorT, DeviceT>
  {
public: 
    BCEOp(){}; 
    ~BCEOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += (-(
				expected_chip * (predicted_chip + expected_chip.constant(this->eps_)).log() + 
				(expected_chip.constant(ones) - expected_chip) * (expected_chip.constant(ones) - (predicted_chip - expected_chip.constant(this->eps_))).log())).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief Binary Cross Entropy loss function gradient.

	The derivative of -(z * log(x) + (1 - z)*log(1-x)) is the following
		= (1-z)/(1-x) - z/x
		= -(x-z)/((x-1)*x)
  */
  template<typename TensorT, typename DeviceT>
  class BCEGradOp : public LossFunctionGradOp<TensorT, DeviceT>
  {
public: 
    BCEGradOp(){}; 
    ~BCEGradOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			//error_tensor.chip(time_step, 1).device(device) += (-(predicted_chip - expected_chip) / ((predicted_chip.unaryExpr(OffsetOp<TensorT>(- this->eps_ - ones))) * predicted_chip)).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
			error_tensor.chip(time_step, 1).device(device) += (-(predicted_chip - expected_chip) / ((predicted_chip - expected_chip.constant(this->eps_) - expected_chip.constant(ones)) * predicted_chip)).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));

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
  class NegativeLogLikelihoodOp : public LossFunctionOp<TensorT, DeviceT>
  {
public: 
    NegativeLogLikelihoodOp() = default;
		NegativeLogLikelihoodOp(const TensorT& n) { setN(n); };
		void setN(const TensorT& n) { n_ = n; }
    ~NegativeLogLikelihoodOp() = default;
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			//error_tensor.chip(time_step, 1).device(device) += (-expected_chip * (predicted_chip.unaryExpr(ClipOp<TensorT>(1e-6, 0, 1)).log())) * unaryExpr(ScaleOp<TensorT>(1 / batch_size));
			error_tensor.chip(time_step, 1).device(device) += (-expected_chip * (predicted_chip.unaryExpr(ClipOp<TensorT>(1e-6, 0, 1)).log())) * expected_chip.constant(1 / batch_size);
		};
	private:
		TensorT n_ = 1.0; ///< the number of total classifiers
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class NegativeLogLikelihoodGradOp : public LossFunctionGradOp<TensorT, DeviceT>
  {
public: 
    NegativeLogLikelihoodGradOp(){};
		NegativeLogLikelihoodGradOp(const TensorT& n) { setN(n); };
		void setN(const TensorT& n) { n_ = n; }
    ~NegativeLogLikelihoodGradOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			//error_tensor.chip(time_step, 1).device(device) += (expected_chip / (predicted_chip.unaryExpr(OffsetOp<TensorT>(this->eps_)).unaryExpr(ScaleOp<TensorT>(1 / batch_size))).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
			error_tensor.chip(time_step, 1).device(device) += (expected_chip / (predicted_chip + expected_chip.constant(this->eps_)) / expected_chip.constant(batch_size)).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
		};
	private:
		TensorT n_ = 1.0; ///< the number of total classifiers
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSEOp : public LossFunctionOp<TensorT, DeviceT>
  {
public: 
    MSEOp(){}; 
    ~MSEOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += ((expected_chip - predicted_chip).pow(2) * expected_chip.constant(0.5) / expected_chip.constant(batch_size));// .unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSEGradOp : public LossFunctionGradOp<TensorT, DeviceT>
  {
public: 
    MSEGradOp(){}; 
    ~MSEGradOp(){};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += ((expected_chip - predicted_chip) / expected_chip.constant(batch_size)); // .unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
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
	class KLDivergenceMuOp : public LossFunctionOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceMuOp() {};
		~KLDivergenceMuOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += (-expected_chip.constant(0.5) + expected_chip.constant(0.5)*predicted_chip.pow(2)).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
		};
	};

	/**
		@brief KLDivergenceMu  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceMuGradOp : public LossFunctionGradOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceMuGradOp() {};
		~KLDivergenceMuGradOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += expected_chip.constant(2) * predicted_chip;
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
	class KLDivergenceLogVarOp : public LossFunctionOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceLogVarOp() {};
		~KLDivergenceLogVarOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += (-expected_chip.constant(0.5) - expected_chip.constant(0.5)*predicted_chip + expected_chip.constant(0.5)*predicted_chip.exp()).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
		};
	};

	/**
		@brief KLDivergenceLogVar  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceLogVarGradOp : public LossFunctionGradOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceLogVarGradOp() {};
		~KLDivergenceLogVarGradOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += (-expected_chip.constant(0.5) + expected_chip.constant(0.5)*predicted_chip.exp()).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
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
	class BCEWithLogitsOp : public LossFunctionOp<TensorT, DeviceT>
	{
	public:
		BCEWithLogitsOp() {};
		~BCEWithLogitsOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			// Step 1
			auto max_values = (-predicted_chip).cwiseMax(expected_chip.constant(0));
			// Step 2
			error_tensor.chip(time_step, 1).device(device) += predicted_chip - predicted_chip * expected_chip + max_values + ((-max_values).exp() + (-predicted_chip - max_values).exp()).log();
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
	class BCEWithLogitsGradOp : public LossFunctionGradOp<TensorT, DeviceT>
	{
	public:
		BCEWithLogitsGradOp() {};
		~BCEWithLogitsGradOp() {};
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& time_step, const int& n_error_nodes, const int& node_iter, DeviceT& device) const
		{
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, n_error_nodes);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> predicted_tensor(predicted, batch_size, memory_size);
			Eigen::TensorMap < Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto expected_chip = expected_tensor.chip(node_iter, 1);
			error_tensor.chip(time_step, 1).device(device) += -((expected_chip - expected_chip.constant(1))*predicted_chip.exp().unaryExpr(ClipOp<TensorT>(1e-6, 0, 1e9)) + expected_chip)/(predicted_chip.exp().unaryExpr(ClipOp<TensorT>(1e-6, 0, 1e9)) + expected_chip.constant(1));
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
#endif //SMARTPEAK_LOSSFUNCTION_H