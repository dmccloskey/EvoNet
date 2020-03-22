/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LOSSFUNCTIONTENSOR_H
#define SMARTPEAK_LOSSFUNCTIONTENSOR_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <SmartPeak/core/Preprocessing.h>
#include <unsupported/Eigen/CXX11/Tensor>
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
		virtual ~LossFunctionTensorOp() = default;
		virtual std::string getName() = 0;
		virtual void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
	protected:
		TensorT eps_ = TensorT(1e-24);
		TensorT scale_ = TensorT(1.0);
    TensorT reward_ = TensorT(10.0);
    TensorT min_ = TensorT(-1e9);
    TensorT max_ = TensorT(1e9);
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
    TensorT eps_ = TensorT(1e-24);
    TensorT scale_ = TensorT(1.0);
    TensorT reward_ = TensorT(10.0);
    TensorT min_ = TensorT(-1e9);
    TensorT max_ = TensorT(1e9);
	};

  /**
    @brief Manhattan loss function.
  */
  template<typename TensorT, typename DeviceT>
  class ManhattanDistanceLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public: 
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "ManhattanDistanceLossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);	
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip).pow(TensorT(2)).sqrt()).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
		};
  };

  /**
    @brief Manhattan distance loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class ManhattanDistanceLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "ManhattanDistanceLossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto result = (expected_tensor - predicted_chip == predicted_chip.constant(TensorT(0))).select(
        predicted_chip.constant(TensorT(0)),
        ((expected_tensor - predicted_chip) / ((expected_tensor - predicted_chip).pow(TensorT(2)).sqrt()))*error_tensor.chip(time_step, 1).constant(this->scale_));
			error_tensor.chip(time_step, 1).device(device) += result.clip(this->min_, this->max_);
		};
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "L2NormLossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - (predicted_chip).pow(TensorT(2))) * expected_tensor.constant(TensorT(0.5))).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_); // modified to simplify the derivative
		};
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class L2NormLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "L2NormLossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += ((expected_tensor - predicted_chip) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_); // modified to exclude the 0.5
		};
  };

  /**
    @brief Binary Cross Entropy loss function.
  */
  template<typename TensorT, typename DeviceT>
  class BCELossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "BCELossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto tmp = -(
        expected_tensor * predicted_chip.clip(this->eps_, TensorT(1)).log() +
        (expected_tensor.constant(TensorT(1)) - expected_tensor) * (expected_tensor.constant(TensorT(1)) - predicted_chip).clip(this->eps_, TensorT(1)).log()
      );
			error_tensor.chip(time_step, 1).device(device) += (tmp.sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
		};
  };

  /**
    @brief Binary Cross Entropy loss function gradient.

	The derivative of -(z * log(x) + (1 - z)*log(1-x)) is the following
		= (1-z)/(1-x) - z/x
		= -(x-z)/((x-1)*x)
  */
  template<typename TensorT, typename DeviceT>
  class BCELossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "BCELossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto term1 = expected_tensor / predicted_chip.clip(this->eps_, TensorT(1));
      auto term2 = (expected_tensor.constant(TensorT(1)) - expected_tensor) / (expected_tensor.constant(TensorT(1)) - predicted_chip.clip(TensorT(0), TensorT(1) - this->eps_));
      auto result = term1 - term2;
      //auto result = (predicted_chip - expected_tensor) / ((predicted_chip - expected_tensor.constant(TensorT(1))) * predicted_chip);
			error_tensor.chip(time_step, 1).device(device) += (result*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
		};
  };

  /**
    @brief NegativeLogLikelihood loss function.
  */
  template<typename TensorT, typename DeviceT>
  class NegativeLogLikelihoodLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		void setN(const TensorT& n) { n_ = n; }
		std::string getName() { return "NegativeLogLikelihoodLossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);

			//error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (predicted_chip.unaryExpr(ClipTensorOp<TensorT>(1e-6, 0, 1)).log())) * expected_tensor.constant(TensorT(1) / layer_size)).sum(Eigen::array<int, 1>({ 1 }));
			error_tensor.chip(time_step, 1).device(device) += ((-expected_tensor * (predicted_chip.clip(TensorT(1e-6),TensorT(1)).log())) * expected_tensor.constant(TensorT(1) / TensorT(layer_size))).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_);
		};
	private:
		TensorT n_ = TensorT(1); ///< the number of total classifiers
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class NegativeLogLikelihoodLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		void setN(const TensorT& n) { n_ = n; }
		std::string getName() { return "NegativeLogLikelihoodLossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			// NOTE: added - so that the gradient is -
			error_tensor.chip(time_step, 1).device(device) -= ((expected_tensor / (predicted_chip + expected_tensor.constant(TensorT(this->eps_))) / expected_tensor.constant(TensorT(layer_size)))
				*error_tensor.chip(time_step, 1).constant(TensorT(this->scale_))).clip(this->min_, this->max_);
		};
	private:
		TensorT n_ = TensorT(1.0); ///< the number of total classifiers
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSELossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "MSELossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const	{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip).pow(TensorT(2)) * expected_tensor.constant(TensorT(0.5)) / expected_tensor.constant(TensorT(layer_size))).sum(Eigen::array<int, 1>({ 1 }))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
		};
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSELossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "MSELossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip) / expected_tensor.constant(TensorT(layer_size)))
				*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
		};
  };

  /**
    @brief MAE Mean Absolute Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MAELossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MAELossTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);

      error_tensor.chip(time_step, 1).device(device) += (((expected_tensor - predicted_chip).pow(TensorT(2)).sqrt() / expected_tensor.constant(TensorT(layer_size))).sum(Eigen::array<int, 1>({ 1 }))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

  /**
    @brief MAE Mean Absolute Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MAELossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MAELossGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto result = (expected_tensor - predicted_chip == predicted_chip.constant(TensorT(0))).select(
        predicted_chip.constant(TensorT(0)),
        (((expected_tensor - predicted_chip) / (expected_tensor - predicted_chip).pow(TensorT(2)).sqrt() / expected_tensor.constant(TensorT(layer_size)))*error_tensor.chip(time_step, 1).constant(this->scale_))
      );
      error_tensor.chip(time_step, 1).device(device) += result.clip(this->min_, this->max_);
    };
  };

  /**
    @brief MRSE Mean Root Squared Error loss function. WIP.

    Based on the following references:
    https://stats.stackexchange.com/questions/102810/pros-of-jeffries-matusita-distance
    https://en.wikipedia.org/wiki/Bhattacharyya_distance
  */
  template<typename TensorT, typename DeviceT>
  class MRSELossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MRSELossTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto min_offset = predicted_chip.chip(0, 2) - predicted_chip.minimum(Eigen::array<Eigen::Index, 1>({1})).broadcast(Eigen::array<Eigen::Index, 2>({ 1, layer_size }));

      error_tensor.chip(time_step, 1).device(device) += (((expected_tensor.sqrt() - min_offset.sqrt()).pow(TensorT(2)) / expected_tensor.constant(TensorT(layer_size))).sum(Eigen::array<int, 1>({ 1 }))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

  /**
    @brief MRSE Mean Root Squared Error loss function gradient. WIP.
  */
  template<typename TensorT, typename DeviceT>
  class MRSELossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MRSELossGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto min_offset = predicted_chip.chip(0, 2) - predicted_chip.minimum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, layer_size }));
      error_tensor.chip(time_step, 1).device(device) += (((expected_tensor.sqrt() - min_offset.sqrt()) / (min_offset.sqrt() - expected_tensor.constant(this->eps_)) / expected_tensor.constant(TensorT(layer_size)))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

  /**
    @brief MLE Mean Root Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MLELossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MLELossTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto diff = expected_tensor - predicted_chip;
      auto min_offset = diff.chip(0, 2) - diff.minimum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, layer_size })) + diff.chip(0, 2).constant(TensorT(1));
      error_tensor.chip(time_step, 1).device(device) += ((min_offset.log() / expected_tensor.chip(0, 2).constant(TensorT(layer_size))).sum(Eigen::array<int, 1>({ 1 }))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

  /**
    @brief MLE Mean Root Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MLELossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MLELossGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto diff = expected_tensor - predicted_chip;
      auto min_offset = diff.chip(0, 2) - diff.minimum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, layer_size })) + diff.chip(0, 2).constant(TensorT(1));
      // TODO: change to (min_offset == min_offset.constant(TensorT(0))).select(min_offset.constant(TensorT(0)), ((expected_tensor.chip(0, 2).constant(TensorT(1)) / min_offset / expected_tensor.chip(0, 2).constant(TensorT(layer_size)))*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_));
      error_tensor.chip(time_step, 1).device(device) += ((expected_tensor.chip(0, 2).constant(TensorT(1)) / (min_offset - expected_tensor.chip(0, 2).constant(this->eps_)) / expected_tensor.chip(0, 2).constant(TensorT(layer_size)))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

	/**
		@brief KLDivergenceMu loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(TensorT(2)) - logvar.exp())
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceMuLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceMuLossTensorOp() = default;
		KLDivergenceMuLossTensorOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionTensorOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceMuLossTensorOp() = default;
		std::string getName() { return "KLDivergenceMuLossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			
			auto kl_div = (-expected_tensor.constant(TensorT(0.5)) + expected_tensor.constant(TensorT(0.5)) * predicted_chip.pow(TensorT(2))).sum(Eigen::array<int, 1>({ 1 }));
			auto kl_div_cap = kl_div - error_tensor.chip(time_step, 1).constant(this->capacity_);
      auto result = kl_div_cap * error_tensor.chip(time_step, 1).constant(this->scale_);
			error_tensor.chip(time_step, 1).device(device) += (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));
		};
	private:
		TensorT capacity_ = TensorT(0);
	};

	/**
		@brief KLDivergenceMu  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceMuLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceMuLossGradTensorOp() = default;
		KLDivergenceMuLossGradTensorOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionGradTensorOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceMuLossGradTensorOp() = default;
		std::string getName() { return "KLDivergenceMuLossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto kl_div = expected_tensor.constant(TensorT(2)) * predicted_chip;
			auto kl_div_cap = kl_div - expected_tensor.constant(this->capacity_);
			// NOTE: changed to -= to ensure a negative gradient
			error_tensor.chip(time_step, 1).device(device) -= (kl_div_cap * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
		};
	private:
		TensorT capacity_ = TensorT(0);
	};

	/**
		@brief KLDivergenceLogVar loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(TensorT(2)) - logvar.exp())
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceLogVarLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceLogVarLossTensorOp() = default;
		KLDivergenceLogVarLossTensorOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionTensorOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceLogVarLossTensorOp() = default;
		std::string getName() { return "KLDivergenceLogVarLossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto kl_div = (-expected_tensor.constant(TensorT(0.5)) - expected_tensor.constant(TensorT(0.5)) * predicted_chip + (expected_tensor.constant(TensorT(0.5)) * predicted_chip).exp()).sum(Eigen::array<int, 1>({ 1 }));
			auto kl_div_cap = kl_div - error_tensor.chip(time_step, 1).constant(this->capacity_);
      auto result = kl_div_cap * error_tensor.chip(time_step, 1).constant(this->scale_);
      error_tensor.chip(time_step, 1).device(device) += (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));
		};
	private:
		TensorT capacity_ = TensorT(0);
	};

	/**
		@brief KLDivergenceLogVar  loss function gradient.
	*/
	template<typename TensorT, typename DeviceT>
	class KLDivergenceLogVarLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		KLDivergenceLogVarLossGradTensorOp() = default;
		KLDivergenceLogVarLossGradTensorOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionGradTensorOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceLogVarLossGradTensorOp() = default;
		std::string getName() { return "KLDivergenceLogVarLossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto kl_div = -expected_tensor.constant(TensorT(0.5)) + (expected_tensor.constant(TensorT(0.5)) * predicted_chip).exp();
			auto kl_div_cap = kl_div - expected_tensor.constant(this->capacity_);
      auto result = kl_div_cap * error_tensor.chip(time_step, 1).constant(this->scale_);
			// NOTE: changed to -= to ensure a negative gradient
			error_tensor.chip(time_step, 1).device(device) -= (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));
		};
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
	template<typename TensorT, typename DeviceT>
	class BCEWithLogitsLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "BCEWithLogitsLossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
			//auto max_values = (-predicted_chip).cwiseMax(expected_tensor.constant(TensorT(0))); // pytorch version
      //auto max_values = predicted_chip.cwiseMax(expected_tensor.constant(TensorT(0))); // tensorFlow version
      auto max_values = (expected_tensor < expected_tensor.constant(TensorT(0))).select(predicted_chip.cwiseMin(expected_tensor.constant(TensorT(0))), predicted_chip.cwiseMax(expected_tensor.constant(TensorT(0)))); // custom version
      auto abs_values = -(predicted_chip.abs()); // tensorFlow and custom versions

      // Temporary memory for computation
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> result(tmp_data, batch_size);
      //result.device(device) = (
      //  predicted_chip - predicted_chip * expected_tensor + max_values + ((-max_values).exp() + (-predicted_chip - max_values).exp()).log()
      //  ).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_); // pytorch version
      //result.device(device) = (
      //  max_values - predicted_chip * expected_tensor + (expected_tensor.constant(TensorT(1)) + abs_values.exp()).log()
      //  ).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_); // tensorFlow version
      result.device(device) = (
        max_values - predicted_chip * expected_tensor.abs() + (expected_tensor.constant(TensorT(1)) + abs_values.exp()).log()
        ).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_); // custom version
      error_tensor.chip(time_step, 1).device(device) += (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));

      // Deallocate temporary memory
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
	@brief BCEWithLogits loss function gradient.

  Starting from the following BCEWithLogits formula
  maxOrMin[z](x, 0) - x * abs(z) + log(1 + exp(-abs(x)))

  The derivative with respect to x can be formulated as
  -x*exp(-abs(x)) / ((exp(-abs(x)) + 1) * abs(x)) - abs(z) + maxOrMin[z](x/abs(x), 0)

	*/
	template<typename TensorT, typename DeviceT>
	class BCEWithLogitsLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "BCEWithLogitsLossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto predicted_dir = predicted_chip / predicted_chip.abs();
      auto max_values = (expected_tensor < expected_tensor.constant(TensorT(0))).select(predicted_dir.cwiseMin(expected_tensor.constant(TensorT(0))), predicted_dir.cwiseMax(expected_tensor.constant(TensorT(0)))); // custom version
      auto abs_values = -(predicted_chip.abs()); // tensorFlow and custom versions

      // Temporary memory for computation
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> result(tmp_data, batch_size, layer_size);
      //auto result = ((expected_tensor - expected_tensor.constant(TensorT(1))) * predicted_chip.exp() + expected_tensor) / (predicted_chip.exp() + expected_tensor.constant(TensorT(1)));
      result.device(device) = (-predicted_chip * abs_values.exp() / ((abs_values.exp() + expected_tensor.constant(TensorT(1))) * predicted_chip.abs())) - expected_tensor.abs() + max_values;
      auto result_scale = result * error_tensor.chip(time_step, 1).constant(this->scale_);
      error_tensor.chip(time_step, 1).device(device) += (result_scale == result_scale).select(result_scale.clip(this->min_, this->max_), result_scale.constant(TensorT(0)));

      // Deallocate temporary memory
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
	class CrossEntropyWithLogitsLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
		std::string getName() { return "CrossEntropyWithLogitsLossTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> predicted_tensor(predicted, batch_size, memory_size, layer_size, 1, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1); // 4 dims
			auto exps = (predicted_chip.chip(0, 3) - predicted_chip.maximum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 3>({1, layer_size, 1}))).exp(); // 3 dims
			auto stable_softmax = exps.chip(0, 2) / exps.sum(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 2>({ 1, layer_size }));  // 2 dims

      // Temporary memory for computation
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> result(tmp_data, batch_size);
      result.device(device) = ((-expected_tensor * (stable_softmax.clip(this->eps_, TensorT(1)).log())) * expected_tensor.constant(TensorT(1) / TensorT(layer_size))).sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_);
			error_tensor.chip(time_step, 1).device(device) += (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));

      // Deallocate temporary memory
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
		@brief Softmax + Cross Entropy loss function gradient.

    See for derivations:
      https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
	*/
	template<typename TensorT, typename DeviceT>
	class CrossEntropyWithLogitsLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
	{
	public:
		using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
		std::string getName() { return "CrossEntropyWithLogitsLossGradTensorOp"; }
		void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
		{
      // Option 1: from derivation
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> expected_tensor(expected, batch_size, layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
			auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto expected_sum = expected_tensor.sum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, layer_size }));
      auto result = (((predicted_chip * expected_sum - expected_tensor.chip(0, 2)) / error_tensor.chip(time_step, 1).constant(TensorT(layer_size))) * error_tensor.chip(time_step, 1).constant(this->scale_));
			error_tensor.chip(time_step, 1).device(device) -= (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));
		};
	};

  /**
    @brief MSERangeUB Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeUBLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MSERangeUBLossTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse = ((expected_tensor - predicted_chip).pow(TensorT(2)) * expected_tensor.constant(TensorT(0.5)) / expected_tensor.constant(TensorT(layer_size)));        
      auto in_range = predicted_chip > expected_tensor;
      auto result = in_range.select(mse, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += (result.sum(Eigen::array<int, 1>({ 1 })) * error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

  /**
    @brief MSERangeUB Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeUBLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MSERangeUBLossGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse_grad = (((expected_tensor - predicted_chip) / expected_tensor.constant(TensorT(layer_size)))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
      auto in_range = predicted_chip > expected_tensor;
      auto result = in_range.select(mse_grad, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += result;
    };
  };

  /**
    @brief MSERangeLB Mean Squared Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeLBLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MSERangeLBLossTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse = ((expected_tensor - predicted_chip).pow(TensorT(2)) * expected_tensor.constant(TensorT(0.5)) / expected_tensor.constant(TensorT(layer_size)));
      auto in_range = predicted_chip < expected_tensor;
      auto result = in_range.select(mse, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += (result.sum(Eigen::array<int, 1>({ 1 }))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

  /**
    @brief MSERangeLB Mean Squared Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MSERangeLBLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MSERangeLBLossGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto mse_grad = (((expected_tensor - predicted_chip) / expected_tensor.constant(TensorT(layer_size)))
        *error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
      auto in_range = predicted_chip < expected_tensor;
      auto result = in_range.select(mse_grad, predicted_chip.constant((TensorT)0));

      error_tensor.chip(time_step, 1).device(device) += result;
    };
  };

  /**
    @brief KLDivergenceCat loss function.

    See implementation: https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py#L311
      KLD = alpha * log(alpha) + log(n) where n is the number of categories
      for predicted = log(alpha) as is the case here
      KLD = exp(predicted) * predicted + log(n)
  */
  template<typename TensorT, typename DeviceT>
  class KLDivergenceCatLossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
		KLDivergenceCatLossTensorOp() = default;
		KLDivergenceCatLossTensorOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionTensorOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceCatLossTensorOp() = default;
    std::string getName() { return "KLDivergenceCatLossTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);

      //auto neg_entropy = (predicted_chip * predicted_chip.log()).sum(Eigen::array<int, 1>({ 1 }));
      auto neg_entropy = (predicted_chip.exp() * predicted_chip).sum(Eigen::array<int, 1>({ 1 }));
      auto log_cat = error_tensor.chip(time_step, 1).constant(layer_size).log();
			auto kl_div_cap = neg_entropy + log_cat - (error_tensor.chip(time_step, 1).constant(this->capacity_)).cwiseMin(log_cat);
      auto result = kl_div_cap * error_tensor.chip(time_step, 1).constant(this->scale_);
      error_tensor.chip(time_step, 1).device(device) += (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));
    };
	private:
		TensorT capacity_ = TensorT(0);
  };

  /**
    @brief KLDivergenceCat  loss function gradient.

		d/dx of x*exp(x) + log(a) = (x+1)*exp(x)
  */
  template<typename TensorT, typename DeviceT>
  class KLDivergenceCatLossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
		KLDivergenceCatLossGradTensorOp() = default;
		KLDivergenceCatLossGradTensorOp(const TensorT & eps, const TensorT & scale, const TensorT & capacity) : LossFunctionGradTensorOp(eps, scale), capacity_(capacity) {};
		~KLDivergenceCatLossGradTensorOp() = default;
    std::string getName() { return "KLDivergenceCatLossGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
			auto kl_div = (predicted_chip + predicted_chip.constant(TensorT(1))) * predicted_chip.exp();
			auto log_cat = expected_tensor.constant(layer_size).log();
			auto kl_div_cap = kl_div - (error_tensor.chip(time_step, 1).constant(this->capacity_)).cwiseMin(log_cat);
      auto result = kl_div_cap * error_tensor.chip(time_step, 1).constant(this->scale_);
      // NOTE: changed to -= to ensure a negative gradient
      error_tensor.chip(time_step, 1).device(device) -= (result == result).select(result.clip(this->min_, this->max_), result.constant(TensorT(0)));
    };
	private:
		TensorT capacity_ = TensorT(0);
  };

  /**
    @brief MAPE Mean Absolute Percent Error loss function.
  */
  template<typename TensorT, typename DeviceT>
  class MAPELossTensorOp : public LossFunctionTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionTensorOp<TensorT, DeviceT>::LossFunctionTensorOp;
    std::string getName() { return "MAPELossTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_tensor(error, batch_size, memory_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto result = (expected_tensor == expected_tensor.constant(TensorT(0))).select(expected_tensor.constant(TensorT(0)),
        ((expected_tensor - predicted_chip) / expected_tensor).abs() / expected_tensor.constant(TensorT(layer_size))
      );
      error_tensor.chip(time_step, 1).device(device) += (result.sum(Eigen::array<int, 1>({ 1 }))*error_tensor.chip(time_step, 1).constant(this->scale_)).clip(this->min_, this->max_);
    };
  };

  /**
    @brief MAPE Mean Absolute Percent Error loss function gradient.
  */
  template<typename TensorT, typename DeviceT>
  class MAPELossGradTensorOp : public LossFunctionGradTensorOp<TensorT, DeviceT>
  {
  public:
    using LossFunctionGradTensorOp<TensorT, DeviceT>::LossFunctionGradTensorOp;
    std::string getName() { return "MAPELossGradTensorOp"; }
    void operator()(TensorT* predicted, TensorT* expected, TensorT* error, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> expected_tensor(expected, batch_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> predicted_tensor(predicted, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_tensor(error, batch_size, memory_size, layer_size);
      auto predicted_chip = predicted_tensor.chip(time_step, 1);
      auto result = (expected_tensor - predicted_chip) / (expected_tensor - predicted_chip).abs() / expected_tensor.abs() / expected_tensor.constant(TensorT(layer_size));
      auto result_selected = ((expected_tensor - predicted_chip) == expected_tensor.constant(TensorT(0)) || expected_tensor == expected_tensor.constant(TensorT(0))).select(
        expected_tensor.constant(TensorT(0)), result);
      error_tensor.chip(time_step, 1).device(device) += result_selected * expected_tensor.constant(this->scale_).clip(this->min_, this->max_);
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