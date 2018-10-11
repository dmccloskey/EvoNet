/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SHAREDFUNCTIONS_H
#define SMARTPEAK_SHAREDFUNCTIONS_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <vector>
#include <limits>
#include <random>
#include <cmath>
#include <iostream>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>


namespace SmartPeak
{
	/**
	@brief Functor for use with calculate activation/derivative.
	*/
	template<typename T>
	class FunctorOp
	{
	public:
		FunctorOp() {};
		FunctorOp(ActivationOp<T>* node_activation) : activation_(node_activation) {};
		~FunctorOp() {};
		T operator()(const T& x_I) const {
			return (*activation_)(x_I);
		}
	private:
		ActivationOp<T>* activation_;
	};
  /**
  @brief The current output is passed through an activation function.
  Contents are updated in place.

  @param[in] time_step Time step to activate all samples in the batch

  [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
  */
	template<typename TensorT>
  Eigen::Tensor<TensorT, 1> calculateActivation(
    ActivationOp<TensorT>* node_activation,
    const Eigen::Tensor<TensorT, 1>& net_input, const Eigen::Tensor<TensorT, 1>& dt,
    int n_threads = 1)
	{
		Eigen::Tensor<TensorT, 1> output(net_input.dimension(0));
		Eigen::DefaultDevice threadPoolDevice;
		if (n_threads > 1)
		{  // transfer incurs a significant cost
			 // create the ThreadPoolDevice only if > 1 threads are available!
			Eigen::ThreadPool threadPool(n_threads);
			Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_threads);
		}

		// Scale the current output by the designated non-linearity
		// Scale the activated output by the time scale
		output.device(threadPoolDevice) = net_input.unaryExpr(FunctorOp<TensorT>(node_activation)) * dt;

		return output;
	};

  /**
  @brief Calculate the derivative from the output.

  @param[in] time_step Time step to calculate the derivative
  for all samples in the batch

  [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
  */
	template<typename TensorT>
  Eigen::Tensor<TensorT, 1> calculateDerivative(
    ActivationOp<TensorT>* node_activation_grad,
    const Eigen::Tensor<TensorT, 1>& output,
    int n_threads = 1)
	{
		Eigen::Tensor<TensorT, 1> derivative(output.dimension(0));
		Eigen::DefaultDevice threadPoolDevice;
		if (n_threads > 1)
		{  // transfer incurs a significant cost
			 // create the ThreadPoolDevice only if > 1 threads are available!
			Eigen::ThreadPool threadPool(n_threads);
			Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_threads);
		}

		derivative.device(threadPoolDevice) = output.unaryExpr(FunctorOp<TensorT>(node_activation_grad));
		return derivative;
	};

	/**
	@brief Replaces NaN and Inf with 0

	@param[in] time_step Time step to calculate the derivative
	for all samples in the batch

	[THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
	*/

	template<typename T>
	T checkNan(
		const T& x)
	{
		if (std::isnan(x))
			return T(0);
		else
			return x;
	}

	template<typename T>
	T substituteNanInf(const T& x)
	{
		if (x == std::numeric_limits<T>::infinity())
		{
			return T(1e9);
		}			
		else if (x == -std::numeric_limits<T>::infinity())
		{
			return T(-1e9);
		}			
		else if (std::isnan(x))
		{
			return T(0);
		}			
		else
		{
			return x;
		}			
	}

	/**
	@brief Clip
	*/
	template<typename T>
	class ClipOp
	{
	public:
		ClipOp() = default;
		ClipOp(const T& eps, const T& min, const T& max) : eps_(eps), min_(min), max_(max) {};
		~ClipOp() = default;
		T operator()(const T& x) const {
			if (x < min_ + eps_)
				return min_ + eps_;
			else if (x > max_ - eps_)
				return max_ - eps_;
			else
				return x;
		}
	private:
		T eps_ = 1e-12; ///< threshold to clip between min and max
		T min_ = 0;
		T max_ = 1;
	};

	/**
	@brief return x or 0 with a specified probability
	*/
	template<typename T>
	class RandBinaryOp
	{
	public:
		RandBinaryOp() = default;
		RandBinaryOp(const T& p) : p_(p) {};
		~RandBinaryOp() = default;
		T operator()(const T& x) const {
			std::random_device rd;
			std::mt19937 gen(rd());
			std::discrete_distribution<> distrib({ p_, 1 - p_ });
			return x*distrib(gen);
		}
	private:
		T p_ = 1; ///< probablity of 0
	};
}
#endif //SMARTPEAK_SHAREDFUNCTIONS_H