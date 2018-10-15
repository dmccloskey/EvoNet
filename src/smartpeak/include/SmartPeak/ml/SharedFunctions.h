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
}
#endif //SMARTPEAK_SHAREDFUNCTIONS_H