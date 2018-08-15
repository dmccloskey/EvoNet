/**TODO:  Add copyright*/

#ifndef SMARTPEAK_SHAREDFUNCTIONS_H
#define SMARTPEAK_SHAREDFUNCTIONS_H

#include <SmartPeak/ml/Node.h>

#include <vector>
#include <limits>

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
  Eigen::Tensor<float, 1> calculateActivation(
    ActivationOp<float>* node_activation,
    const Eigen::Tensor<float, 1>& net_input, const Eigen::Tensor<float, 1>& dt,
    int n_threads = 1);

  /**
  @brief Calculate the derivative from the output.

  @param[in] time_step Time step to calculate the derivative
  for all samples in the batch

  [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
  */
  Eigen::Tensor<float, 1> calculateDerivative(
    ActivationOp<float>* node_activation_grad,
    const Eigen::Tensor<float, 1>& output,
    int n_threads = 1);

	/**
	@brief Replaces NaN and Inf with 0

	@param[in] time_step Time step to calculate the derivative
	for all samples in the batch

	[THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
	*/

	template<typename T>
	T checkNanInf(
		const T& x)
	{
		if (std::isinf(x) || std::isnan(x))
			return T(0);
		else
			return x;
	}

	template<typename T>
	T substituteNanInf(const T& x)
	{
		if (x == std::numeric_limits<T>::infinity())
		{
			return T(1e24);
		}			
		else if (x == -std::numeric_limits<T>::infinity())
		{
			return T(-1e24);
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
}
#endif //SMARTPEAK_SHAREDFUNCTIONS_H