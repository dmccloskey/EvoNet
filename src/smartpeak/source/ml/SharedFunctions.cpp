/**TODO:  Add copyright*/

#include <SmartPeak/ml/SharedFunctions.h>
#include <SmartPeak/ml/ActivationFunction.h>

#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{ 

	template<typename TensorT>
  Eigen::Tensor<TensorT, 1> calculateActivation(
    ActivationOp<TensorT>* node_activation,
    const Eigen::Tensor<TensorT, 1>& net_input, const Eigen::Tensor<TensorT, 1>& dt,
    int n_threads)
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
  }

	template<typename TensorT>
  Eigen::Tensor<TensorT, 1> calculateDerivative(
    ActivationOp<TensorT>* node_activation_grad,
    const Eigen::Tensor<TensorT, 1>& output,
    int n_threads)
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
  }
}