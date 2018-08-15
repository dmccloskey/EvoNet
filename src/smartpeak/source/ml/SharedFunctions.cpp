/**TODO:  Add copyright*/

#include <SmartPeak/ml/SharedFunctions.h>
#include <SmartPeak/ml/ActivationFunction.h>

#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{ 

  Eigen::Tensor<float, 1> calculateActivation(
    ActivationOp<float>* node_activation,
    const Eigen::Tensor<float, 1>& net_input, const Eigen::Tensor<float, 1>& dt,
    int n_threads)
  {
    Eigen::Tensor<float, 1> output(net_input.dimension(0));
    Eigen::DefaultDevice threadPoolDevice;
    if (n_threads > 1)
    {  // transfer incurs a significant cost
       // create the ThreadPoolDevice only if > 1 threads are available!
      Eigen::ThreadPool threadPool(n_threads); 
      Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_threads);
    }

    // Scale the current output by the designated non-linearity
    // Scale the activated output by the time scale
		output.device(threadPoolDevice) = net_input.unaryExpr(FunctorOp<float>(node_activation)) * dt;

    return output;
  }

  Eigen::Tensor<float, 1> calculateDerivative(
    ActivationOp<float>* node_activation_grad,
    const Eigen::Tensor<float, 1>& output,
    int n_threads)
  {
    Eigen::Tensor<float, 1> derivative(output.dimension(0));
    Eigen::DefaultDevice threadPoolDevice;
    if (n_threads > 1)
    {  // transfer incurs a significant cost
       // create the ThreadPoolDevice only if > 1 threads are available!
      Eigen::ThreadPool threadPool(n_threads); 
      Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_threads);
    }

		derivative.device(threadPoolDevice) = output.unaryExpr(FunctorOp<float>(node_activation_grad));
    return derivative;
  }
}