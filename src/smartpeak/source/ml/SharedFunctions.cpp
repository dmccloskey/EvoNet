/**TODO:  Add copyright*/

#include <SmartPeak/ml/SharedFunctions.h>
#include <SmartPeak/ml/ActivationFunction.h>

#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{ 

  Eigen::Tensor<float, 1> calculateActivation(
    const NodeType& node_type, const NodeActivation& node_activation,
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
    switch(node_activation)
    {
      case NodeActivation::ReLU:
      {
        output.device(threadPoolDevice) = net_input.unaryExpr(ReLUOp<float>()) * dt;
        break;
      }
      case NodeActivation::ELU:
      {
        output.device(threadPoolDevice) = net_input.unaryExpr(ELUOp<float>(1.0)) * dt;
        break;
      }
      case NodeActivation::Sigmoid:
      {
        output.device(threadPoolDevice) = net_input.unaryExpr(SigmoidOp<float>()) * dt;
        break;
      }
      case NodeActivation::TanH:
      {
        output.device(threadPoolDevice) = net_input.unaryExpr(TanHOp<float>()) * dt;
        break;
      }
			case NodeActivation::Linear:
			{
				output.device(threadPoolDevice) = net_input.unaryExpr(LinearOp<float>()) * dt;
				break;
			}
			case NodeActivation::Inverse:
			{
				output.device(threadPoolDevice) = net_input.unaryExpr(InverseOp<float>()) * dt;
				break;
			}
			case NodeActivation::Exponential:
			{
				output.device(threadPoolDevice) = net_input.unaryExpr(ExponentialOp<float>()) * dt;
				break;
			}
      default:
      {
        std::cout << "Node activation not supported." << std::endl;
        return net_input;
      }
    }

    return output;
  }

  Eigen::Tensor<float, 1> calculateDerivative(
    const NodeType& node_type, const NodeActivation& node_activation,
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

    switch (node_activation)
    {       
      case NodeActivation::ReLU:
      {
        derivative.device(threadPoolDevice) = output.unaryExpr(ReLUGradOp<float>());
        break;
      }
      case NodeActivation::ELU:
      {
        derivative.device(threadPoolDevice) = output.unaryExpr(ELUGradOp<float>(1.0));
        break;
      } 
      case NodeActivation::Sigmoid:
      {
        derivative.device(threadPoolDevice) = output.unaryExpr(SigmoidGradOp<float>());
        break;
      } 
      case NodeActivation::TanH:
      {
        derivative.device(threadPoolDevice) = output.unaryExpr(TanHGradOp<float>());
        break;
      }
			case NodeActivation::Linear:
			{
				derivative.device(threadPoolDevice) = output.unaryExpr(LinearGradOp<float>());
				break;
			}
			case NodeActivation::Inverse:
			{
				derivative.device(threadPoolDevice) = output.unaryExpr(InverseGradOp<float>());
				break;
			}
			case NodeActivation::Exponential:
			{
				derivative.device(threadPoolDevice) = output.unaryExpr(ExponentialGradOp<float>());
				break;
			}
      default:
      {
        std::cout << "Node activation not supported." << std::endl;
        return output;
      }
    }
    return derivative;
  }
}