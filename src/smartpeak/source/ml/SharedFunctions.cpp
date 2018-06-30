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
    Eigen::ThreadPool threadPool(n_threads); 
    Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_threads);

    // Scale the current output by the designated non-linearity
    // Scale the activated output by the time scale
    switch (node_type)
    {
      // no activation
      case NodeType::bias: {return net_input;} 
      case NodeType::input: {return net_input;} 
      case NodeType::hidden: {break;} 
      case NodeType::output: {break;} 
      default:
      {
        std::cout << "Node type not supported." << std::endl;
        return net_input;
      }
    }

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
    Eigen::ThreadPool threadPool(n_threads); 
    Eigen::ThreadPoolDevice threadPoolDevice(&threadPool, n_threads);

    switch (node_type)
    {
      case NodeType::bias:
      {
        derivative.setConstant(0.0);
        return derivative;
      } 
      case NodeType::input:
      {
        derivative.setConstant(0.0);
        return derivative;
      }   
      case NodeType::hidden: {break;}  
      case NodeType::output: {break;}   
      default:
      {
        std::cout << "Node type not supported." << std::endl;
        return output;
      }
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
      default:
      {
        std::cout << "Node activation not supported." << std::endl;
        return output;
      }
    }
    return derivative;
  }
}