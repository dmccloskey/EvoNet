/**TODO:  Add copyright*/

#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/ActivationFunction.h>

#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{
  Node::Node()
  {        
  }

  Node::Node(const int& id, const SmartPeak::NodeType& type,
    const SmartPeak::NodeStatus& status):
    id_(id), type_(type), status_(status)
  {
  }

  Node::~Node()
  {
  }
  
  void Node::setId(const int& id)
  {
    id_ = id;
  }
  int Node::getId() const
  {
    return id_;
  }

  void Node::setType(const SmartPeak::NodeType& type)
  {
    type_ = type;
  }
  SmartPeak::NodeType Node::getType() const
  {
    return type_;
  }

  void Node::setStatus(const SmartPeak::NodeStatus& status)
  {
    status_ = status;
  }
  SmartPeak::NodeStatus Node::getStatus() const
  {
    return status_;
  }

  void Node::setOutput(const Eigen::Tensor<float, 2>& output)
  {
    output_ = output;
  }
  Eigen::Tensor<float, 2> Node::getOutput() const
  {
    return output_;
  }
  Eigen::Tensor<float, 2>* Node::getOutputMutable()
  {
    return &output_;
  }
  float* Node::getOutputPointer()
  {
    return output_.data();
  }

  void Node::setError(const Eigen::Tensor<float, 2>& error)
  {
    error_ = error;
  }
  Eigen::Tensor<float, 2> Node::getError() const
  {
    return error_;
  }
  Eigen::Tensor<float, 2>* Node::getErrorMutable()
  {
    return &error_;
  }
  float* Node::getErrorPointer()
  {
    return error_.data();
  }

  void Node::setDerivative(const Eigen::Tensor<float, 2>& derivative)
  {
    derivative_ = derivative;
  }
  Eigen::Tensor<float, 2> Node::getDerivative() const
  {
    return derivative_;
  }
  Eigen::Tensor<float, 2>* Node::getDerivativeMutable()
  {
    return &derivative_;
  }
  float* Node::getDerivativePointer()
  {
    return derivative_.data();
  }

  void Node::initNode(const int& batch_size, const int& memory_size)
  {
    Eigen::Tensor<float, 2> init_values(batch_size, memory_size);
    init_values.setConstant(0.0f);
    setOutput(init_values);
    setError(init_values);
    setDerivative(init_values);
    setStatus(NodeStatus::initialized);
  }

  void Node::calculateActivation(const int& time_step)
  {
    if (!checkTimeStep(time_step)) return;
    Eigen::Tensor<float, 1> output_step = output_.chip(time_step, 1);
    switch (type_)
    {
      case NodeType::bias: {break;} 
      case NodeType::input: {break;}        
      case NodeType::ReLU:
      {
        output_step = output_step.unaryExpr(ReLUOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          output_(i, time_step) = output_step(i);
        }
        break;
      }
      case NodeType::ELU:
      {
        output_step = output_step.unaryExpr(ELUOp<float>(1.0));
        for (int i=0; i<output_step.size(); ++i)
        {
          output_(i, time_step) = output_step(i);
        }
        break;
      }
      default:
      {
        std::cout << "Node type not supported." << std::endl;
        break;
      }
    }
  }

  void Node::calculateDerivative(const int& time_step)
  {
    if (!checkTimeStep(time_step)) return;
    Eigen::Tensor<float, 1> output_step = output_.chip(time_step, 1);
    switch (type_)
    {
      case NodeType::bias:
      {
        output_step.setConstant(0.0);
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      } 
      case NodeType::input:
      {
        output_step.setConstant(0.0);
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      }        
      case NodeType::ReLU:
      {
        output_step = output_step.unaryExpr(ReLUGradOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      }
      case NodeType::ELU:
      {
        output_step = output_step.unaryExpr(ELUGradOp<float>(1.0));
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      }
      default:
      {
        std::cout << "Node type not supported." << std::endl;
        break;
      }
    }
  }

  bool Node::checkTimeStep(const int&time_step)
  {
    const int memory_size = output_.dimension(1);
    if (time_step < 0)
    {
      std::cout << "time_step is less than 0." << std::endl;
      return false;
    }
    else if (time_step >= memory_size)
    {
      std::cout << "time_step is greater than the node memory_size." << std::endl;
      return false;
    }
    else
    {
      return true;
    }
  }

  void Node::saveCurrentOutput()
  {
    const int batch_size = output_.dimension(0);
    const int memory_size = output_.dimension(1);
    for (int i=0; i<batch_size; ++i)
    {
      for (int j=memory_size-1; j>=0 ; --j)
      {
        if (j==0)
        {
          output_(i, j) = 0.0;
        }
        else
        {
          output_(i, j) = output_(i, j-1);
        }
      }
    }
  }

  void Node::saveCurrentDerivative()
  {
    const int batch_size = derivative_.dimension(0);
    const int memory_size = derivative_.dimension(1);
    for (int i=0; i<batch_size; ++i)
    {
      for (int j=memory_size-1; j>=0 ; --j)
      {
        if (j==0)
        {
          derivative_(i, j) = 0.0;
        }
        else
        {
          derivative_(i, j) = derivative_(i, j-1);
        }
      }
    }
  }

  void Node::saveCurrentError()
  {
    const int batch_size = error_.dimension(0);
    const int memory_size = error_.dimension(1);
    for (int i=0; i<batch_size; ++i)
    {
      for (int j=memory_size-1; j>=0 ; --j)
      {
        if (j==0)
        {
          error_(i, j) = 0.0;
        }
        else
        {
          error_(i, j) = error_(i, j-1);
        }
      }
    }
  }
}