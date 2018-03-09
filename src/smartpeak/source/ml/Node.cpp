/**TODO:  Add copyright*/

#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Operation.h>

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

  void Node::setOutput(const Eigen::Tensor<float, 1>& output)
  {
    output_ = output;
  }
  Eigen::Tensor<float, 1> Node::getOutput() const
  {
    return output_;
  }
  float* Node::getOutputPointer()
  {
    return output_.data();
  }

  void Node::setError(const Eigen::Tensor<float, 1>& error)
  {
    error_ = error;
  }
  Eigen::Tensor<float, 1> Node::getError() const
  {
    return error_;
  }
  float* Node::getErrorPointer()
  {
    return error_.data();
  }

  void Node::setDerivative(const Eigen::Tensor<float, 1>& derivative)
  {
    derivative_ = derivative;
  }
  Eigen::Tensor<float, 1> Node::getDerivative() const
  {
    return derivative_;
  }
  float* Node::getDerivativePointer()
  {
    return derivative_.data();
  }

  void Node::calculateActivation()
  {
    switch (type_)
    {
      case NodeType::bias:
        break;
      case NodeType::input:
        break;
      case NodeType::ReLU:
        ReLUOp<float> operation;
        output_.unaryExpr(operation);
        break;
      case NodeType::ELU:
        ELUOp<float> operation(1.0);
        output_.unaryExpr(operation);
        break;
      default:
        std::cout << "Node type not supported." << std::endl;
        break;
    }
  }
}