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

  Node::Node(const Node& other)
  {    
    id_ = other.id_;
    name_ = other.name_;
    type_ = other.type_;
    status_ = other.status_;
    activation_ = other.activation_;
    output_min_ = other.output_min_;
    output_max_ = other.output_max_;
    output_ = other.output_;
    error_ = other.error_;
    derivative_ = other.derivative_;
    dt_ = other.dt_;
  }

  Node::Node(const std::string& name, const SmartPeak::NodeType& type,
    const SmartPeak::NodeStatus& status, const SmartPeak::NodeActivation& activation):
    name_(name), type_(type), status_(status), activation_(activation)
  {
  }

  Node::Node(const int& id, const SmartPeak::NodeType& type,
    const SmartPeak::NodeStatus& status, const SmartPeak::NodeActivation& activation):
    id_(id), type_(type), status_(status), activation_(activation)
  {
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }

  Node::~Node()
  {
  }
  
  void Node::setId(const int& id)
  {
    id_ = id;
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }
  int Node::getId() const
  {
    return id_;
  }
  
  void Node::setName(const std::string& name)
  {
    name_ = name;    
  }
  std::string Node::getName() const
  {
    return name_;
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

  void Node::setActivation(const SmartPeak::NodeActivation& activation)
  {
    activation_ = activation;
  }
  SmartPeak::NodeActivation Node::getActivation() const
  {
    return activation_;
  }

  void Node::setOutput(const Eigen::Tensor<float, 2>& output)
  {
    output_ = output;
    checkOutput();
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

  void Node::setDt(const Eigen::Tensor<float, 2>& dt)
  {
    dt_ = dt;
  }
  Eigen::Tensor<float, 2> Node::getDt() const
  {
    return dt_;
  }
  Eigen::Tensor<float, 2>* Node::getDtMutable()
  {
    return &dt_;
  }
  float* Node::getDtPointer()
  {
    return dt_.data();
  }

  void Node::setOutputMin(const float& output_min)
  {
    output_min_ = output_min;
  }
  void Node::setOutputMax(const float& output_max)
  {
    output_max_ = output_max;
  }

  void Node::initNode(const int& batch_size, const int& memory_size)
  {
    Eigen::Tensor<float, 2> init_values(batch_size, memory_size);
    init_values.setConstant(0.0f);
    setError(init_values);
    setDerivative(init_values);

    init_values.setConstant(1.0f);
    setDt(init_values);
    
    if (type_ == NodeType::bias)
    {
      init_values.setConstant(1.0f);
      setStatus(NodeStatus::activated);
      setOutput(init_values);
    }
    else
    {
      init_values.setConstant(0.0f);
      setStatus(NodeStatus::initialized);
      setOutput(init_values);
    }

  }

  void Node::calculateActivation(const int& time_step)
  {
    if (!checkTimeStep(time_step)) return;
    Eigen::Tensor<float, 1> output_step = output_.chip(time_step, 1);

    // Scale the current output by the designated non-linearity
    // Scale the activated output by the time scale
    switch (type_)
    {
      // no activation
      case NodeType::bias: {return;} 
      case NodeType::input: {return;} 
      case NodeType::hidden: {break;} 
      case NodeType::output: {break;} 
      default:
      {
        std::cout << "Node type not supported." << std::endl;
        return;
      }
    }

    switch(activation_)
    {
      case NodeActivation::ReLU:
      {
        output_step = output_step.unaryExpr(ReLUOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          output_(i, time_step) = output_step(i) * dt_(i, time_step);
        }
        break;
      }
      case NodeActivation::ELU:
      {
        output_step = output_step.unaryExpr(ELUOp<float>(1.0));
        for (int i=0; i<output_step.size(); ++i)
        {
          output_(i, time_step) = output_step(i) * dt_(i, time_step);
        }
        break;
      }
      case NodeActivation::Sigmoid:
      {
        output_step = output_step.unaryExpr(SigmoidOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          output_(i, time_step) = output_step(i) * dt_(i, time_step);
        }
        break;
      }
      case NodeActivation::TanH:
      {
        output_step = output_step.unaryExpr(TanHOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          output_(i, time_step) = output_step(i) * dt_(i, time_step);
        }
        break;
      }
			case NodeActivation::Linear:
			{
				output_step = output_step.unaryExpr(LinearOp<float>());
				for (int i = 0; i<output_step.size(); ++i)
				{
					output_(i, time_step) = output_step(i) * dt_(i, time_step);
				}
				break;
			}
      default:
      {
        std::cout << "Node activation not supported." << std::endl;
        return;
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
        return;
      } 
      case NodeType::input:
      {
        output_step.setConstant(0.0);
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        return;
      }   
      case NodeType::hidden: {break;}  
      case NodeType::output: {break;}   
      default:
      {
        std::cout << "Node type not supported." << std::endl;
        return;
      }
    }

    switch (activation_)
    {       
      case NodeActivation::ReLU:
      {
        output_step = output_step.unaryExpr(ReLUGradOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      }
      case NodeActivation::ELU:
      {
        output_step = output_step.unaryExpr(ELUGradOp<float>(1.0));
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      } 
      case NodeActivation::Sigmoid:
      {
        output_step = output_step.unaryExpr(SigmoidGradOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      } 
      case NodeActivation::TanH:
      {
        output_step = output_step.unaryExpr(TanHGradOp<float>());
        for (int i=0; i<output_step.size(); ++i)
        {
          derivative_(i, time_step) = output_step(i);
        }
        break;
      }
			case NodeActivation::Linear:
			{
				output_step = output_step.unaryExpr(LinearGradOp<float>());
				for (int i = 0; i<output_step.size(); ++i)
				{
					derivative_(i, time_step) = output_step(i);
				}
				break;
			}
      default:
      {
        std::cout << "Node activation not supported." << std::endl;
        return;
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

  void Node::saveCurrentDt()
  {
    const int batch_size = dt_.dimension(0);
    const int memory_size = dt_.dimension(1);
    for (int i=0; i<batch_size; ++i)
    {
      for (int j=memory_size-1; j>=0 ; --j)
      {
        if (j==0)
        {
          dt_(i, j) = 0.0;
        }
        else
        {
          dt_(i, j) = dt_(i, j-1);
        }
      }
    }
  }

  void Node::checkOutput()
  {
    const int batch_size = derivative_.dimension(0);
    const int memory_size = derivative_.dimension(1);
    for (int i=0; i<batch_size; ++i)
    {
      for (int j=0; j<memory_size ; ++j)
      {
        if (output_(i,j) < output_min_)
          output_(i,j) = output_min_;
        else if (output_(i,j) > output_max_)
          output_(i,j) = output_max_;
      }
    }
  }
}