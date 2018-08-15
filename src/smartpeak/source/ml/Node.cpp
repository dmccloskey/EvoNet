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
		module_id_ = other.module_id_;
		module_name_ = other.module_name_;
    type_ = other.type_;
    status_ = other.status_;
    activation_ = other.activation_;
		activation_grad_ = other.activation_grad_;
		integration_ = other.integration_;
    output_min_ = other.output_min_;
    output_max_ = other.output_max_;
		input_ = other.input_;
    output_ = other.output_;
    error_ = other.error_;
    derivative_ = other.derivative_;
    dt_ = other.dt_;
  }

  Node::Node(const std::string& name, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status,
		const std::shared_ptr<ActivationOp<float>>& activation, const std::shared_ptr<ActivationOp<float>>& activation_grad,
		const SmartPeak::NodeIntegration& integration):
    name_(name), type_(type), status_(status), integration_(integration)
  {
		setActivation(activation);
		setActivationGrad(activation_grad);
  }

  //Node::Node(const int& id, const SmartPeak::NodeType& type,
  //  const SmartPeak::NodeStatus& status, const SmartPeak::NodeActivation& activation, const SmartPeak::NodeIntegration& integration):
  //  id_(id), type_(type), status_(status), activation_(activation), integration_(integration)
  //{
  //  if (name_ == "")
  //  {
  //    name_ = std::to_string(id);
  //  }
  //}

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

  void Node::setActivation(const std::shared_ptr<ActivationOp<float>>& activation)
  {
		activation_.reset();
		activation_ = std::move(activation);
  }
	std::shared_ptr<ActivationOp<float>> Node::getActivationShared() const
	{
		return activation_;
	}
	ActivationOp<float>*  Node::getActivation() const
  {
    return activation_.get();
  }

	void Node::setActivationGrad(const std::shared_ptr<ActivationOp<float>>& activation_grad)
	{
		activation_grad_.reset();
		activation_grad_ = std::move(activation_grad);
	}

	std::shared_ptr<ActivationOp<float>> Node::getActivationGradShared() const
	{
		return activation_grad_;
	}

	ActivationOp<float>* Node::getActivationGrad() const
	{
		return activation_grad_.get();
	}

	void Node::setIntegration(const SmartPeak::NodeIntegration& integration)
	{
		integration_ = integration;
	}
	SmartPeak::NodeIntegration Node::getIntegration() const
	{
		return integration_;
	}

	void Node::setModuleId(const int & module_id)
	{
		module_id_ = module_id;
	}

	int Node::getModuleId() const
	{
		return module_id_;
	}

	void Node::setModuleName(const std::string & module_name)
	{
		module_name_ = module_name;
	}

	std::string Node::getModuleName() const
	{
		return module_name_;
	}

	void Node::setInput(const Eigen::Tensor<float, 2>& input)
	{
		input_ = input;
	}
	Eigen::Tensor<float, 2> Node::getInput() const
	{
		return input_;
	}
	Eigen::Tensor<float, 2>* Node::getInputMutable()
	{
		return &input_;
	}
	float* Node::getInputPointer()
	{
		return input_.data();
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
		setInput(init_values);
    setError(init_values);
    setDerivative(init_values);

    init_values.setConstant(1.0f);
    setDt(init_values);
    
		// corections for specific node types
    if (type_ == NodeType::bias)
    {
      init_values.setConstant(1.0f);
      setStatus(NodeStatus::activated);
      setOutput(init_values);
			//setDerivative(init_values);
		}
		else if (type_ == NodeType::input)
		{
			//init_values.setConstant(1.0f);
			//setDerivative(init_values);
			init_values.setConstant(0.0f);
			setStatus(NodeStatus::initialized);
			setOutput(init_values);
		}
    else
    {
      init_values.setConstant(0.0f);
      setStatus(NodeStatus::initialized);
      setOutput(init_values);
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