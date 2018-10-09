/**TODO:  Add copyright*/

#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/ActivationFunction.h>

#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{
	template<typename TensorT>
	Node<TensorT>::Node(const Node<TensorT>& other)
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
		integration_error_ = other.integration_error_;
		integration_weight_grad_ = other.integration_weight_grad_;
		integration_ = other.integration_;
		output_min_ = other.output_min_;
		output_max_ = other.output_max_;
		node_data_ = other.node_data_;
		drop_probability_ = other.drop_probability_;
		drop_ = other.drop_;
	}

	template<typename TensorT>
	Node<TensorT>::Node(const std::string& name, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status,
		const std::shared_ptr<ActivationOp<TensorT>>& activation, const std::shared_ptr<ActivationOp<TensorT>>& activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& integration, const std::shared_ptr<IntegrationErrorOp<TensorT>>& integration_error, const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& integration_weight_grad):
    name_(name), type_(type), status_(status)
  {
		setActivation(activation);
		setActivationGrad(activation_grad);
		setIntegration(integration);
		setIntegrationError(integration_error);
		setIntegrationWeightGrad(integration_weight_grad);
  }

	template<typename TensorT>
  void Node<TensorT>::setId(const int& id)
  {
    id_ = id;
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }

	template<typename TensorT>
  int Node<TensorT>::getId() const
  {
    return id_;
  }

	template<typename TensorT>
  void Node<TensorT>::setName(const std::string& name)
  {
    name_ = name;    
  }
	template<typename TensorT>
  std::string Node<TensorT>::getName() const
  {
    return name_;
  }

	template<typename TensorT>
  void Node<TensorT>::setType(const SmartPeak::NodeType& type)
  {
    type_ = type;
  }
	template<typename TensorT>
  SmartPeak::NodeType Node<TensorT>::getType() const
  {
    return type_;
  }

	template<typename TensorT>
  void Node<TensorT>::setStatus(const SmartPeak::NodeStatus& status)
  {
    status_ = status;
  }
	template<typename TensorT>
  SmartPeak::NodeStatus Node<TensorT>::getStatus() const
  {
    return status_;
  }

	template<typename TensorT>
  void Node<TensorT>::setActivation(const std::shared_ptr<ActivationOp<TensorT>>& activation)
  {
		activation_.reset();
		activation_ = std::move(activation);
  }
	template<typename TensorT>
	std::shared_ptr<ActivationOp<TensorT>> Node<TensorT>::getActivationShared() const
	{
		return activation_;
	}
	template<typename TensorT>
	ActivationOp<TensorT>*  Node<TensorT>::getActivation() const
  {
    return activation_.get();
  }

	template<typename TensorT>
	void Node<TensorT>::setActivationGrad(const std::shared_ptr<ActivationOp<TensorT>>& activation_grad)
	{
		activation_grad_.reset();
		activation_grad_ = std::move(activation_grad);
	}

	template<typename TensorT>
	std::shared_ptr<ActivationOp<TensorT>> Node<TensorT>::getActivationGradShared() const
	{
		return activation_grad_;
	}

	template<typename TensorT>
	ActivationOp<TensorT>* Node<TensorT>::getActivationGrad() const
	{
		return activation_grad_.get();
	}

	template<typename TensorT>
	void Node<TensorT>::setIntegration(const std::shared_ptr<IntegrationOp<TensorT>>& integration)
	{
		integration_.reset();
		integration_ = std::move(integration);
	}
	template<typename TensorT>
	std::shared_ptr<IntegrationOp<TensorT>> Node<TensorT>::getIntegrationShared() const
	{
		return integration_;
	}
	template<typename TensorT>
	IntegrationOp<TensorT>*  Node<TensorT>::getIntegration() const
	{
		return integration_.get();
	}

	template<typename TensorT>
	void Node<TensorT>::setIntegrationError(const std::shared_ptr<IntegrationErrorOp<TensorT>>& integration_error)
	{
		integration_error_.reset();
		integration_error_ = std::move(integration_error);
	}
	template<typename TensorT>
	std::shared_ptr<IntegrationErrorOp<TensorT>> Node<TensorT>::getIntegrationErrorShared() const
	{
		return integration_error_;
	}
	template<typename TensorT>
	IntegrationErrorOp<TensorT>*  Node<TensorT>::getIntegrationError() const
	{
		return integration_error_.get();
	}

	template<typename TensorT>
	void Node<TensorT>::setIntegrationWeightGrad(const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& integration_weight_grad)
	{
		integration_weight_grad_.reset();
		integration_weight_grad_ = std::move(integration_weight_grad);
	}
	template<typename TensorT>
	std::shared_ptr<IntegrationWeightGradOp<TensorT>> Node<TensorT>::getIntegrationWeightGradShared() const
	{
		return integration_weight_grad_;
	}
	template<typename TensorT>
	IntegrationWeightGradOp<TensorT>*  Node<TensorT>::getIntegrationWeightGrad() const
	{
		return integration_weight_grad_.get();
	}

	template<typename TensorT>
	void Node<TensorT>::setModuleId(const int & module_id)
	{
		module_id_ = module_id;
	}

	template<typename TensorT>
	int Node<TensorT>::getModuleId() const
	{
		return module_id_;
	}

	template<typename TensorT>
	void Node<TensorT>::setModuleName(const std::string & module_name)
	{
		module_name_ = module_name;
	}

	template<typename TensorT>
	std::string Node<TensorT>::getModuleName() const
	{
		return module_name_;
	}

	template<typename TensorT>
  void Node<TensorT>::setOutputMin(const TensorT& output_min)
  {
    output_min_ = output_min;
  }
	template<typename TensorT>
  void Node<TensorT>::setOutputMax(const TensorT& output_max)
  {
    output_max_ = output_max;
  }

	template<typename TensorT>
	void Node<TensorT>::setDropProbability(const TensorT & drop_probability)
	{
		drop_probability_ = drop_probability;
	}

	template<typename TensorT>
	TensorT Node<TensorT>::getDropProbability() const
	{
		return drop_probability_;
	}

	template<typename TensorT>
	void Node<TensorT>::setDrop(const Eigen::Tensor<TensorT, 2>& drop)
	{
		drop_ = drop;
	}
	template<typename TensorT>
	Eigen::Tensor<TensorT, 2> Node<TensorT>::getDrop() const
	{
		return drop_;
	}

	template<typename TensorT>
	size_t Node<TensorT>::getBatchSize() const
	{
		return node_data_.getBatchSize();
	}

	template<typename TensorT>
	size_t Node<TensorT>::getMemorySize() const
	{
		return node_data_.getMemorySize();
	}

	template<typename TensorT>
	void Node<TensorT>::setNodeData(const std::shared_ptr<NodeData<TensorT>>& node_data)
	{
		node_data_.reset(std::move(node_data));
	}

	template<typename TensorT>
	std::shared_ptr<NodeData<TensorT>> Node<TensorT>::getNodeData()
	{
		return node_data_;
	}

	template<typename TensorT>
  void Node<TensorT>::initNode(const int& batch_size, const int& memory_size, bool train)
  {

#ifndef EVONET_CUDA
		node_data_.reset(new NodeDataGpu<TensorT>());
#else
		node_data_.reset(new NodeDataCpu<TensorT>());
#endif
		node_data_.getBatchSize(batch_size);
		node_data_.getMemorySize(memory_size);

		// Template zero and one tensor
    Eigen::Tensor<TensorT, 2> zero_values(batch_size, memory_size); zero_values.setConstant(0);
		Eigen::Tensor<TensorT, 2> one_values(batch_size, memory_size); one_values.setConstant(1);
		Eigen::Tensor<TensorT, 2> init_values;

		// set the input, error, and derivatives
		init_values = zero_values;
		node_data_->setInput(init_values.data());
		init_values = zero_values;
		node_data_->setError(init_values.data());
		init_values = zero_values;
		node_data_->setDerivative(init_values.data());

		// set Dt
		init_values = one_values;
		node_data_->setDt(init_values.data());

		// set Drop probabilities [TODO: broke when adding NodeData...]
		if (train) {
			init_values.unaryExpr(RandBinaryOp<TensorT>(getDropProbability()));
			init_values = one_values;
			setDrop(one_values);
		}
		else {
			init_values = one_values;
			setDrop(one_values);
		}
    
		// corections for specific node types
    if (type_ == NodeType::bias)
    {
      setStatus(NodeStatus::activated);
			init_values = one_values;
			node_data_->setOutput(init_values.data());
		}
		else if (type_ == NodeType::input)
		{
			setStatus(NodeStatus::initialized);
			init_values = zero_values;
			node_data_->setOutput(init_values.data());
		}
		else if (type_ == NodeType::zero)
		{
			setStatus(NodeStatus::activated);
			init_values = zero_values;
			node_data_->setOutput(init_values.data());
		}
    else
    {
      setStatus(NodeStatus::initialized);
			init_values = zero_values;
			node_data_->setOutput(init_values.data());
    }
  }

	template<typename TensorT>
  bool Node<TensorT>::checkTimeStep(const int&time_step)
  {
    if (time_step < 0)
    {
      std::cout << "time_step is less than 0." << std::endl;
      return false;
    }
    else if (time_step >= memory_size_)
    {
      std::cout << "time_step is greater than the node memory_size." << std::endl;
      return false;
    }
    else
    {
      return true;
    }
  }

	template<typename TensorT>
  void Node<TensorT>::checkOutput()
  {
    for (int i=0; i<batch_size_; ++i)
    {
      for (int j=0; j<memory_size_ ; ++j)
      {
        if (node_data_->getOutput()(i,j) < output_min_)
					node_data_->getOutput()(i,j) = output_min_;
        else if (node_data_->getOutput()(i,j) > output_max_)
					node_data_->getOutput()(i,j) = output_max_;
      }
    }
  }
}