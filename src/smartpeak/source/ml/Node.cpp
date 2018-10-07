/**TODO:  Add copyright*/

#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/ActivationFunction.h>

#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{
	template<typename HDelT, typename DDelT, typename TensorT>
	Node<HDelT, DDelT, TensorT>::Node(const Node<HDelT, DDelT, TensorT>& other)
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

	template<typename HDelT, typename DDelT, typename TensorT>
	Node<HDelT, DDelT, TensorT>::Node(const std::string& name, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status,
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

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::setId(const int& id)
  {
    id_ = id;
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  int Node<HDelT, DDelT, TensorT>::getId() const
  {
    return id_;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::setName(const std::string& name)
  {
    name_ = name;    
  }
	template<typename HDelT, typename DDelT, typename TensorT>
  std::string Node<HDelT, DDelT, TensorT>::getName() const
  {
    return name_;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::setType(const SmartPeak::NodeType& type)
  {
    type_ = type;
  }
	template<typename HDelT, typename DDelT, typename TensorT>
  SmartPeak::NodeType Node<HDelT, DDelT, TensorT>::getType() const
  {
    return type_;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::setStatus(const SmartPeak::NodeStatus& status)
  {
    status_ = status;
  }
	template<typename HDelT, typename DDelT, typename TensorT>
  SmartPeak::NodeStatus Node<HDelT, DDelT, TensorT>::getStatus() const
  {
    return status_;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::setActivation(const std::shared_ptr<ActivationOp<TensorT>>& activation)
  {
		activation_.reset();
		activation_ = std::move(activation);
  }
	template<typename HDelT, typename DDelT, typename TensorT>
	std::shared_ptr<ActivationOp<TensorT>> Node<HDelT, DDelT, TensorT>::getActivationShared() const
	{
		return activation_;
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	ActivationOp<TensorT>*  Node<HDelT, DDelT, TensorT>::getActivation() const
  {
    return activation_.get();
  }

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setActivationGrad(const std::shared_ptr<ActivationOp<TensorT>>& activation_grad)
	{
		activation_grad_.reset();
		activation_grad_ = std::move(activation_grad);
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	std::shared_ptr<ActivationOp<TensorT>> Node<HDelT, DDelT, TensorT>::getActivationGradShared() const
	{
		return activation_grad_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	ActivationOp<TensorT>* Node<HDelT, DDelT, TensorT>::getActivationGrad() const
	{
		return activation_grad_.get();
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setIntegration(const std::shared_ptr<IntegrationOp<TensorT>>& integration)
	{
		integration_.reset();
		integration_ = std::move(integration);
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	std::shared_ptr<IntegrationOp<TensorT>> Node<HDelT, DDelT, TensorT>::getIntegrationShared() const
	{
		return integration_;
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	IntegrationOp<TensorT>*  Node<HDelT, DDelT, TensorT>::getIntegration() const
	{
		return integration_.get();
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setIntegrationError(const std::shared_ptr<IntegrationErrorOp<TensorT>>& integration_error)
	{
		integration_error_.reset();
		integration_error_ = std::move(integration_error);
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	std::shared_ptr<IntegrationErrorOp<TensorT>> Node<HDelT, DDelT, TensorT>::getIntegrationErrorShared() const
	{
		return integration_error_;
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	IntegrationErrorOp<TensorT>*  Node<HDelT, DDelT, TensorT>::getIntegrationError() const
	{
		return integration_error_.get();
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setIntegrationWeightGrad(const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& integration_weight_grad)
	{
		integration_weight_grad_.reset();
		integration_weight_grad_ = std::move(integration_weight_grad);
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	std::shared_ptr<IntegrationWeightGradOp<TensorT>> Node<HDelT, DDelT, TensorT>::getIntegrationWeightGradShared() const
	{
		return integration_weight_grad_;
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	IntegrationWeightGradOp<TensorT>*  Node<HDelT, DDelT, TensorT>::getIntegrationWeightGrad() const
	{
		return integration_weight_grad_.get();
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setModuleId(const int & module_id)
	{
		module_id_ = module_id;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int Node<HDelT, DDelT, TensorT>::getModuleId() const
	{
		return module_id_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setModuleName(const std::string & module_name)
	{
		module_name_ = module_name;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	std::string Node<HDelT, DDelT, TensorT>::getModuleName() const
	{
		return module_name_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::setOutputMin(const TensorT& output_min)
  {
    output_min_ = output_min;
  }
	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::setOutputMax(const TensorT& output_max)
  {
    output_max_ = output_max;
  }

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setDropProbability(const TensorT & drop_probability)
	{
		drop_probability_ = drop_probability;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	TensorT Node<HDelT, DDelT, TensorT>::getDropProbability() const
	{
		return drop_probability_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setDrop(const Eigen::Tensor<TensorT, 2>& drop)
	{
		drop_ = drop;
	}
	template<typename HDelT, typename DDelT, typename TensorT>
	Eigen::Tensor<TensorT, 2> Node<HDelT, DDelT, TensorT>::getDrop() const
	{
		return drop_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int Node<HDelT, DDelT, TensorT>::getBatchSize() const
	{
		return batch_size_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	int Node<HDelT, DDelT, TensorT>::getMemorySize() const
	{
		return memory_size_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	void Node<HDelT, DDelT, TensorT>::setNodeData(const std::shared_ptr<NodeData<HDelT, DDelT, TensorT>>& node_data)
	{
		node_data_.reset();
		node_data_ = std::move(node_data);
	}

	template<typename HDelT, typename DDelT, typename TensorT>
	std::shared_ptr<NodeData<HDelT, DDelT, TensorT>> Node<HDelT, DDelT, TensorT>::getNodeData()
	{
		return node_data_;
	}

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::initNode(const int& batch_size, const int& memory_size, bool train)
  {
		batch_size_ = batch_size;
		memory_size_ = memory_size;

    Eigen::Tensor<TensorT, 2> init_values(batch_size, memory_size);
    init_values.setConstant(0.0f);
		node_data_->setInput(init_values);
		node_data_->setError(init_values);
		node_data_->setDerivative(init_values);

		// set Dt
    init_values.setConstant(1.0f);
		node_data_->setDt(init_values);

		// set Drop probabilities
		if (train) {
			init_values.unaryExpr(RandBinaryOp<TensorT>(getDropProbability()));
			setDrop(init_values);
		}
		else {
			setDrop(init_values);
		}
    
		// corections for specific node types
    if (type_ == NodeType::bias)
    {
      init_values.setConstant(1.0f);
      setStatus(NodeStatus::activated);
			node_data_->setOutput(init_values);
			//node_data_->setDerivative(init_values);
		}
		else if (type_ == NodeType::input)
		{
			//init_values.setConstant(1.0f);
			//node_data_->setDerivative(init_values);
			init_values.setConstant(0.0f);
			setStatus(NodeStatus::initialized);
			node_data_->setOutput(init_values);
		}
		else if (type_ == NodeType::zero)
		{
			//init_values.setConstant(1.0f);
			//setDerivative(init_values);
			init_values.setConstant(0.0f);
			setStatus(NodeStatus::activated);
			node_data_->setOutput(init_values);
		}
    else
    {
      init_values.setConstant(0.0f);
      setStatus(NodeStatus::initialized);
			node_data_->setOutput(init_values);
    }
  }

	template<typename HDelT, typename DDelT, typename TensorT>
  bool Node<HDelT, DDelT, TensorT>::checkTimeStep(const int&time_step)
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

	template<typename HDelT, typename DDelT, typename TensorT>
  void Node<HDelT, DDelT, TensorT>::checkOutput()
  {
    for (int i=0; i<batch_size_; ++i)
    {
      for (int j=0; j<memory_size_ ; ++j)
      {
        if (node_data_->getOutput()(i,j) < output_min_)
					node_data_->getOutputMutable()->operator()(i,j) = output_min_;
        else if (node_data_->getOutput()(i,j) > output_max_)
					node_data_->getOutputMutable()->operator()(i,j) = output_max_;
      }
    }
  }
}