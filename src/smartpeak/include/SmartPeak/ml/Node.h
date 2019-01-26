/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODE_H
#define SMARTPEAK_NODE_H

// .h
#include <SmartPeak/ml/ActivationFunction.h>
#include <SmartPeak/ml/IntegrationFunction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/utility.hpp> // std::pair

namespace SmartPeak
{
  enum class NodeStatus
  {
    // TODO: will these be the final set of states a node can be in?
    deactivated = 0, // Optional: utilized to indicate that there should be no change in node status
    initialized = 1, // Memory has been allocated for Tensors
    activated = 2, // Output has been calculated
    corrected = 3 // Error has been calculated
  };

  enum class NodeType
  {
    input = 1, // No activation function
    bias = 2, // Value of 1
    output = 3, 
    hidden = 4,
		unmodifiable = 5,
		zero = 6, // value of 0
		recursive = 7 // special case of hidden where the node should be treated as the source of any cyclic pair
  };

  /**
    @brief Network Node
  */
	template<typename TensorT>
  class Node
  {
public:
    Node() = default; ///< Default constructor
    Node(const Node& other); ///< Copy constructor // [TODO: add test]
    Node(const std::string& name, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status, 
			const std::shared_ptr<ActivationOp<TensorT>>& activation, const std::shared_ptr<ActivationOp<TensorT>>& activation_grad, 
			const std::shared_ptr<IntegrationOp<TensorT>>& integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& integration_weight_grad); ///< Explicit constructor
    ~Node() = default; ///< Default destructor

    inline bool operator==(const Node& other) const
    {
      return
        std::tie(
          id_,
          type_,
          status_,
          //activation_->getName(),
					//activation_grad_->getName(),
					//integration_->getName(),
					//integration_error_->getName(),
					//integration_weight_grad_->getName(),
          name_,
					module_id_,
					module_name_,
					tensor_index_
        ) == std::tie(
          other.id_,
          other.type_,
          other.status_,
          //other.activation_->getName(),
					//other.activation_grad_->getName(),
					//other.integration_->getName(),
					//other.integration_error_->getName(),
					//other.integration_weight_grad_->getName(),
          other.name_,
					other.module_id_,
					other.module_name_,
					other.tensor_index_
        )
      ;
    }

    inline bool operator!=(const Node& other) const
    {
      return !(*this == other);
    }

    inline Node& operator=(const Node& other)
    { // [TODO: add test]
      id_ = other.id_;
      name_ = other.name_;
			module_id_ = other.module_id_;
			module_name_ = other.module_name_;
			tensor_index_ = other.tensor_index_;
			layer_name_ = other.layer_name_;
      type_ = other.type_;
      activation_ = other.activation_;
			activation_grad_ = other.activation_grad_;
			integration_ = other.integration_;
			integration_error_ = other.integration_error_;
			integration_weight_grad_ = other.integration_weight_grad_;
      status_ = other.status_;
      output_min_ = other.output_min_;
      output_max_ = other.output_max_;
			drop_probability_ = other.drop_probability_;
			drop_ = other.drop_;
      return *this;
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< naem setter
    std::string getName() const; ///< name getter

    void setType(const SmartPeak::NodeType& type); ///< type setter
    SmartPeak::NodeType getType() const; ///< type getter

    void setStatus(const SmartPeak::NodeStatus& status); ///< status setter
    SmartPeak::NodeStatus getStatus() const; ///< status getter

    void setActivation(const std::shared_ptr<ActivationOp<TensorT>>& activation); ///< activation setter
		std::shared_ptr<ActivationOp<TensorT>> getActivationShared() const; // [TODO: add tests]
		ActivationOp<TensorT>* getActivation() const; ///< activation getter

		void setActivationGrad(const std::shared_ptr<ActivationOp<TensorT>>& activation_grad); ///< activation setter
		std::shared_ptr<ActivationOp<TensorT>> getActivationGradShared() const; // [TODO: add tests]
		ActivationOp<TensorT>* getActivationGrad() const; ///< activation getter

		void setIntegration(const std::shared_ptr<IntegrationOp<TensorT>>& integration); ///< integration setter
		std::shared_ptr<IntegrationOp<TensorT>> getIntegrationShared() const; // [TODO: add tests]
		IntegrationOp<TensorT>* getIntegration() const; ///< integration getter

		void setIntegrationError(const std::shared_ptr<IntegrationErrorOp<TensorT>>& integration); ///< integration setter
		std::shared_ptr<IntegrationErrorOp<TensorT>> getIntegrationErrorShared() const; // [TODO: add tests]
		IntegrationErrorOp<TensorT>* getIntegrationError() const; ///< integration getter

		void setIntegrationWeightGrad(const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& integration); ///< integration setter
		std::shared_ptr<IntegrationWeightGradOp<TensorT>> getIntegrationWeightGradShared() const; // [TODO: add tests]
		IntegrationWeightGradOp<TensorT>* getIntegrationWeightGrad() const; ///< integration getter

		void setModuleId(const int& module_id); ///< module id setter
		int getModuleId() const; ///< module id getter

		void setTensorIndex(const std::pair<int, int>& layer_id); ///< layer id setter
		std::pair<int, int> getTensorIndex() const; ///< layer id getter

		void setModuleName(const std::string& module_name); ///< module name setter
		std::string getModuleName() const; ///< module name getter

		void setLayerName(const std::string& layer_name); ///< layer name setter
		std::string getLayerName() const; ///< layer name getter

    void setOutputMin(const TensorT& min_output); ///< min output setter
    void setOutputMax(const TensorT& output_max); ///< max output setter

		void setDropProbability(const TensorT& drop_probability); ///< drop_probability setter
		TensorT getDropProbability() const; ///< drop_probability getter

		void setDrop(const Eigen::Tensor<TensorT, 2>& drop); ///< drop setter
		Eigen::Tensor<TensorT, 2> getDrop() const; ///< drop copy getter

		void setInput(const Eigen::Tensor<TensorT, 2>& input); ///< input setter
		Eigen::Tensor<TensorT, 2> getInput() const; ///< input copy getter

		void setOutput(const Eigen::Tensor<TensorT, 2>& output); ///< output setter
		Eigen::Tensor<TensorT, 2> getOutput() const; ///< output copy getter

		void setError(const Eigen::Tensor<TensorT, 2>& error); ///< error setter
		Eigen::Tensor<TensorT, 2> getError() const; ///< error copy getter

		void setDerivative(const Eigen::Tensor<TensorT, 2>& derivative); ///< derivative setter
		Eigen::Tensor<TensorT, 2> getDerivative() const; ///< derivative copy getter

		void setDt(const Eigen::Tensor<TensorT, 2>& dt); ///< dt setter
		Eigen::Tensor<TensorT, 2> getDt() const; ///< dt copy getter
		
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(id_, name_, module_id_, module_name_, layer_name_, output_max_, output_min_,
				tensor_index_, type_, status_, 
				activation_, activation_grad_, 
				integration_, integration_error_, integration_weight_grad_
			);
		}
    int id_ = -1; ///< Node ID (used internally by Model)
    std::string name_ = ""; ///< Node Name
		int module_id_ = -1; ///< Module ID (used internally by Model)
		std::string module_name_ = ""; ///<Module Name
		std::pair<int, int> tensor_index_ = std::make_pair(-1,-1); ///< Layer ID: pair consisting of OperationsList index and Layer index(used internally by Model)
		std::string layer_name_ = ""; ///< Layer name
		SmartPeak::NodeType type_; ///< Node Type
    SmartPeak::NodeStatus status_; ///< Node Status   
    std::shared_ptr<ActivationOp<TensorT>> activation_; ///< Node activation function 
		std::shared_ptr<ActivationOp<TensorT>> activation_grad_; ///< Node activation function 
		std::shared_ptr<IntegrationOp<TensorT>> integration_; ///< Node integration function 
		std::shared_ptr<IntegrationErrorOp<TensorT>> integration_error_; ///< Node integration error function 
		std::shared_ptr<IntegrationWeightGradOp<TensorT>> integration_weight_grad_; ///< Node integration weight grad function 
    TensorT output_min_ = -1.0e6; ///< Min Node output
    TensorT output_max_ = 1.0e6; ///< Max Node output
		TensorT drop_probability_ = 0.0;
		Eigen::Tensor<TensorT, 2> drop_; ///< Node Output drop tensor (initialized once per epoch)

		/**
			@brief output, error and derivative have the following dimensions:
				rows: # of samples, cols: # of time steps
				where the number of samples spans 0 to n samples
				and the number of time steps spans m time points to 0
		*/
		Eigen::Tensor<TensorT, 2> input_; ///< Node Net Input (rows: # of samples, cols: # of time steps)
		Eigen::Tensor<TensorT, 2> output_; ///< Node Output (rows: # of samples, cols: # of time steps)
		Eigen::Tensor<TensorT, 2> error_; ///< Node Error (rows: # of samples, cols: # of time steps)
		Eigen::Tensor<TensorT, 2> derivative_; ///< Node Error (rows: # of samples, cols: # of time steps)
		Eigen::Tensor<TensorT, 2> dt_; ///< Resolution of each time-step (rows: # of samples, cols: # of time steps)

  };
	template<typename TensorT>
	Node<TensorT>::Node(const Node<TensorT>& other)
	{
		id_ = other.id_;
		name_ = other.name_;
		module_id_ = other.module_id_;
		module_name_ = other.module_name_;
		tensor_index_ = other.tensor_index_;
		layer_name_ = other.layer_name_;
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
		drop_probability_ = other.drop_probability_;
		drop_ = other.drop_;
	}

	template<typename TensorT>
	Node<TensorT>::Node(const std::string& name, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status,
		const std::shared_ptr<ActivationOp<TensorT>>& activation, const std::shared_ptr<ActivationOp<TensorT>>& activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& integration, const std::shared_ptr<IntegrationErrorOp<TensorT>>& integration_error, const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& integration_weight_grad) :
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
	void Node<TensorT>::setTensorIndex(const std::pair<int, int>& layer_id)
	{
		tensor_index_ = layer_id;
	}

	template<typename TensorT>
	std::pair<int, int> Node<TensorT>::getTensorIndex() const
	{
		return tensor_index_;
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
	inline void Node<TensorT>::setLayerName(const std::string & layer_name)
	{
		layer_name_ = layer_name;
	}

	template<typename TensorT>
	inline std::string Node<TensorT>::getLayerName() const
	{
		return layer_name_;
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
	void Node<TensorT>::setInput(const Eigen::Tensor<TensorT, 2>& input)
	{
		input_ = input;
	}
	template<typename TensorT>
	Eigen::Tensor<TensorT, 2> Node<TensorT>::getInput() const
	{
		return input_;
	}

	template<typename TensorT>
	void Node<TensorT>::setOutput(const Eigen::Tensor<TensorT, 2>& output)
	{
		output_ = output;
	}
	template<typename TensorT>
	Eigen::Tensor<TensorT, 2> Node<TensorT>::getOutput() const
	{
		return output_;// *getDrop();
	}

	template<typename TensorT>
	void Node<TensorT>::setError(const Eigen::Tensor<TensorT, 2>& error)
	{
		error_ = error;
	}
	template<typename TensorT>
	Eigen::Tensor<TensorT, 2> Node<TensorT>::getError() const
	{
		return error_;
	}

	template<typename TensorT>
	void Node<TensorT>::setDerivative(const Eigen::Tensor<TensorT, 2>& derivative)
	{
		derivative_ = derivative;
	}
	template<typename TensorT>
	Eigen::Tensor<TensorT, 2> Node<TensorT>::getDerivative() const
	{
		return derivative_;
	}

	template<typename TensorT>
	void Node<TensorT>::setDt(const Eigen::Tensor<TensorT, 2>& dt)
	{
		dt_ = dt;
	}
	template<typename TensorT>
	Eigen::Tensor<TensorT, 2> Node<TensorT>::getDt() const
	{
		return dt_;
	}
}

#endif //SMARTPEAK_NODE_H