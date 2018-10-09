/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODE_H
#define SMARTPEAK_NODE_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <SmartPeak/ml/IntegrationFunction.h>
#include <SmartPeak/ml/NodeData.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>

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
					module_name_
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
					other.module_name_
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
      type_ = other.type_;
      activation_ = other.activation_;
			activation_grad_ = other.activation_grad_;
			integration_ = other.integration_;
			integration_error_ = other.integration_error_;
			integration_weight_grad_ = other.integration_weight_grad_;
      status_ = other.status_;
      output_min_ = other.output_min_;
      output_max_ = other.output_max_;
			node_data_ = other.node_data_;
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

		void setModuleName(const std::string& module_name); ///< module name setter
		std::string getModuleName() const; ///< module name getter

    void setOutputMin(const TensorT& min_output); ///< min output setter
    void setOutputMax(const TensorT& output_max); ///< max output setter

		void setDropProbability(const TensorT& drop_probability); ///< drop_probability setter
		TensorT getDropProbability() const; ///< drop_probability getter

		void setDrop(const Eigen::Tensor<TensorT, 2>& drop); ///< drop setter
		Eigen::Tensor<TensorT, 2> getDrop() const; ///< drop copy getter

		size_t getBatchSize() const;
		size_t getMemorySize() const;

		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getInput() { return node_data_->getInput(); }; ///< input copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getOutput() { return node_data_->getOutput(); }; ///< output copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getError() { return node_data_->getError(); }; ///< error copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getDerivative() { return node_data_->getDerivative(); }; ///< derivative copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getDt() { return node_data_->getDt(); }; ///< dt copy getter

		void setNodeData(const std::shared_ptr<NodeData<TensorT>>& node_data);
		std::shared_ptr<NodeData<TensorT>> getNodeData();

    /**
      @brief Initialize node output to zero.
        The node statuses are then changed to NodeStatus::deactivated

      @param[in] batch_size Size of the row dim for the output, error, and derivative node vectors
      @param[in] memory_size Size of the col dim output, error, and derivative node vectors
			@param[in] train True if training, False if validation (effectively shuts of any node regularlization, i.e., DropOut)
    */ 
    void initNode(const int& batch_size, const int& memory_size, bool train = false);

    /**
      @brief CHeck that the time_step is greater than 0 and not larger than
        the node memory size.

      @param[in] time_step Time step

      @returns true if valid time_step, false otherwise
    */ 
    bool checkTimeStep(const int& time_step);

    /**
      @brief Check if the output is within the min/max.  

    */ 
    void checkOutput();

private:
    int id_ = -1; ///< Weight ID
    std::string name_ = ""; ///< Weight Name
		int module_id_ = -1; ///< Module ID
		std::string module_name_ = ""; ///<Module Name
		SmartPeak::NodeType type_; ///< Node Type
    SmartPeak::NodeStatus status_; ///< Node Status   
    std::shared_ptr<ActivationOp<TensorT>> activation_; ///< Node activation function 
		std::shared_ptr<ActivationOp<TensorT>> activation_grad_; ///< Node activation function 
		std::shared_ptr<IntegrationOp<TensorT>> integration_; ///< Node integration function 
		std::shared_ptr<IntegrationErrorOp<TensorT>> integration_error_; ///< Node integration error function 
		std::shared_ptr<IntegrationWeightGradOp<TensorT>> integration_weight_grad_; ///< Node integration weight grad function 
    TensorT output_min_ = -1.0e6; ///< Min Node output
    TensorT output_max_ = 1.0e6; ///< Max Node output
		std::shared_ptr<NodeData<TensorT>> node_data_; ///< Node data
		TensorT drop_probability_ = 0.0;
		Eigen::Tensor<TensorT, 2> drop_; ///< Node Output drop tensor (initialized once per epoch)
  };
}

#endif //SMARTPEAK_NODE_H