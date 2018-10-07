/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODE_H
#define SMARTPEAK_NODE_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <SmartPeak/ml/IntegrationFunction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>

#ifndef EVONET_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

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
  class Node
  {
public:
    Node(); ///< Default constructor
    Node(const Node& other); ///< Copy constructor // [TODO: add test]
    Node(const std::string& name, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status, 
			const std::shared_ptr<ActivationOp<float>>& activation, const std::shared_ptr<ActivationOp<float>>& activation_grad, 
			const std::shared_ptr<IntegrationOp<float>>& integration,
			const std::shared_ptr<IntegrationErrorOp<float>>& integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<float>>& integration_weight_grad); ///< Explicit constructor
    ~Node(); ///< Default destructor

    inline bool operator==(const Node& other) const
    {
      return
        std::tie(
          id_,
          type_,
          status_,
     //     activation_->getName(),
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
 /*         other.activation_->getName(),
					other.activation_grad_->getName(),
					other.integration_->getName(),
					other.integration_error_->getName(),
					other.integration_weight_grad_->getName(),*/
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
			input_ = other.input_;
      output_ = other.output_;
      error_ = other.error_;
      derivative_ = other.derivative_;
      dt_ = other.dt_;
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

    void setActivation(const std::shared_ptr<ActivationOp<float>>& activation); ///< activation setter
		std::shared_ptr<ActivationOp<float>> getActivationShared() const; // [TODO: add tests]
		ActivationOp<float>* getActivation() const; ///< activation getter

		void setActivationGrad(const std::shared_ptr<ActivationOp<float>>& activation_grad); ///< activation setter
		std::shared_ptr<ActivationOp<float>> getActivationGradShared() const; // [TODO: add tests]
		ActivationOp<float>* getActivationGrad() const; ///< activation getter

		void setIntegration(const std::shared_ptr<IntegrationOp<float>>& integration); ///< integration setter
		std::shared_ptr<IntegrationOp<float>> getIntegrationShared() const; // [TODO: add tests]
		IntegrationOp<float>* getIntegration() const; ///< integration getter

		void setIntegrationError(const std::shared_ptr<IntegrationErrorOp<float>>& integration); ///< integration setter
		std::shared_ptr<IntegrationErrorOp<float>> getIntegrationErrorShared() const; // [TODO: add tests]
		IntegrationErrorOp<float>* getIntegrationError() const; ///< integration getter

		void setIntegrationWeightGrad(const std::shared_ptr<IntegrationWeightGradOp<float>>& integration); ///< integration setter
		std::shared_ptr<IntegrationWeightGradOp<float>> getIntegrationWeightGradShared() const; // [TODO: add tests]
		IntegrationWeightGradOp<float>* getIntegrationWeightGrad() const; ///< integration getter


		void setModuleId(const int& module_id); ///< module id setter
		int getModuleId() const; ///< module id getter

		void setModuleName(const std::string& module_name); ///< module name setter
		std::string getModuleName() const; ///< module name getter

		void setInput(const Eigen::Tensor<float, 2>& input); ///< input setter
		Eigen::Tensor<float, 2> getInput() const; ///< input copy getter
		Eigen::Tensor<float, 2>* getInputMutable(); ///< input copy getter
		float* getInputPointer(); ///< input pointer getter

    void setOutput(const Eigen::Tensor<float, 2>& output); ///< output setter
    Eigen::Tensor<float, 2> getOutput() const; ///< output copy getter
    Eigen::Tensor<float, 2>* getOutputMutable(); ///< output copy getter
    float* getOutputPointer(); ///< output pointer getter

    void setError(const Eigen::Tensor<float, 2>& error); ///< error setter
    Eigen::Tensor<float, 2> getError() const; ///< error copy getter
    Eigen::Tensor<float, 2>* getErrorMutable(); ///< error copy getter
    float* getErrorPointer(); ///< error pointer getter

    void setDerivative(const Eigen::Tensor<float, 2>& derivative); ///< derivative setter
    Eigen::Tensor<float, 2> getDerivative() const; ///< derivative copy getter
    Eigen::Tensor<float, 2>* getDerivativeMutable(); ///< derivative copy getter
    float* getDerivativePointer(); ///< derivative pointer getter

    void setDt(const Eigen::Tensor<float, 2>& dt); ///< dt setter
    Eigen::Tensor<float, 2> getDt() const; ///< dt copy getter
    Eigen::Tensor<float, 2>* getDtMutable(); ///< dt copy getter
    float* getDtPointer(); ///< dt pointer getter

    void setOutputMin(const float& min_output); ///< min output setter
    void setOutputMax(const float& output_max); ///< max output setter

		void setDropProbability(const float& drop_probability); ///< drop_probability setter
		float getDropProbability() const; ///< drop_probability getter

		void setDrop(const Eigen::Tensor<float, 2>& drop); ///< drop setter
		Eigen::Tensor<float, 2> getDrop() const; ///< drop copy getter

		int getBatchSize() const;
		int getMemorySize() const;

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
    std::shared_ptr<ActivationOp<float>> activation_; ///< Node activation function 
		std::shared_ptr<ActivationOp<float>> activation_grad_; ///< Node activation function 
		std::shared_ptr<IntegrationOp<float>> integration_; ///< Node integration function 
		std::shared_ptr<IntegrationErrorOp<float>> integration_error_; ///< Node integration error function 
		std::shared_ptr<IntegrationWeightGradOp<float>> integration_weight_grad_; ///< Node integration weight grad function 

    float output_min_ = -1.0e6; ///< Min Node output
    float output_max_ = 1.0e6; ///< Max Node output

		size_t batch_size_ = 1; ///< Mini batch size
		size_t memory_size_ = 2; ///< Memory size

    /**
      @brief output, error and derivative have the following dimensions:
        rows: # of samples, cols: # of time steps
        where the number of samples spans 0 to n samples
        and the number of time steps spans m time points to 0
    */
		Eigen::TensorMap<Eigen::Tensor<float, 2>> input_; ///< Node Net Input (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<float, 2>> output_; ///< Node Output (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<float, 2>> error_; ///< Node Error (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<float, 2>> derivative_; ///< Node Error (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<float, 2>> dt_; ///< Resolution of each time-step (rows: # of samples, cols: # of time steps)
		std::shared_ptr<float> h_input_;
		std::shared_ptr<float> h_output_;
		std::shared_ptr<float> h_error_;
		std::shared_ptr<float> h_derivative_;
		std::shared_ptr<float> h_dt_; 

		float drop_probability_ = 0.0;
		Eigen::Tensor<float, 2> drop_; ///< Node Output drop tensor (initialized once per epoch)
  };
}

#endif //SMARTPEAK_NODE_H