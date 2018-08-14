/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODE_H
#define SMARTPEAK_NODE_H

#include <unsupported/Eigen/CXX11/Tensor>
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
    bias = 2, // Zero value
    output = 3, 
    hidden = 4,
		unmodifiable = 5
  };

  enum class NodeActivation
  {
    Linear = 0,
    ReLU = 1,
    ELU = 2,
    Sigmoid = 3,
    TanH = 4,
		Inverse = 5,
		Exponential = 6
  };

	enum class NodeIntegration
	{
		Sum = 0,
		Product = 1,
		Max = 2
	};

  /**
    @brief Network Node
  */
  class Node
  {
public:
    Node(); ///< Default constructor
    Node(const Node& other); ///< Copy constructor // [TODO: add test]
    Node(const std::string& name, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status, const SmartPeak::NodeActivation& activation, const SmartPeak::NodeIntegration& integration); ///< Explicit constructor  
    Node(const int& id, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status, const SmartPeak::NodeActivation& activation, const SmartPeak::NodeIntegration& integration); ///< Explicit constructor  
    ~Node(); ///< Default destructor

    inline bool operator==(const Node& other) const
    {
      return
        std::tie(
          id_,
          type_,
          status_,
          activation_,
					integration_,
          name_
        ) == std::tie(
          other.id_,
          other.type_,
          other.status_,
          other.activation_,
					other.integration_,
          other.name_
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
			integration_ = other.integration_;
      status_ = other.status_;
      output_min_ = other.output_min_;
      output_max_ = other.output_max_;
			input_ = other.input_;
      output_ = other.output_;
      error_ = other.error_;
      derivative_ = other.derivative_;
      dt_ = other.dt_;
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

    void setActivation(const SmartPeak::NodeActivation& activation); ///< activation setter
    SmartPeak::NodeActivation getActivation() const; ///< activation getter

		void setIntegration(const SmartPeak::NodeIntegration & integration); ///< integration setter
		SmartPeak::NodeIntegration getIntegration() const; ///< integration 

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

    /**
      @brief Initialize node output to zero.
        The node statuses are then changed to NodeStatus::deactivated

      @param[in] batch_size Size of the row dim for the output, error, and derivative node vectors
      @param[in] memory_size Size of the col dim output, error, and derivative node vectors
    */ 
    void initNode(const int& batch_size, const int& memory_size);

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
    SmartPeak::NodeActivation activation_; ///< Node Status   
		SmartPeak::NodeIntegration integration_; ///< Node Integration   

    float output_min_ = -1.0e6; ///< Min Node output
    float output_max_ = 1.0e6; ///< Max Node output

    /**
      @brief output, error and derivative have the following dimensions:
        rows: # of samples, cols: # of time steps
        where the number of samples spans 0 to n samples
        and the number of time steps spans m time points to 0
    */
		Eigen::Tensor<float, 2> input_; ///< Node Net Input (rows: # of samples, cols: # of time steps)
    Eigen::Tensor<float, 2> output_; ///< Node Output (rows: # of samples, cols: # of time steps)
    Eigen::Tensor<float, 2> error_; ///< Node Error (rows: # of samples, cols: # of time steps)
    Eigen::Tensor<float, 2> derivative_; ///< Node Error (rows: # of samples, cols: # of time steps)
    Eigen::Tensor<float, 2> dt_; ///< Resolution of each time-step (rows: # of samples, cols: # of time steps)

  };
}

#endif //SMARTPEAK_NODE_H