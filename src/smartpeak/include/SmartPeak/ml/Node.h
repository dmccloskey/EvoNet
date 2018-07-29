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
    hidden = 4
  };

  enum class NodeActivation
  {
    Linear = 0,
    ReLU = 1,
    ELU = 2,
    Sigmoid = 3,
    TanH = 4
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
          name_
        ) == std::tie(
          other.id_,
          other.type_,
          other.status_,
          other.activation_,
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
      type_ = other.type_;
      activation_ = other.activation_;
      status_ = other.status_;
      output_min_ = other.output_min_;
      output_max_ = other.output_max_;
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
		SmartPeak::NodeIntegration getIntegration() const; ///< integration getter

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
    @brief The current output is passed through an activation function.
    Contents are updated in place.

    @param[in] time_step Time step to activate all samples in the batch

    [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
    */
    void calculateActivation(const int& time_step);

    /**
    @brief Calculate the derivative from the output.

    @param[in] time_step Time step to calculate the derivative
    for all samples in the batch

    [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
    */
    void calculateDerivative(const int& time_step);
    
    /**
      @brief Shifts the current output batch by 1 unit back in memory.

			[DEPRECATED]
    */
    void saveCurrentOutput();
    
    /**
      @brief Shifts the current derivative batch by 1 unit back in memory.

			[DEPRECATED]
    */
    void saveCurrentDerivative();
    
    /**
      @brief Shifts the current error batch by 1 unit back in memory.

			[DEPRECATED]
    */
    void saveCurrentError();
    
    /**
      @brief Shifts the current dt batch by 1 unit back in memory.

			[DEPRECATED]
    */
    void saveCurrentDt();
 
    /**
      @brief Check if the output is within the min/max.  

			[DEPRECATED]
    */ 
    void checkOutput();

private:
    int id_ = NULL; ///< Weight ID
    std::string name_ = ""; ///< Weight Name
    SmartPeak::NodeType type_ = SmartPeak::NodeType::hidden; ///< Node Type
    SmartPeak::NodeStatus status_ = SmartPeak::NodeStatus::deactivated; ///< Node Status   
    SmartPeak::NodeActivation activation_ = SmartPeak::NodeActivation::ReLU; ///< Node Status   
		SmartPeak::NodeIntegration integration_ = SmartPeak::NodeIntegration::Sum; ///< Node Integration   

    float output_min_ = -1.0e6; ///< Min Node output
    float output_max_ = 1.0e6; ///< Max Node output

    /**
      @brief output, error and derivative have the following dimensions:
        rows: # of samples, cols: # of time steps
        where the number of samples spans 0 to n samples
        and the number of time steps spans m time points to 0
    */
    Eigen::Tensor<float, 2> output_; ///< Node Output (rows: # of samples, cols: # of time steps)
    Eigen::Tensor<float, 2> error_; ///< Node Error (rows: # of samples, cols: # of time steps)
    Eigen::Tensor<float, 2> derivative_; ///< Node Error (rows: # of samples, cols: # of time steps)
    Eigen::Tensor<float, 2> dt_; ///< Resolution of each time-step (rows: # of samples, cols: # of time steps)

  };
}

#endif //SMARTPEAK_NODE_H