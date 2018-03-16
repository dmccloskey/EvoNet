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
    deactivated = 0, // Weights have been updated (optional), ready to be re-initialized.
    initialized = 1, // Memory has been allocated for Tensors
    activated = 2, // Output has been calculated
    corrected = 3 // Error has been calculated
  };

  enum class NodeType
  {
    ReLU = 0,
    ELU = 1,
    input = 2, // No activation function
    bias = 3 // Zero value
  };

  /**
    @brief Network Node
  */
  class Node
  {
public:
    Node(); ///< Default constructor
    Node(const int& id, const SmartPeak::NodeType& type, const SmartPeak::NodeStatus& status); ///< Explicit constructor  
    ~Node(); ///< Default destructor

    inline bool operator==(const Node& other) const
    {
      return
        std::tie(
          id_,
          type_,
          status_
        ) == std::tie(
          other.id_,
          other.type_,
          other.status_
        )
      ;
    }

    inline bool operator!=(const Node& other) const
    {
      return !(*this == other);
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setType(const SmartPeak::NodeType& type); ///< type setter
    SmartPeak::NodeType getType() const; ///< type getter

    void setStatus(const SmartPeak::NodeStatus& status); ///< status setter
    SmartPeak::NodeStatus getStatus() const; ///< status getter

    // TODO: will this be needed or can we point to the Tensor value?
    void setOutput(const Eigen::Tensor<float, 1>& output); ///< output setter
    Eigen::Tensor<float, 1> getOutput() const; ///< output copy getter
    float* getOutputPointer(); ///< output pointer getter

    void setError(const Eigen::Tensor<float, 1>& error); ///< error setter
    Eigen::Tensor<float, 1> getError() const; ///< error copy getter
    float* getErrorPointer(); ///< error pointer getter

    void setDerivative(const Eigen::Tensor<float, 1>& derivative); ///< derivative setter
    Eigen::Tensor<float, 1> getDerivative() const; ///< derivative copy getter
    float* getDerivativePointer(); ///< derivative pointer getter

    /**
      @brief Initialize node output to zero.
        The node statuses are then changed to NodeStatus::deactivated

      @param[in] batch_size Size of the output, error, and derivative node vectors
    */ 
    void initNode(const int& batch_size);

    /**
      @brief The current output is passed through an activation function.
        Contents are updated in place.
    */
    void calculateActivation();
    
    /**
      @brief Calculate the derivative from the output.
    */
    void calculateDerivative();

private:
    int id_; ///< Node ID
    SmartPeak::NodeType type_; ///< Node Type
    SmartPeak::NodeStatus status_; ///< Node Status
    
    Eigen::Tensor<float, 1> output_; ///< Node Output (dim is the # of samples)
    Eigen::Tensor<float, 1> error_; ///< Node Error (dim is the # of samples)
    Eigen::Tensor<float, 1> derivative_; ///< Node Error (dim is the # of samples)

  };
}

#endif //SMARTPEAK_NODE_H