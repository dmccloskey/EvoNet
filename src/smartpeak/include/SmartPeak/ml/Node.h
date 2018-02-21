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
    corrected = 3, // Error has been calculated
  };

  enum class NodeType
  {
    ReLU = 0,
    ELU = 1
  };

  /**
    @brief Network Node
  */
  class Node
  {
public:
    Node(); ///< Default constructor
    Node(const int& id, SmartPeak::NodeType& type, SmartPeak::NodeStatus& status); ///< Explicit constructor  
    ~Node(); ///< Default destructor

    void setId(const double& id); ///< id setter
    double getId() const; ///< id getter

    void setType(const SmartPeak::NodeType& type); ///< type setter
    SmartPeak::NodeType getType() const; ///< type getter

    void setStatus(const SmartPeak::NodeStatus& status); ///< status setter
    SmartPeak::NodeStatus getStatus() const; ///< status getter

    void setOutput(const Eigen::Tensor<float, 1>& output); ///< ouptput setter
    Eigen::Tensor<float, 1> getOutput() const; ///< output getter

    void setError(const Eigen::Tensor<float, 1>& error); ///< error setter
    Eigen::Tensor<float, 1> getError() const; ///< error getter

private:
    int id_; ///< Node ID
    SmartPeak::NodeType type_; ///< Node Type
    SmartPeak::NodeStatus status_; ///< Node Status
    Eigen::Tensor<float, 1> output_; ///< Node Output (dim is the # of samples)
    Eigen::Tensor<float, 1> error_; ///< Node Error (dim is the # of samples)

  };
}

#endif //SMARTPEAK_NODE_H