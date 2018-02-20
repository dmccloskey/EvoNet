/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODE_H
#define SMARTPEAK_NODE_H

#include <string>

namespace SmartPeak
{
  enum class NodeStatus
  {
    deactivated,
    activated
  };

  enum class NodeType
  {
    ReLU,
    ELU
  };

  /**
    @brief Network Node
  */
  class Node
  {
public:
    Node(); ///< Default constructor
    Node(const int& id, NodeType& type, NodeStatus& status, double& output, double& error); ///< Explicit constructor  
    ~Node(); ///< Default destructor

    void setId(const double& id); ///< id setter
    double getId() const; ///< id getter


private:
    int id_; ///< Node ID
    NodeType type_; ///< Node Type
    NodeStatus status_; ///< Node Status
    double output_; ///< Node Output
    double error_; ///< Node Error

  };
}

#endif //SMARTPEAK_NODE_H