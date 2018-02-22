/**TODO:  Add copyright*/

#include <SmartPeak/ml/Node.h>

#include <vector>
#include <cmath>

namespace SmartPeak
{
  Node::Node()
  {        
  }

  Node::Node(const int& id, const SmartPeak::NodeType& type,
    const SmartPeak::NodeStatus& status):
    id_(id), type_(type), status_(status)
  {
  }

  Node::~Node()
  {
  }
  
  void Node::setId(const int& id)
  {
    id_ = id;
  }
  int Node::getId() const
  {
    return id_;
  }

  void Node::setType(const SmartPeak::NodeType& type)
  {
    type_ = type;
  }
  SmartPeak::NodeType Node::getType() const
  {
    return type_;
  }

  void Node::setStatus(const SmartPeak::NodeStatus& status)
  {
    status_ = status;
  }
  SmartPeak::NodeStatus Node::getStatus() const
  {
    return status_;
  }
}