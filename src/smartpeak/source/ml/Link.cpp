/**TODO:  Add copyright*/

#include <SmartPeak/ml/Link.h>

#include <vector>
#include <cmath>

namespace SmartPeak
{
  Link::Link()
  {        
  }

  Link::Link(const int& id, const SmartPeak::Node& source_node,
      const SmartPeak::Node& sink_node):
    id_(id), source_node_(source_node), sink_node_(sink_node)
  {
  }

  Link::~Link()
  {
  }
  
  void Link::setId(const int& id)
  {
    id_ = id;
  }
  int Link::getId() const
  {
    return id_;
  }

  void Link::setSourceNode(const SmartPeak::Node& source_node)
  {
    source_node_ = source_node;
  }
  SmartPeak::Node Link::getSourceNode() const
  {
    return source_node_;
  }

  void Link::setSinkNode(const SmartPeak::Node& sink_node)
  {
    sink_node_ = sink_node;
  }
  SmartPeak::Node Link::getSinkNode() const
  {
    return sink_node_;
  }

  void Link::setWeight(const double& weight)
  {
    weight_ = weight;
  }
  double Link::getWeight() const
  {
    return weight_;
  }
}