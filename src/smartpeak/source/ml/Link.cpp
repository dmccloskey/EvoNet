/**TODO:  Add copyright*/

#include <SmartPeak/ml/Link.h>

#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{
  Link::Link()
  {        
  }

  Link::Link(const int& id, const int& source_node_id,
      const int& sink_node_id):
    id_(id)
  {
    setSourceNodeId(source_node_id);
    setSinkNodeId(sink_node_id);
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

  void Link::setSourceNodeId(const int& source_node_id)
  {
    if (sink_node_id_ == source_node_id)
    {
      std::cout << "Source and Sink nodes are the same!" << std::endl;
    }
    else
    {
      source_node_id_ = source_node_id;
    }    
  }
  int Link::getSourceNode() const
  {
    return source_node_id_;
  }

  void Link::setSinkNodeId(const int& sink_node_id)
  {
    if (source_node_id_ == sink_node_id)
    {
      std::cout << "Source and Sink nodes are the same!" << std::endl;
    }
    else
    {
      sink_node_id_ = sink_node_id;
    }    
  }
  int Link::getSinkNode() const
  {
    return sink_node_id_;
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