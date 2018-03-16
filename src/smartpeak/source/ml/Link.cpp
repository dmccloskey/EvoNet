/**TODO:  Add copyright*/

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Operation.h>

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

  Link::Link(const int& id, const int& source_node_id,
      const int& sink_node_id,
      const SmartPeak::WeightInitMethod& weight_init):
    id_(id), weight_init_(weight_init)
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
  int Link::getSourceNodeId() const
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
  int Link::getSinkNodeId() const
  {
    return sink_node_id_;
  }

  void Link::setWeight(const float& weight)
  {
    weight_ = weight;
  }
  float Link::getWeight() const
  {
    return weight_;
  }

  void Link::setWeightInitMethod(const SmartPeak::WeightInitMethod& weight_init)
  {
    weight_init_ = weight_init;
  }
  SmartPeak::WeightInitMethod Link::getWeightInitMethod() const
  {
    return weight_init_;
  }

  void Link::initWeight(const float& op_input)
  {
    switch (weight_init_)
    {
      case SmartPeak::WeightInitMethod::RandWeightInit:
      {
        RandWeightInitOp operation;
        weight_ = operation(op_input);
        break;
      }
      case SmartPeak::WeightInitMethod::ConstWeightInit:
      {
        ConstWeightInitOp operation;
        weight_ = operation(op_input);
        break;
      }
      default:
      {
        weight_ = 0.0;
        break;
      }
    }
  }
}