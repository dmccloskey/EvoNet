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

  Link::Link(const int& id):
    id_(id)
  {
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }

  Link::Link(const std::string& name):
    name_(name)
  {
  }

  Link::Link(const std::string& name, const std::string& source_node_name,
      const std::string& sink_node_name,
      const std::string& weight_name):
    name_(name), weight_name_(weight_name)
  {
    setSourceNodeName(source_node_name);
    setSinkNodeName(sink_node_name);
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
  
  void Link::setName(const std::string& name)
  {
    name_ = name;    
  }
  std::string Link::getName() const
  {
    return name_;
  }

  void Link::setSourceNodeName(const std::string& source_node_name)
  {
    if (sink_node_name_ == source_node_name)
    {
      std::cout << "Source and Sink nodes are the same!" << std::endl;
    }
    else
    {
      source_node_name_ = source_node_name;
    }    
  }
  std::string Link::getSourceNodeName() const
  {
    return source_node_name_;
  }

  void Link::setSinkNodeName(const std::string& sink_node_name)
  {
    if (source_node_name_ == sink_node_name)
    {
      std::cout << "Source and Sink nodes are the same!" << std::endl;
    }
    else
    {
      sink_node_name_ = sink_node_name;
    }    
  }
  std::string Link::getSinkNodeName() const
  {
    return sink_node_name_;
  }

  void Link::setWeightName(const std::string& weight_name)
  {
    weight_name_ = weight_name;
  }
  std::string Link::getWeightName() const
  {
    return weight_name_;
  }
}