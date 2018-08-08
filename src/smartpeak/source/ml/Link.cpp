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
  
  Link::Link(const Link& other)
  {    
    id_ = other.id_;
    name_ = other.name_;
		module_id_ = other.module_id_;
		module_name_ = other.module_name_;
    source_node_name_ = other.source_node_name_;
    sink_node_name_ = other.sink_node_name_;
    weight_name_ = other.weight_name_;
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
      source_node_name_ = source_node_name;
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
    sink_node_name_ = sink_node_name;   
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

	void Link::setModuleId(const int & module_id)
	{
		module_id_ = module_id;
	}

	int Link::getModuleId() const
	{
		return module_id_;
	}

	void Link::setModuleName(const std::string & module_name)
	{
		module_name_ = module_name;
	}

	std::string Link::getModuleName() const
	{
		return module_name_;
	}
}