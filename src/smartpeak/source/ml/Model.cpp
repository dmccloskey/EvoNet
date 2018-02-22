/**TODO:  Add copyright*/

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <map>

namespace SmartPeak
{
  Model::Model()
  {        
  }

  Model::Model(const int& id):
    id_(id)
  {
  }

  Model::~Model()
  {
  }
  
  void Model::setId(const int& id)
  {
    id_ = id;
  }
  int Model::getId() const
  {
    return id_;
  }

  void Model::setError(const double& error)
  {
    error_ = error;
  }
  double Model::getError() const
  {
    return error_;
  }

  void Model::addNodes(const std::vector<Node>& Nodes)
  { 
    for (Node& node: Nodes)
    {
      // check for duplicate nodes (by id)
      if (nodes_.count(node.getId()) == 0)
      {
        nodes_[node.getId()] = node;
      }
      else
      {
        std::cout << "Node id " << node.getId() << " already exists!" << std::endl;
      }
    }
  }

  void Model::removeNodes(const std::vector<int>& node_ids)
  { 
    for (int& node_id: node_ids)
    {
      // check for duplicate nodes (by id)
      if (nodes_.count(node_id) == 0))
      {
        nodes_.erase(node_id);
      }
    }
  }

  void Model::addLinks(const std::vector<Link>& Links)
  { 
    for (Link& link: Links)
    {
      // check for duplicate links (by id)
      if (links_.count(link.getId()) == 0)
      {
        links_[link.getId()] = link;
        // add in the nodes
        addNodes(link.getSourceNode());
        addNodes(link.getSinkNode());
      }
      else
      {
        std::cout << "Link id " << link.getId() << " already exists!" << std::endl;
      }
    }
  }

  void Model::removeLinks(const std::vector<int>& link_ids)
  { 
    for (int& link_id: link_ids)
    {
      // check for duplicate links (by id)
      if (links_.count(link_id) == 0))
      {
        links_.erase(link_id);
      }
    }
  }
}