/**TODO:  Add copyright*/

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <map>
#include <iostream>

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

  void Model::addNodes(const std::vector<Node>& nodes)
  { 
    for (Node const& node: nodes)
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

  Node Model::getNode(const int& node_id) const
  {
    if (nodes_.count(node_id) != 0)
    {
      return nodes_.at(node_id);
    }
    else
    {
      std::cout << "Node id " << node_id << " not found!" << std::endl;
    }
  }

  void Model::removeNodes(const std::vector<int>& node_ids)
  { 
    for (int const& node_id: node_ids)
    {
      // check for duplicate nodes (by id)
      if (nodes_.count(node_id) == 0)
      {
        nodes_.erase(node_id);
      }
    }
    pruneLinks();
  }

  void Model::addLinks(const std::vector<Link>& links)
  { 
    for (Link const& link: links)
    {
      // check for duplicate links (by id)
      if (links_.count(link.getId()) == 0)
      {
        links_[link.getId()] = link;
        // add in the nodes
        addNodes({link.getSourceNode(), link.getSinkNode()});
      }
      else
      {
        std::cout << "Link id " << link.getId() << " already exists!" << std::endl;
      }
    }
  }

  void Model::removeLinks(const std::vector<int>& link_ids)
  { 
    for (int const& link_id: link_ids)
    {
      // check for duplicate links (by id)
      if (links_.count(link_id) == 0)
      {
        links_.erase(link_id);
      }
    }
    pruneNodes();
  }

  Link Model::getLink(const int& link_id) const
  {
    if (links_.count(link_id) != 0)
    {
      return links_.at(link_id);
    }
    else
    {
      std::cout << "Link id " << link_id << " not found!" << std::endl;
    }
  }

  void Model::pruneNodes()
  {
    std::vector<int> node_ids;
    for (auto const& node : nodes_)
    {
      bool found = false;
      for (auto const& link: links_)
      {
        if (node.second == link.second.getSourceNode() ||
          node.second == link.second.getSinkNode())
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        node_ids.push_back(node.first);
      }
    }
    removeNodes(node_ids);
  }

  void Model::pruneLinks()
  {
    std::vector<int> link_ids;
    for (auto const& link: links_)
    {
      bool found = false;
      for (auto const& node : nodes_)
      {
        if (node.second == link.second.getSourceNode() ||
          node.second == link.second.getSinkNode())
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        link_ids.push_back(link.first);
      }
    }
    removeLinks(link_ids);
  }
}