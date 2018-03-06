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
        // TODO: move to debug log
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
      // TODO: move to debug log
      std::cout << "Node id " << node_id << " not found!" << std::endl;
    }
  }

  void Model::removeNodes(const std::vector<int>& node_ids)
  { 
    for (int const& node_id: node_ids)
    {
      // check for duplicate nodes (by id)
      if (nodes_.count(node_id) != 0)
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
      }
      else
      {
        // TODO: move to debug log
        std::cout << "Link id " << link.getId() << " already exists!" << std::endl;
      }
    }
  }

  void Model::removeLinks(const std::vector<int>& link_ids)
  { 
    for (int const& link_id: link_ids)
    {
      // check for duplicate links (by id)
      if (links_.count(link_id) != 0)
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
      // TODO: move to debug log
      std::cout << "Link id " << link_id << " not found!" << std::endl;
    }
  }

  void Model::pruneNodes()
  {
    std::vector<int> node_ids;
    if (nodes_.empty()) { return; }
    for (auto const& node : nodes_)
    {
      bool found = false;
      if (links_.empty()) { return; }
      for (auto const& link: links_)
      {
        if (node.second.getId() == link.second.getSourceNodeId() ||
          node.second.getId() == link.second.getSinkNodeId())
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
    if (node_ids.size() != 0) { removeNodes(node_ids); }    
  }

  void Model::pruneLinks()
  {
    std::vector<int> link_ids;
    if (links_.empty()) { return; }
    for (auto const& link: links_)
    {
      bool found = false;
      if (nodes_.empty()) { return; }
      for (auto const& node : nodes_)
      {
        if (node.second.getId() == link.second.getSourceNodeId() ||
          node.second.getId() == link.second.getSinkNodeId())
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
    if (link_ids.size() != 0) { removeLinks(link_ids); }
  }

  
  void Model::getNextInactiveLayer(Eigen::Tensor<float, 2> weights, Eigen::Tensor<float, 2> nodes) const
  {
    // get all links where the source node is active and the sink node is inactive
    std::vector<Link> links;
    std::vector<Node> source_nodes, sink_nodes;
    for (const std::map<int, Link>& link_map : links_)
    {
      if (node_[link.second.getSourceNodeId()].getNodeStatus() == NodeStatus::activated && 
        link.second.getSinkNodeId()].getNodeStatus() == NodeStatus::deactivated)
      {
        links.push_back(link.second);
        source_nodes.push_back(node_[link.second.getSourceNodeId()]);
        sink_nodes.push_back(node_[link.second.getSinkNodeId()]);
      }
    }
    // construct the source, weight, and output tensors
    
  }
}