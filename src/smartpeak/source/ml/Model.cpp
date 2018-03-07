/**TODO:  Add copyright*/

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

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
    for (const auto& node : nodes_)
    {
      bool found = false;
      if (links_.empty()) { return; }
      for (const auto& link: links_)
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
    for (const auto& link: links_)
    {
      bool found = false;
      if (nodes_.empty()) { return; }
      for (const auto& node : nodes_)
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
  
  void Model::getNextInactiveLayer(
    std::vector<Link>& links,
    std::vector<Node>& source_nodes,
    std::vector<Node>& sink_nodes) const
  {
    links.clear();
    source_nodes.clear();
    sink_nodes.clear();

    // get all links where the source node is active and the sink node is inactive
    // except for biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSourceNodeId()).getType() != NodeType::bias &&
        nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::activated && 
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::deactivated && 
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::deactivated &&
        // could use std::set instead to check for duplicates
        std::find(source_nodes.begin(), source_nodes.end(), nodes_.at(link_map.second.getSourceNodeId())) == source_nodes.end() &&
        std::find(sink_nodes.begin(), sink_nodes.end(), nodes_.at(link_map.second.getSinkNodeId())) == sink_nodes.end())
      {
        links.push_back(link_map.second);
        if 
        source_nodes.push_back(nodes_.at(link_map.second.getSourceNodeId()));
        sink_nodes.push_back(nodes_.at(link_map.second.getSinkNodeId()));
      }
    }

    // get all the biases for the sink nodes
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSourceNodeId()).getType() == NodeType::bias &&
        nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::activated && 
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::deactivated &&
        std::find(sink_nodes.begin(), sink_nodes.end(), nodes_.at(link_map.second.getSinkNodeId())) != sink_nodes.end())
      {
        links.push_back(link_map.second);
        source_nodes.push_back(nodes_.at(link_map.second.getSourceNodeId()));
      }
    }
  }

  // void Model::forwardPropogateLayerNetInput(
  //   std::vector<Link>& links,
  //   std::vector<Node>& source_nodes,
  //   std::vector<Node>& sink_nodes) const
  // {
  //   // construct the source, weight, and output tensors
  //   const int batch_size = source_nodes.getError().size();
  //   Eigen::TensorMap<Tensor<float, 2>> source_tensor(source_nodes.size(), batch_size);
  //   Eigen::Tensor<float, 2> weight_tensor(source_nodes.size(), sink_nodes.size());
  //   Eigen::Tensor<float, 2> sink_tensor(sink_nodes.size(), batch_size);
  // }
}