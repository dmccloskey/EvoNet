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
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::deactivated)
      {
        links.push_back(link_map.second);
        // could use std::set instead to check for duplicates
        if (std::find(source_nodes.begin(), source_nodes.end(), nodes_.at(link_map.second.getSourceNodeId())) == source_nodes.end())
        {
          source_nodes.push_back(nodes_.at(link_map.second.getSourceNodeId()));
        }
        if (std::find(sink_nodes.begin(), sink_nodes.end(), nodes_.at(link_map.second.getSinkNodeId())) == sink_nodes.end())
        {
          sink_nodes.push_back(nodes_.at(link_map.second.getSinkNodeId()));
        }
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
        // could use std::set instead to check for duplicates
        if (std::find(source_nodes.begin(), source_nodes.end(), nodes_.at(link_map.second.getSourceNodeId())) == source_nodes.end())
        {
          source_nodes.push_back(nodes_.at(link_map.second.getSourceNodeId()));
        }
      }
    }
  }

  void Model::initNodes(const int& batch_size)
  {
    Eigen::Tensor<float, 1> init_values(batch_size);
    init_values.setConstant(0.0f);
    for (auto& node_map : nodes_)
    {
      node_map.second.setOutput(init_values);
      node_map.second.setError(init_values);
      node_map.second.setDerivative(init_values);
      node_map.second.setStatus(NodeStatus::deactivated);
    }
  }
  
  void Model::mapValuesToNodes(
    const Eigen::Tensor<float, 2>& values,
    const std::vector<int>& node_ids)
  {
    // check dimension mismatches
    if (node_ids.size() != values.dimension(1))
    {
      std::cout << "The number of input features and the number of nodes do not match." << std::endl;
      return;
    }
    // assumes the node exists
    else if (nodes_.at(node_ids[0]).getOutput().size() != values.dimension(0))
    {
      std::cout << "The number of input samples and the node batch size does not match." << std::endl;
      return;
    }

    // copy over the input values
    for (int i=0; i<node_ids.size(); ++i)
    {
      for (int j=0; j<nodes_.at(node_ids[i]).getOutput().size(); ++j)
      {
        // SANITY CHECK:
        // std::cout << "i" << i << " j" << j << " values: " << values.data()[i*values.dimension(0) + j] << std::endl;
        nodes_.at(node_ids[i]).getOutputPointer()[j] = values.data()[i*values.dimension(0) + j];
        nodes_.at(node_ids[i]).setStatus(NodeStatus::activated);
      }
    }
  }

  void Model::forwardPropogateLayerNetInput(
    std::vector<Link>& links,
    std::vector<Node>& source_nodes,
    std::vector<Node>& sink_nodes)
  {
    // infer the batch size from the first source node
    const int batch_size = source_nodes[0].getOutput().size();

    // concatenate the source and weight tensors
    // using col-major ordering where rows are the batch vectors
    // and cols are the nodes

    // source_ptr
    float source_ptr [source_nodes.size() * batch_size];
    for (int i=0; i<source_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        source_ptr[i*batch_size + j] = source_nodes[i].getOutputPointer()[j];
      }
    }

    // weight_ptr
    float weight_ptr [batch_size * sink_nodes.size()];
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<source_nodes.size(); ++j)
      {
        for (const Link& link : links)
        {
          if (link.getSinkNodeId() == sink_nodes[i].getId() &&
          link.getSourceNodeId() == source_nodes[j].getId())
          {
            weight_ptr[i*source_nodes.size() + j] = link.getWeight();
            break;
          }
        }
      }
    }
    float sink_ptr [sink_nodes.size() * batch_size];
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        sink_ptr[i*batch_size + j] = sink_nodes[i].getOutputPointer()[j];
      }
    }

    // construct the source and weight tensors
    Eigen::TensorMap<Eigen::Tensor<float, 2>> source_tensor(source_ptr, batch_size, source_nodes.size());
    Eigen::TensorMap<Eigen::Tensor<float, 2>> weight_tensor(weight_ptr, source_nodes.size(), sink_nodes.size());
    Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_tensor(sink_ptr, batch_size, sink_nodes.size());

    // compute the output tensor
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    sink_tensor = source_tensor.contract(weight_tensor, product_dims);

    // update the sink nodes
    std::vector<int> node_ids;
    for (const Node& sink_node : sink_nodes)
    {
      node_ids.push_back(sink_node.getId());
    }
    mapValuesToNodes(sink_tensor, node_ids);
    // std::cout<<sink_tensor<<std::endl;
    // std::cout<<&sink_ptr[0]<<std::endl;
    // std::cout<<sink_ptr[0]<<std::endl;
    // std::cout<<&sink_nodes[0].getOutputPointer()[0]<<std::endl;
    // std::cout<<sink_nodes[0].getOutputPointer()[0]<<std::endl;
    // std::cout<<&getNode(sink_nodes[0].getId()).getOutputPointer()[0]<<std::endl;
    // std::cout<<getNode(sink_nodes[0].getId()).getOutputPointer()[0]<<std::endl;
  }
}