/**TODO:  Add copyright*/

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/LossFunction.h>

#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

namespace SmartPeak
{
  Model::Model()
  {        
  }

  Model::Model(const Model& other)
  {
    id_ = other.id_;
    name_ = other.name_;
    links_ = other.links_;
    nodes_ = other.nodes_;
    weights_ = other.weights_;
    error_ = other.error_;
    loss_function_ = other.loss_function_;
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
  
  void Model::setName(const std::string& name)
  {
    name_ = name;    
  }
  std::string Model::getName() const
  {
    return name_;
  }

  void Model::setError(const Eigen::Tensor<float, 1>& error)
  {
    error_ = error;
  }
  Eigen::Tensor<float, 1> Model::getError() const
  {
    return error_;
  }
  
  void Model::setLossFunction(const SmartPeak::ModelLossFunction& loss_function)
  {
    loss_function_ = loss_function;
  }
  SmartPeak::ModelLossFunction Model::getLossFunction() const
  {
    return loss_function_;
  }

  void Model::addNodes(const std::vector<Node>& nodes)
  { 
    for (const Node& node: nodes)
    {
      auto found = nodes_.emplace(node.getName(), node);
      if (!found.second)
      {
        // TODO: move to debug log
        std::cout << "Node name " << node.getName() << " already exists!" << std::endl;
      }
    }
  }

  Node Model::getNode(const std::string& node_name) const
  {
    if (!nodes_.empty() && nodes_.count(node_name) != 0)
    {
      return nodes_.at(node_name);
    }
    else
    {
      // TODO: move to debug log
      std::cout << "Node name " << node_name << " not found!" << std::endl;
    }
  }

  std::vector<Node> Model::getNodes() const
  {
    std::vector<Node> nodes;
    for (const auto& node: nodes_)
    {
      nodes.push_back(node.second);
    }
    return nodes;
  }

  void Model::removeNodes(const std::vector<std::string>& node_names)
  { 
    for (const std::string& node_name: node_names)
    {
      // check for duplicate nodes (by id)
      if (nodes_.count(node_name) != 0)
      {
        nodes_.erase(node_name);
      }
    }
    // pruneLinks(); // Allow for dangling links
  }


  void Model::addWeights(const std::vector<Weight>& weights)
  { 
    for (const Weight& weight: weights)
    {
      auto found = weights_.emplace(weight.getName(), std::move(weight));
      if (!found.second)
      {
        // TODO: move to debug log
        std::cout << "Weight name " << weight.getName() << " already exists!" << std::endl;
      }
    }
  }

  Weight Model::getWeight(const std::string& weight_name) const
  {
    if (!weights_.empty() && weights_.count(weight_name) != 0)
    {
      return std::move(weights_.at(weight_name));
    }
    else
    {
      // TODO: move to debug log
      std::cout << "Weight name " << weight_name << " not found!" << std::endl;
    }
  }
  
  std::vector<Weight> Model::getWeights() const
  {
    std::vector<Weight> weights;
    for (const auto& weight: weights_)
    {
      weights.push_back(weight.second);
    }
    return weights;
  }

  void Model::removeWeights(const std::vector<std::string>& weight_names)
  { 
    for (std::string const& weight_name: weight_names)
    {
      // check for duplicate weights (by id)
      if (weights_.count(weight_name) != 0)
      {
        weights_.erase(weight_name);
      }
    }
    pruneLinks();
  }

  void Model::addLinks(const std::vector<Link>& links)
  { 
    for (const Link& link: links)
    {
      auto found = links_.emplace(link.getName(), link);
      if (!found.second)
      {
        // TODO: move to debug log
        std::cout << "Link name " << link.getName() << " already exists!" << std::endl;
      }
    }
  }

  void Model::removeLinks(const std::vector<std::string>& link_names)
  { 
    for (const std::string& link_name: link_names)
    {
      // check for duplicate links (by id)
      if (links_.count(link_name) != 0)
      {
        links_.erase(link_name);
      }
    }
    // pruneNodes(); // Allow dangling nodes to exist
    //pruneWeights();  // testing
  }

  Link Model::getLink(const std::string& link_name) const
  {
    if (!links_.empty() && links_.count(link_name) != 0)
    {
      return links_.at(link_name);
    }
    else
    {
      // TODO: move to debug log
      std::cout << "Link name " << link_name << " not found!" << std::endl;
    }
  }

  std::vector<Link> Model::getLinks() const
  {
    std::vector<Link> links;
    for (const auto& link: links_)
    {
      links.push_back(link.second);
    }
    return links;
  }

  bool Model::pruneNodes()
  {
    std::vector<std::string> node_names;
    if (nodes_.empty()) { return false; }
    for (const auto& node : nodes_)
    {
      bool found = false;
      // if (links_.empty()) { found = true; }
      for (const auto& link: links_)
      {
        if (node.second.getName() == link.second.getSourceNodeName() ||
          node.second.getName() == link.second.getSinkNodeName())
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        node_names.push_back(node.first);
      }
    }
    if (node_names.size() != 0)
    {
      removeNodes(node_names); 
      return true;
    }  
    else
      return false;  
  }

  bool Model::pruneWeights()
  {
    std::vector<std::string> weight_names;
    if (weights_.empty()) { return false; }
    for (const auto& weight : weights_)
    {
      bool found = false;
      // if (links_.empty()) { found = true; }
      for (const auto& link: links_)
      {
        if (weight.second.getName() == link.second.getWeightName())
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        weight_names.push_back(weight.first);
      }
    }
    if (weight_names.size() != 0)
    { 
      removeWeights(weight_names);
      return true;
    }  
    else
      return false;     
  }

  bool Model::pruneLinks()
  {
    std::vector<std::string> link_names;
    if (links_.empty()) { return false; }
    for (const auto& link: links_)
    {
      bool source_node_found = false;
      bool sink_node_found = false;
      // if (nodes_.empty())
      // {
      //   source_node_found = true;
      //   sink_node_found = true;
      // }
      for (const auto& node : nodes_)
      {
        if (node.second.getName() == link.second.getSourceNodeName())
          source_node_found = true;
        if (node.second.getName() == link.second.getSinkNodeName())
          sink_node_found = true;
        if (source_node_found && sink_node_found)
          break;
      }
      bool weight_found = false;
      // if (weights_.empty()) { weight_found = true; }
      for (const auto& weight : weights_)
      {
        if (weight.second.getName() == link.second.getWeightName())
        {
          weight_found = true;
          break;
        }
      }
      if (!source_node_found || !sink_node_found)
      {
        link_names.push_back(link.first);
      }
    }
    if (link_names.size() != 0)
    {
      removeLinks(link_names);
      return true;
    }  
    else
      return false; 
  }

  void Model::pruneModel(int iterations)
  {
    int cnt = 0;
    while (pruneLinks() || pruneWeights() || pruneNodes())
    {
      if (cnt >= iterations) {break;}
      // std::cout<<"Pruning model iteration: "<<cnt<<std::endl;
      cnt += 1;
    }
  }

  void Model::initNodes(const int& batch_size, const int& memory_size)
  {
    for (auto& node_map : nodes_)
    {
      node_map.second.initNode(batch_size, memory_size);
    }
  }

  void Model::initWeights()
  {
    for (auto& weight_map : weights_)
    {
      weight_map.second.initWeight();
    }
  }
  
  void Model::mapValuesToNodes(
    const Eigen::Tensor<float, 1>& values,
    const int& memory_step,
    const NodeStatus& status_update,
    const std::string& value_type)
  {

    // copy over the input values
    for (auto& node_map : nodes_)
    {
      for (int j=0; j<values.dimension(0); ++j)
      {
        if (value_type == "output")
        {
          node_map.second.getOutputMutable()->operator()(j, memory_step) = values(j);
        }
        else if (value_type == "error")
        {
          node_map.second.getErrorMutable()->operator()(j, memory_step) = values(j);
        }
        else if (value_type == "dt")
        {
          node_map.second.getDtMutable()->operator()(j, memory_step) = values(j);
        }
        node_map.second.setStatus(status_update);
      }
    }
  }
  
  void Model::mapValuesToNodes(
    const Eigen::Tensor<float, 2>& values,
    const int& memory_step,
    const std::vector<std::string>& node_names,
    const NodeStatus& status_update,
    const std::string& value_type)
  {
    // check dimension mismatches
    if (node_names.size() != values.dimension(1))
    {
      std::cout << "The number of input features and the number of nodes do not match." << std::endl;
      return;
    }
    // assumes the node exists
    else if (nodes_.at(node_names[0]).getOutput().dimension(0) != values.dimension(0))
    {
      std::cout << "The number of input samples and the node batch size does not match." << std::endl;
      return;
    }
    // assumes the node exists
    else if (nodes_.at(node_names[0]).getOutput().dimension(1) <= memory_step)
    {
      std::cout << "The memory_step is greater than the memory_size." << std::endl;
      return;
    }

    // // infer the memory size from the node output size
    // const int memory_size = nodes_.at(node_names[0]).getOutput().dimension(1);

    // copy over the input values
    for (int i=0; i<node_names.size(); ++i)
    {
      for (int j=0; j<values.dimension(0); ++j)
      {
        if (value_type == "output")
        {
          // SANITY CHECK:
          // std::cout << "i" << i << " j" << j << " values: " << values.data()[i*values.dimension(0) + j] << std::endl;
          // nodes_.at(node_names[i]).getOutputPointer()[j + values.dimension(0)*memory_step] = std::move(values.data()[i*values.dimension(0) + j]);
          // nodes_.at(node_names[i]).getOutputPointer()[j + values.dimension(0)*memory_step] = values(j, i);
          nodes_.at(node_names[i]).getOutputMutable()->operator()(j, memory_step) = values(j, i);
        }
        else if (value_type == "error")
        {
          nodes_.at(node_names[i]).getErrorMutable()->operator()(j, memory_step) = values(j, i);
        }
        else if (value_type == "dt")
        {
          nodes_.at(node_names[i]).getDtMutable()->operator()(j, memory_step) = values(j, i);
        }
        nodes_.at(node_names[i]).setStatus(status_update);
      }
    }
  }
  
  void Model::mapValuesToNodes(
    const Eigen::Tensor<float, 3>& values,
    const std::vector<std::string>& node_names,
    const NodeStatus& status_update,
    const std::string& value_type)
  {
    // check dimension mismatches
    if (node_names.size() != values.dimension(2))
    {
      printf("The number of input features %d and the number of nodes %d do not match.\n", (int)values.dimension(2), node_names.size());
      return;
    }
    // assumes the node exists
    else if (nodes_.at(node_names[0]).getOutput().dimension(0) != values.dimension(0))
    {
      printf("The number of input samples %d and the node batch size %d does not match.\n", (int)values.dimension(0), (int)nodes_.at(node_names[0]).getOutput().dimension(0));
      return;
    }
    else if (nodes_.at(node_names[0]).getOutput().dimension(1) != values.dimension(1))
    {
      printf("The number of input time steps %d and the node memory size %d does not match.\n", (int)values.dimension(1), (int)nodes_.at(node_names[0]).getOutput().dimension(1));
      return;
    }

    // copy over the input values
    for (int i=0; i<node_names.size(); ++i)
    {
      for (int k=0; k<values.dimension(1); ++k)
      {
        for (int j=0; j<values.dimension(0); ++j)
        {
          if (value_type == "output")
          {
            // nodes_.at(node_names[i]).getOutputPointer()[k*values.dimension(0) + j] = values(j, k, i);
            nodes_.at(node_names[i]).getOutputMutable()->operator()(j, k) = values(j, k, i);
          }
          else if (value_type == "error")
          {
            nodes_.at(node_names[i]).getErrorMutable()->operator()(j, k) = values(j, k, i);
          }
          else if (value_type == "dt")
          {
            nodes_.at(node_names[i]).getDtMutable()->operator()(j, k) = values(j, k, i);
          }
          nodes_.at(node_names[i]).setStatus(status_update);
        }
      }
    }
  }
  
  void Model::getNextInactiveLayer(
    std::map<std::string, std::vector<std::string>>& sink_links_map)
  {

    // get all links where the source node is active and the sink node is inactive
    // except for biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSourceNodeName()).getType() != NodeType::bias &&
        nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::activated && 
        nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::initialized)
      {
        std::vector<std::string> links = {link_map.second.getName()};
        auto found = sink_links_map.emplace(link_map.second.getSinkNodeName(), links);        
        if (!found.second)
        {
          sink_links_map[link_map.second.getSinkNodeName()].push_back(link_map.second.getName());
        }
      }
    }
  }  
  
  void Model::getNextInactiveLayer(
    std::vector<std::string>& links,
    std::vector<std::string>& source_nodes,
    std::vector<std::string>& sink_nodes)
  {
    links.clear();
    source_nodes.clear();
    sink_nodes.clear();

    // get all links where the source node is active and the sink node is inactive
    // except for biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSourceNodeName()).getType() != NodeType::bias &&
        nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::activated && 
        nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::initialized)
      {
        // std::cout << "Model::getNextInactiveLayer() link_name: " << link_map.second.getName() << std::endl;
        // std::cout << "Model::getNextInactiveLayer() source_node_name: " << link_map.second.getSourceNodeName() << std::endl;
        // std::cout << "Model::getNextInactiveLayer() sink_node_name: " << link_map.second.getSinkNodeName() << std::endl;
        links.push_back(link_map.second.getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSourceNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second.getSourceNodeName());
        }
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSinkNodeName()) == 0)
        {
          sink_nodes.push_back(link_map.second.getSinkNodeName());
        }
      }
    }
  }
  
  void Model::getNextInactiveLayerBiases(
    std::map<std::string, std::vector<std::string>>& sink_links_map,
    std::vector<std::string>& sink_nodes_with_biases)
  {

    // get all the biases for the sink nodes
    for (auto& link_map : links_)
    {
      if (        
        // does not allow for cycles
        nodes_.at(link_map.second.getSourceNodeName()).getType() == NodeType::bias && 
        nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::activated &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::initialized &&
        sink_links_map.count(link_map.second.getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        sink_links_map[link_map.second.getSinkNodeName()].push_back(link_map.second.getName());
        sink_nodes_with_biases.push_back(link_map.second.getSinkNodeName());
      }
    }
  }
  
  void Model::getNextInactiveLayerBiases(
    std::vector<std::string>& links,
    std::vector<std::string>& source_nodes,
    const std::vector<std::string>& sink_nodes,
    std::vector<std::string>& sink_nodes_with_biases)
  {

    // get all the biases for the sink nodes
    for (auto& link_map : links_)
    {
      if (        
        // does not allow for cycles
        nodes_.at(link_map.second.getSourceNodeName()).getType() == NodeType::bias && 
        nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::activated &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::initialized &&
        std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        // std::cout << "Model::getNextInactiveLayerBiases() link_name: " << link_map.second.getName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerBiases() source_node_name: " << link_map.second.getSourceNodeName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerBiases() sink_node_name: " << link_map.second.getSinkNodeName() << std::endl;
        links.push_back(link_map.second.getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSourceNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second.getSourceNodeName());
        }
        if (std::count(sink_nodes_with_biases.begin(), sink_nodes_with_biases.end(), link_map.second.getSinkNodeName()) == 0)
        {
          sink_nodes_with_biases.push_back(link_map.second.getSinkNodeName());
        }
      }
    }
  }
  
  void Model::getNextInactiveLayerCycles(
    std::map<std::string, std::vector<std::string>>& sink_links_map,
    std::vector<std::string>& sink_nodes_with_cycles)
  {

    // get cyclic source nodes
    for (auto& link_map : links_)
    {
      if (
        (nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::initialized) &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::initialized &&
        sink_links_map.count(link_map.second.getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        sink_links_map[link_map.second.getSinkNodeName()].push_back(link_map.second.getName());
        sink_nodes_with_cycles.push_back(link_map.second.getSinkNodeName());
      }
    }
  }
  
  void Model::getNextInactiveLayerCycles(
    std::vector<std::string>& links,
    std::vector<std::string>& source_nodes,
    const std::vector<std::string>& sink_nodes,
    std::vector<std::string>& sink_nodes_with_cycles)
  {

    // get cyclic source nodes
    for (auto& link_map : links_)
    {
      if (
        (nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::initialized) &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::initialized &&
        std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        // std::cout << "Model::getNextInactiveLayerCycles() link_name: " << link_map.second.getName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerCycles() source_node_name: " << link_map.second.getSourceNodeName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerCycles() sink_node_name: " << link_map.second.getSinkNodeName() << std::endl;
        links.push_back(link_map.second.getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSourceNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second.getSourceNodeName());
        }
        if (std::count(sink_nodes_with_cycles.begin(), sink_nodes_with_cycles.end(), link_map.second.getSinkNodeName()) == 0)
        {
          sink_nodes_with_cycles.push_back(link_map.second.getSinkNodeName());
        }
      }
    }
  }

  void Model::forwardPropogateLayerNetInput(
    std::map<std::string, std::vector<std::string>>& sink_links_map,
    const int& time_step)
  {

    // get all the information needed to construct the tensors
    int batch_size = 0;
    int memory_size = 0;
    for (const auto& sink_links : sink_links_map)
    {
      batch_size = nodes_.at(sink_links.first).getOutput().dimension(0);
      memory_size = nodes_.at(sink_links.first).getOutput().dimension(1);
      break;
    }

    // iterate through each sink node and calculate the net input
    // invoke the activation function once the net input is calculated
    for (const auto& sink_links : sink_links_map)
    {
      Eigen::Tensor<float, 2> sink_tensor(batch_size, 1);
      sink_tensor.setConstant(0.0f);
      Eigen::Tensor<float, 2> weight_tensor(batch_size, 1);
      for (const std::string& link : sink_links.second)
      {
        weight_tensor.setConstant(weights_.at(links_.at(link).getWeightName()).getWeight());
        if (nodes_.at(links_.at(link).getSourceNodeName()).getStatus() == NodeStatus::activated)
        {
          Eigen::array<int, 2> offsets = {0, time_step};
          Eigen::array<int, 2> extent = {batch_size, time_step + 1};
          sink_tensor = sink_tensor + weight_tensor * nodes_.at(links_.at(link).getSourceNodeName()).getOutput().slice(offsets, extent); //current time-step
        }
        else if (nodes_.at(links_.at(link).getSourceNodeName()).getStatus() == NodeStatus::initialized)
        {
          Eigen::array<int, 2> offsets = {0, time_step + 1};
          Eigen::array<int, 2> extent = {batch_size, time_step + 2};
          if (time_step + 1 < memory_size)
          {
            sink_tensor = sink_tensor + weight_tensor * nodes_.at(links_.at(link).getSourceNodeName()).getOutput().slice(offsets, extent); //previous time-step
          }
          else
          {
            std::cout<<"time_step exceeded memory size in forwardPropogateLayerNetInput."<<std::endl;
          }
        }
      }

      // update the sink output
      mapValuesToNodes(sink_tensor, time_step, {sink_links.first}, NodeStatus::activated, "output");

      // calculate the output and derivative
      nodes_.at(sink_links.first).calculateActivation(time_step);
      nodes_.at(sink_links.first).calculateDerivative(time_step);
    }
  }

  void Model::forwardPropogateLayerNetInput(
    const std::vector<std::string>& links,
    const std::vector<std::string>& source_nodes,
    const std::vector<std::string>& sink_nodes,
    const int& time_step)
  {
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(source_nodes[0]).getOutput().dimension(0);
    const int memory_size = nodes_.at(source_nodes[0]).getOutput().dimension(1);

    if (time_step >= memory_size)
    {
      std::cout<<"time step: "<<time_step<<" exceeds the memory_size!"<<std::endl;
      return;
    }

    // concatenate the source and weight tensors
    // using col-major ordering where rows are the batch vectors
    // and cols are the nodes

    // construct the source and weight tensors
    Eigen::Tensor<float, 2> source_tensor(batch_size, source_nodes.size());
    for (int i=0; i<source_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        if (nodes_.at(source_nodes[i]).getStatus() == NodeStatus::activated)
        {
          source_tensor(j, i) = nodes_.at(source_nodes[i]).getOutput()(j, time_step); //current time-step
        }
        else if (nodes_.at(source_nodes[i]).getStatus() == NodeStatus::initialized)
        {
          if (time_step + 1 < memory_size)
          {
            source_tensor(j, i) = nodes_.at(source_nodes[i]).getOutput()(j, time_step + 1); //previous time-step
          }
          else
          {
            std::cout<<"time_step exceeded memory size in forwardPropogateLayerNetInput."<<std::endl;
            source_tensor(j, i) = 0.0;
          }
        }
      }
    }

    Eigen::Tensor<float, 2> weight_tensor(source_nodes.size(), sink_nodes.size());
    // [NOTE: High CPU demand identified here...]
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<source_nodes.size(); ++j)
      {
        for (const std::string& link : links)
        {
          if (links_.at(link).getSinkNodeName() == sink_nodes[i] &&
          links_.at(link).getSourceNodeName() == source_nodes[j])
          {
            weight_tensor(j, i) = weights_.at(links_.at(link).getWeightName()).getWeight();
            break;
          }
        }
      }
    }

    // compute the output tensor
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<float, 2> sink_tensor = source_tensor.contract(weight_tensor, product_dims);

    // update the sink nodes
    mapValuesToNodes(sink_tensor, time_step, sink_nodes, NodeStatus::activated, "output");
  }
  
  void Model::forwardPropogateLayerActivation(
    const std::vector<std::string>& sink_nodes,
      const int& time_step)
  {
    for (const std::string& node : sink_nodes)
    {
      nodes_.at(node).calculateActivation(time_step);
      nodes_.at(node).calculateDerivative(time_step);
    }
  }

  void Model::cacheFPLayerNodeAndLinkNames(
	const int& time_step,
	std::vector<std::string>& links_O,
	std::vector<std::string>& source_nodes_O,
	std::vector<std::string>& sink_nodes_O)
  {
    const int max_iters = 1e6;
    for (int iter = 0; iter<max_iters; ++iter)
    {
      // std::cout<<"Model::forwardPropogate() iter: "<<iter<<std::endl;

      // get the next hidden layer
      std::vector<std::string> links, source_nodes, sink_nodes;
      getNextInactiveLayer(links, source_nodes, sink_nodes);
      // std::cout<<"Model::forwardPropogate() getNextInactiveLayer links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      // get biases,
      std::vector<std::string> sink_nodes_with_biases;
      getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);
      // std::cout<<"Model::forwardPropogate() getNextInactiveLayerBiases links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      // get cycles
      std::vector<std::string> links_cycles, source_nodes_cycles, sink_nodes_cycles;
      getNextInactiveLayerCycles(links_cycles, source_nodes_cycles, sink_nodes, sink_nodes_cycles);
      // std::cout<<"Model::forwardPropogate() getNextInactiveLayerCycles links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink_nodes_cycles: "<<sink_nodes_cycles.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() links: "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      if (sink_nodes_cycles.size() == sink_nodes.size())
      { // all forward propogation steps have caught up
        // add sink nodes with cycles to the forward propogation step
        links.insert(links.end(), links_cycles.begin(), links_cycles.end());
        source_nodes.insert(source_nodes.end(), source_nodes_cycles.begin(), source_nodes_cycles.end());
      }
      else
      { // remove source/sink nodes with cycles from the forward propogation step
        for (const std::string node_name : sink_nodes_cycles)
        {
          sink_nodes.erase(std::remove(sink_nodes.begin(), sink_nodes.end(), node_name), sink_nodes.end());
        }
      }

      // check if all nodes have been activated
      if (links.size() == 0)
      {
        break;
      }
      // std::cout<<"Model::forwardPropogate() final links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      // [OPTIMIZATION: for now, all nodes must have the same activation function]
      // // split sink nodes and links by activation function
      // std::map<NodeActivation, std::vector<std::string>> source_nodes_map;
      // std::map<NodeActivation, std::vector<std::string>> sink_nodes_map;
      // std::map<NodeActivation, std::vector<std::string>> links_map;
      // int index = 0;
      // for (const std::string& node_name : sink_nodes)
      // {
      //   std::vector<std::string> node_names = { node_name };
      //   auto found = sink_nodes_map.emplace(nodes_[node_name].getActivation(), node_names);
      //   if (!found.second)
      //   {
      //     sink_nodes_map[nodes_[node_name].getActivation()].push_back(node_name);
      //   }
      // }

      // change the node activation status
      // [NOTE: is this needed or can we just check if the node is already found?]
    }
  }
  
  void Model::forwardPropogate_test(const int& time_step)
  {
    const int max_iters = 1e6;
    for (int iter=0; iter<max_iters; ++iter)
    {      
      // std::cout<<"Model::forwardPropogate() iter: "<<iter<<std::endl;

      // get the next hidden layer
      std::map<std::string, std::vector<std::string>> sink_links_map;
      getNextInactiveLayer(sink_links_map);

      // get biases,
      std::vector<std::string> sink_nodes_with_biases;
      getNextInactiveLayerBiases(sink_links_map, sink_nodes_with_biases);
      
      // get cycles
      std::map<std::string, std::vector<std::string>> sink_links_map_cycles = sink_links_map;
      std::vector<std::string> sink_nodes_cycles;
      getNextInactiveLayerCycles(sink_links_map_cycles, sink_nodes_cycles);

      if (sink_links_map_cycles.size() == sink_links_map.size())
      { // all forward propogation steps have caught up
        // add sink nodes with cycles to the forward propogation step
        sink_links_map = sink_links_map_cycles;
      }

      // check if all nodes have been activated
      if (sink_links_map.size() == 0)
      {
        break;
      }

      // calculate the net input
      forwardPropogateLayerNetInput(sink_links_map, time_step);
    }
  }
  
  void Model::forwardPropogate(const int& time_step)
  {
    const int max_iters = 1e6;
    for (int iter=0; iter<max_iters; ++iter)
    {      
      // std::cout<<"Model::forwardPropogate() iter: "<<iter<<std::endl;

      // get the next hidden layer
      std::vector<std::string> links, source_nodes, sink_nodes;
      getNextInactiveLayer(links, source_nodes, sink_nodes);
      // std::cout<<"Model::forwardPropogate() getNextInactiveLayer links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      // get biases,
      std::vector<std::string> sink_nodes_with_biases;
      getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);
      // std::cout<<"Model::forwardPropogate() getNextInactiveLayerBiases links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;
      
      // get cycles
      std::vector<std::string> links_cycles, source_nodes_cycles, sink_nodes_cycles;
      getNextInactiveLayerCycles(links_cycles, source_nodes_cycles, sink_nodes, sink_nodes_cycles);
      // std::cout<<"Model::forwardPropogate() getNextInactiveLayerCycles links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink_nodes_cycles: "<<sink_nodes_cycles.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() links: "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      if (sink_nodes_cycles.size() == sink_nodes.size())
      { // all forward propogation steps have caught up
        // add sink nodes with cycles to the forward propogation step
        links.insert( links.end(), links_cycles.begin(), links_cycles.end() );
        source_nodes.insert( source_nodes.end(), source_nodes_cycles.begin(), source_nodes_cycles.end() );
      }
      else
      { // remove source/sink nodes with cycles from the forward propogation step
        for (const std::string node_name : sink_nodes_cycles)
        {
          sink_nodes.erase(std::remove(sink_nodes.begin(), sink_nodes.end(), node_name), sink_nodes.end());
        }
      }

      // check if all nodes have been activated
      if (links.size() == 0)
      {
        break;
      }      
      // std::cout<<"Model::forwardPropogate() final links, source, and sink sizes "<<std::endl;
      // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      // calculate the net input
      forwardPropogateLayerNetInput(links, source_nodes, sink_nodes, time_step);

      // calculate the activation
      forwardPropogateLayerActivation(sink_nodes, time_step);
    }
  }

  void Model::FPTT_test(const int& time_steps, 
    const Eigen::Tensor<float, 3>& values,
    const std::vector<std::string> node_names,
    const Eigen::Tensor<float, 2>& dt)
  {
    // check time_steps vs memory_size
    int max_steps = time_steps;
    if (time_steps > nodes_.begin()->second.getOutput().dimension(1))
    {
      std::cout<<"Time_steps will be scaled back to the memory_size."<<std::endl;
      max_steps = nodes_.begin()->second.getOutput().dimension(1);
    }

    for (int time_step=0; time_step<max_steps; ++time_step)
    {
      // std::cout<<"Model::FPTT() time_step: "<<time_step<<std::endl;
      if (time_step>0)
      {
        // move to the next memory step
        for (auto& node_map: nodes_)
        {          
          node_map.second.saveCurrentOutput();
          node_map.second.saveCurrentDerivative();
          node_map.second.saveCurrentDt();
          if (std::count(node_names.begin(), node_names.end(), node_map.first) == 0)
          {
            node_map.second.setStatus(NodeStatus::initialized); // reinitialize non-input nodes
          }   
          // std::cout<<"Model::FPTT() output: "<<node_map.second.getOutput()<<" for node_name: "<<node_map.first<<std::endl;
        }
      }

      // initialize nodes for the next time-step
      const Eigen::Tensor<float, 1> dt_values = dt.chip(time_step, 1);
      mapValuesToNodes(dt_values, 0, NodeStatus::initialized, "dt");
      const Eigen::Tensor<float, 2> active_values = values.chip(time_step, 1);
      // std::cout<<"Model::FPTT() active_values: "<<active_values<<std::endl;
      mapValuesToNodes(active_values, 0, node_names, NodeStatus::activated, "output");

      forwardPropogate_test(0); // always working at the current head of memory
    }
  }

  void Model::FPTT(const int& time_steps, 
    const Eigen::Tensor<float, 3>& values,
    const std::vector<std::string> node_names,
    const Eigen::Tensor<float, 2>& dt)
  {
    // check time_steps vs memory_size
    int max_steps = time_steps;
    if (time_steps > nodes_.begin()->second.getOutput().dimension(1))
    {
      std::cout<<"Time_steps will be scaled back to the memory_size."<<std::endl;
      max_steps = nodes_.begin()->second.getOutput().dimension(1);
    }

    for (int time_step=0; time_step<max_steps; ++time_step)
    {
      // std::cout<<"Model::FPTT() time_step: "<<time_step<<std::endl;
      if (time_step>0)
      {
        // move to the next memory step
        for (auto& node_map: nodes_)
        {          
          node_map.second.saveCurrentOutput();
          node_map.second.saveCurrentDerivative();
          node_map.second.saveCurrentDt();
          if (std::count(node_names.begin(), node_names.end(), node_map.first) == 0)
          {
            node_map.second.setStatus(NodeStatus::initialized); // reinitialize non-input nodes
          }   
          // std::cout<<"Model::FPTT() output: "<<node_map.second.getOutput()<<" for node_name: "<<node_map.first<<std::endl;
        }
      }

      // initialize nodes for the next time-step
      const Eigen::Tensor<float, 1> dt_values = dt.chip(time_step, 1);
      mapValuesToNodes(dt_values, 0, NodeStatus::initialized, "dt");
      const Eigen::Tensor<float, 2> active_values = values.chip(time_step, 1);
      // std::cout<<"Model::FPTT() active_values: "<<active_values<<std::endl;
      mapValuesToNodes(active_values, 0, node_names, NodeStatus::activated, "output");

      forwardPropogate(0); // always working at the current head of memory
    }
  }
  
  void Model::calculateError(
    const Eigen::Tensor<float, 2>& values, const std::vector<std::string>& node_names)
  {
    //TODO: encapsulate into a seperate method
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(node_names[0]).getOutput().dimension(0);

    //TODO: encapsulate into a seperate method
    // check dimension mismatches
    if (node_names.size() != values.dimension(1))
    {
      std::cout << "The number of output features and the number of nodes do not match." << std::endl;
      return;
    }
    // assumes the node exists
    else if (batch_size != values.dimension(0))
    {
      std::cout << "The number of output samples and the node batch size does not match." << std::endl;
      return;
    }

    // make the tensor for the calculated model output
    // float node_ptr [node_names.size() * batch_size];
    // for (int i=0; i<node_names.size(); ++i)
    // {
    //   for (int j=0; j<batch_size; ++j)
    //   {
    //     node_ptr[i*batch_size + j] = nodes_.at(node_names[i]).getOutputPointer()[j];
    //   }
    // }
    // Eigen::TensorMap<Eigen::Tensor<float, 2>> node_tensor(node_ptr, batch_size, node_names.size());
    Eigen::Tensor<float, 2> node_tensor(batch_size, node_names.size());
    for (int i=0; i<node_names.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        node_tensor(j, i) = nodes_.at(node_names[i]).getOutput()(j, 0); // current time-step
      }
    }

    // calculate the model error wrt the expected model output
    Eigen::Tensor<float, 2> error_tensor(batch_size, node_names.size());
    switch (loss_function_)
    {
      case ModelLossFunction::EuclideanDistance:
      {
        EuclideanDistanceOp<float> operation;
        error_ = operation(node_tensor, values);
        EuclideanDistanceGradOp<float> gradient;
        error_tensor = gradient(node_tensor, values);
        break;
      } 
      case ModelLossFunction::L2Norm:
      {
        L2NormOp<float> operation;
        error_ = operation(node_tensor, values);
        L2NormGradOp<float> gradient;
        error_tensor = gradient(node_tensor, values);
        break;
      }
      case ModelLossFunction::CrossEntropy:
      {
        CrossEntropyOp<float> operation;
        error_ = operation(node_tensor, values);
        CrossEntropyGradOp<float> gradient;
        error_tensor = gradient(node_tensor, values);
        break;
      }
      case ModelLossFunction::NegativeLogLikelihood:
      {
        NegativeLogLikelihoodOp<float> operation;
        error_ = operation(node_tensor, values);
        NegativeLogLikelihoodGradOp<float> gradient;
        error_tensor = gradient(node_tensor, values);
        break;
      }
      case ModelLossFunction::MSE:
      {
        MSEOp<float> operation;
        error_ = operation(node_tensor, values);
        MSEGradOp<float> gradient;
        error_tensor = gradient(node_tensor, values);
        break;
      }
      default:
      {
        std::cout << "Loss Function not supported." << std::endl;
        break;
      }
    }

    // update the output node errors
    mapValuesToNodes(error_tensor, 0, node_names, NodeStatus::corrected, "error");
  }
  
  void Model::getNextUncorrectedLayer(
    std::vector<std::string>& links,
    std::vector<std::string>& source_nodes,
    std::vector<std::string>& sink_nodes)
  {
    links.clear();
    source_nodes.clear();
    sink_nodes.clear();

    // get all links where the source node is corrected and the sink node is active
    // including biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::corrected && 
        nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::activated)
      {
        links.push_back(link_map.second.getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSinkNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second.getSinkNodeName());
        }
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSourceNodeName()) == 0)
        {
          sink_nodes.push_back(link_map.second.getSourceNodeName());
        }
      }
    }
  }
  
  void Model::getNextUncorrectedLayerCycles(
    std::vector<std::string>& links,
    const std::vector<std::string>& source_nodes,
    std::vector<std::string>& sink_nodes,
    std::vector<std::string>& source_nodes_with_cycles)
  {

    // allows for cycles
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSourceNodeName()).getStatus() == NodeStatus::corrected && 
        std::count(links.begin(), links.end(), link_map.second.getName()) == 0 && // unique links 
        nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::corrected &&
        std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSinkNodeName()) != 0 // sink node has already been identified)
      ) 
      {
        links.push_back(link_map.second.getName());
        // could use std::set instead to check for duplicates
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSourceNodeName()) == 0)
        {
          sink_nodes.push_back(link_map.second.getSourceNodeName());
        }
        if (std::count(source_nodes_with_cycles.begin(), source_nodes_with_cycles.end(), link_map.second.getSinkNodeName()) == 0)
        {
          source_nodes_with_cycles.push_back(link_map.second.getSinkNodeName());
        }
      }
    }
  }

  void Model::backPropogateLayerError(
    const std::vector<std::string>& links,
    const std::vector<std::string>& source_nodes,
    const std::vector<std::string>& sink_nodes,
    const int& time_step)
  {
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(source_nodes[0]).getOutput().dimension(0);
    const int memory_size = nodes_.at(source_nodes[0]).getOutput().dimension(1);

    if (time_step >= memory_size)
    {
      std::cout<<"time step: "<<time_step<<" exceeds the memory_size!"<<std::endl;
      return;
    }

    // concatenate the source and weight tensors
    // using col-major ordering where rows are the batch vectors
    // and cols are the nodes

    // construct the source tensors
    Eigen::Tensor<float, 2> source_tensor(batch_size, source_nodes.size());
    for (int i=0; i<source_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        source_tensor(j, i) = nodes_.at(source_nodes[i]).getError()(j, time_step); // current time-step
      }
    }

    // construct the weight tensors
    Eigen::Tensor<float, 2> weight_tensor(source_nodes.size(), sink_nodes.size());
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<source_nodes.size(); ++j)
      {
        for (const std::string& link : links)
        {
          if (links_.at(link).getSourceNodeName() == sink_nodes[i] &&
          links_.at(link).getSinkNodeName() == source_nodes[j])
          {
            weight_tensor(j, i) = weights_.at(links_.at(link).getWeightName()).getWeight();
            break;
          }
        }
      }
    }
    
    // construct the derivative tensors
    std::vector<std::string> sink_nodes_prev;
    Eigen::Tensor<float, 2> derivative_tensor(batch_size, sink_nodes.size());
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        if (nodes_.at(sink_nodes[i]).getStatus() == NodeStatus::activated)
        {
          derivative_tensor(j, i) = nodes_.at(sink_nodes[i]).getDerivative()(j, time_step); // current time-step
        }
        else if (nodes_.at(sink_nodes[i]).getStatus() == NodeStatus::corrected)
        {
          // std::cout << "Model::backPropogateLayerError() Previous derivative (batch_size, Sink) " << j << "," << i << std::endl;
          if (time_step + 1 < memory_size)
          {
            derivative_tensor(j, i) = nodes_.at(sink_nodes[i]).getDerivative()(j, time_step + 1); // previous time-step
          }
          else
          {
            derivative_tensor(j, i) = 0.0; // previous time-step
          }
          
          if (std::count(sink_nodes_prev.begin(), sink_nodes_prev.end(), sink_nodes[i]) == 0) sink_nodes_prev.push_back(sink_nodes[i]);
        }        
      }
    }

    // compute the output tensor
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<float, 2> sink_tensor = source_tensor.contract(weight_tensor, product_dims) * derivative_tensor;

    if (sink_nodes_prev.size()>0)
    {
      // split the sink_tensor into current and previous
      Eigen::Tensor<float, 2> sink_tensor_cur(batch_size, sink_nodes.size() - sink_nodes_prev.size());
      Eigen::Tensor<float, 2> sink_tensor_prev(batch_size, sink_nodes_prev.size());
      int sink_tensor_cur_iter = 0;
      int sink_tensor_prev_iter = 0;
      std::vector<std::string> sink_nodes_cur;
      for (int i=0; i<sink_nodes.size(); ++i)
      {
        if (std::count(sink_nodes_prev.begin(), sink_nodes_prev.end(), sink_nodes[i]) != 0)
        {
          // std::cout<<"Model::backPropogateLayerError() sink_name is prev: "<<i<<std::endl;
          for (int j=0; j<batch_size; ++j)
          {
            sink_tensor_prev(j, sink_tensor_prev_iter) = sink_tensor(j, i);
          }
          sink_tensor_prev_iter += 1;
        }
        else
        {
          // std::cout<<"Model::backPropogateLayerError() sink_name is cur: "<<i<<std::endl;
          for (int j=0; j<batch_size; ++j)
          {
            sink_tensor_cur(j, sink_tensor_cur_iter) = sink_tensor(j, i);            
          }
          sink_tensor_cur_iter += 1;
          sink_nodes_cur.push_back(sink_nodes[i]);
        }
      }

      // update the sink nodes errors for the current time-step
      mapValuesToNodes(sink_tensor_cur, time_step, sink_nodes_cur, NodeStatus::corrected, "error");

      // update the sink nodes errors for the previous time-step
      if (time_step + 1 < memory_size)
      {
        mapValuesToNodes(sink_tensor_prev, time_step + 1, sink_nodes_prev, NodeStatus::corrected, "error");
      }
    }
    else
    {
      // update the sink nodes errors for the current time-step
      mapValuesToNodes(sink_tensor, time_step, sink_nodes, NodeStatus::corrected, "error");
    }
  }
  
  std::vector<std::string> Model::backPropogate(const int& time_step)
  {
    std::vector<std::string> node_names_with_cycles;
    const int max_iters = 1e6;
    for (int iter=0; iter<max_iters; ++iter)
    {
      // std::cout<<"Model::backPropogate() iter :"<<iter<<::std::endl;

      // std::cout<<"Model::backPropogate() NodeStatuses :";
      // for (const auto& node_map : nodes_)
      // {
      //   if (node_map.second.getStatus() == NodeStatus::activated) std::cout<<"Node status for Node ID: "<<node_map.first<<" is Activated"<<std::endl;
      //   if (node_map.second.getStatus() == NodeStatus::corrected) std::cout<<"Node status for Node ID: "<<node_map.first<<" is Corrected"<<std::endl;
      // }

      // get the next uncorrected layer
      std::vector<std::string> links, source_nodes, sink_nodes;
      getNextUncorrectedLayer(links, source_nodes, sink_nodes);
      // std::cout<<"link size "<<links.size()<<::std::endl;

      // get cycles
      std::vector<std::string> links_cycles, source_nodes_cycles, sink_nodes_cycles;
      getNextUncorrectedLayerCycles(links_cycles, source_nodes, sink_nodes_cycles, source_nodes_cycles);

      // std::cout << "Back Propogate cycles found: " << source_nodes_cycles.size() << std::endl;
      if (source_nodes_cycles.size() == source_nodes.size())
      { // all backward propogation steps have caught up
        // add source nodes with cycles to the backward propogation step
        links.insert( links.end(), links_cycles.begin(), links_cycles.end() );
        sink_nodes.insert( sink_nodes.end(), sink_nodes_cycles.begin(), sink_nodes_cycles.end() );
        node_names_with_cycles.insert(node_names_with_cycles.end(), sink_nodes_cycles.begin(), sink_nodes_cycles.end() );
      }
      else
      { // remove source/sink nodes with cycles from the backward propogation step
        for (const std::string node_name : source_nodes_cycles)
        {
          source_nodes.erase(std::remove(source_nodes.begin(), source_nodes.end(), node_name), source_nodes.end());
        }
      }

      // std::cout<<"Model::backPropogate() links.size()[after cycles] :"<<links.size()<<::std::endl;
      // check if all nodes have been corrected
      if (links.size() == 0)
      {
        break;
      }

      // calculate the net input
      backPropogateLayerError(links, source_nodes, sink_nodes, time_step);
    }
    return node_names_with_cycles;
  }

  void Model::TBPTT(const int& time_steps)
  {
    // check time_steps vs memory_size
    int max_steps = time_steps;
    if (time_steps >= nodes_.begin()->second.getOutput().dimension(1))
    {
      std::cout<<"Time_steps will be scaled back to the memory_size - 1."<<std::endl;
      max_steps = nodes_.begin()->second.getOutput().dimension(1) - 1;
    }

    std::vector<std::string> node_names; // determined at time step 0
    for (int time_step=0; time_step<max_steps; ++time_step)
    {
      // std::cout<<"Model::TBPTT() time_step: "<<time_step<<std::endl;
      if (time_step > 0 && node_names.size()>0)
      {
        for (auto& node_map: nodes_)
        {
          if (std::count(node_names.begin(), node_names.end(), node_map.first) == 0)
          {
            node_map.second.setStatus(NodeStatus::activated); // reinitialize cyclic nodes
          }   
          // std::cout<<"Model::TBPTT() output: "<<node_map.second.getError()<<" for node_name: "<<node_map.first<<std::endl;
        }
      }

      // calculate the error for each batch of memory
      // backPropogate(time_step);
      node_names = backPropogate(time_step);
    }
    // for (auto& node_map: nodes_)
    // {
    //   std::cout<<"Model::TBPTT() error: "<<node_map.second.getError()<<" for node_name: "<<node_map.first<<std::endl;
    // }
  }

  void Model::updateWeights(const int& time_steps)
  {
    // check time_steps vs memory_size
    int max_steps = time_steps;
    if (time_steps > nodes_.begin()->second.getOutput().dimension(1))
    {
      std::cout<<"Time_steps will be scaled back to the memory_size."<<std::endl;
      max_steps = nodes_.begin()->second.getOutput().dimension(1);
    }

    std::map<std::string, std::vector<float>> weight_derivatives;  
    // initalize the map
    for (const auto& weight_map: weights_)  
    {
      const std::vector<float> derivatives;
      weight_derivatives.emplace(weight_map.first, derivatives);
    }

    // collect the derivative for all weights
    for (const auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSinkNodeName()).getStatus() == NodeStatus::corrected)      
      {
        // Sum the error from current and previous time-steps
        float error_sum = 0.0;
        for (int i=0; i<max_steps; ++i)
        {
          Eigen::Tensor<float, 1> error_tensor = nodes_.at(link_map.second.getSinkNodeName()).getError().chip(0, 1); // first time-step
          Eigen::Tensor<float, 1> output_tensor = nodes_.at(link_map.second.getSourceNodeName()).getOutput().chip(0, 1);  // first time-step
          // auto derivative_tensor = - error_tensor * output_tensor; // derivative of the weight wrt the error
          // Eigen::Tensor<float, 0> derivative_mean_tensor = derivative_tensor.mean(); // average derivative
          Eigen::Tensor<float, 0> derivative_mean_tensor = (- error_tensor * output_tensor).mean(); // average derivative
          // std::cout<<"derivative_mean_tensor "<<derivative_mean_tensor(0)<<std::endl;
          error_sum += derivative_mean_tensor(0);
        } 
        weight_derivatives.at(link_map.second.getWeightName()).push_back(error_sum); 
      }    
    }

    // calculate the average of all error averages 
    // and update the weights
    for (const auto& weight_derivative : weight_derivatives)
    {
      float derivative_sum = 0.0;
      for (const float& derivative : weight_derivative.second)
      {
        derivative_sum += derivative / weight_derivative.second.size();
      }
      weights_.at(weight_derivative.first).updateWeight(derivative_sum);
    }
  }

  void Model::reInitializeNodeStatuses()
  {
    for (auto& node_map : nodes_)
    {
      node_map.second.setStatus(NodeStatus::initialized);
    }
  }

  bool Model::checkNodeNames(const std::vector<std::string> node_names)
  {
    bool nodes_found = true;
    for (const std::string& node_name: node_names)
    {
      if (nodes_.empty() || nodes_.count(node_name) == 0)
      {
        nodes_found = false;
        std::cout << "Node name " << node_name << " not found!" << std::endl;
      }
    }
    return nodes_found;
  }

  bool Model::checkLinkNames(const std::vector<std::string> link_names)
  {
    bool links_found = true;
    for (const std::string& link_name: link_names)
    {
      if (links_.empty() || links_.count(link_name) == 0)
      {
        links_found = false;
        std::cout << "Link name " << link_name << " not found!" << std::endl;
      }
    }
    return links_found;
  }

  bool Model::checkWeightNames(const std::vector<std::string> weight_names)
  {
    bool weights_found = true;
    for (const std::string& weight_name: weight_names)
    {
      if (weights_.empty() || weights_.count(weight_name) == 0)
      {
        weights_found = false;
        std::cout << "Weight name " << weight_name << " not found!" << std::endl;
      }
    }
    return weights_found;
  }
}