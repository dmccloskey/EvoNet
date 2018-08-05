/**TODO:  Add copyright*/

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/SharedFunctions.h>

#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <thread>
#include <future>
#include <mutex>

static std::mutex calculateNetNodeInput_mutex;
static std::mutex calculateNodeInput_mutex;
static std::mutex calculateNetNodeError_mutex;
static std::mutex calculateNodeError_mutex;
static std::mutex calculateModelError_mutex;
static std::mutex calculateOutputNodeError_mutex;

namespace SmartPeak
{
  Model::Model()
  {        
  }

  Model::Model(const Model& other)
  {
    id_ = other.id_;
    name_ = other.name_;
    addLinks(other.getLinks());
		addNodes(other.getNodes());
		addWeights(other.getWeights());
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

  void Model::setError(const Eigen::Tensor<float, 2>& error)
  {
    error_ = error;
  }
  Eigen::Tensor<float, 2> Model::getError() const
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
      std::shared_ptr<Node> node_ptr;
      node_ptr.reset(new Node(node));
      auto found = nodes_.emplace(node.getName(), node_ptr);
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
      return *nodes_.at(node_name);
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
      nodes.push_back(*node.second);
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
      std::shared_ptr<Weight> weight_ptr;
      weight_ptr.reset(new Weight(weight));
      auto found = weights_.emplace(weight.getName(), weight_ptr);
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
      //return *std::move(weights_.at(weight_name));
			return *weights_.at(weight_name);
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
      weights.push_back(*weight.second);
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
      std::shared_ptr<Link> link_ptr;
      link_ptr.reset(new Link(link));
      auto found = links_.emplace(link.getName(), link_ptr);
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
      return *links_.at(link_name);
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
      links.push_back(*link.second);
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
        if (node.second->getName() == link.second->getSourceNodeName() ||
          node.second->getName() == link.second->getSinkNodeName())
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
        if (weight.second->getName() == link.second->getWeightName())
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
        if (node.second->getName() == link.second->getSourceNodeName())
          source_node_found = true;
        if (node.second->getName() == link.second->getSinkNodeName())
          sink_node_found = true;
        if (source_node_found && sink_node_found)
          break;
      }
      bool weight_found = false;
      // if (weights_.empty()) { weight_found = true; }
      for (const auto& weight : weights_)
      {
        if (weight.second->getName() == link.second->getWeightName())
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
    try
    {
      int cnt = 0;
      while (pruneLinks() || pruneWeights() || pruneNodes())
      {
        if (cnt >= iterations) {break;}
        // std::cout<<"Pruning model iteration: "<<cnt<<std::endl;
        cnt += 1;
      }
    }
    catch (std::exception& e)
    {
      printf("Exception: %s", e.what());
    }
  }

  void Model::initNodes(const int& batch_size, const int& memory_size)
  {
    for (auto& node_map : nodes_)
    {
      node_map.second->initNode(batch_size, memory_size);
    }
  }

	void Model::initError(const int & batch_size, const int & memory_size)
	{
		Eigen::Tensor<float, 2> init_values(batch_size, memory_size);
		init_values.setConstant(0.0f);
		setError(init_values);
	}

  void Model::initWeights()
  {
    for (auto& weight_map : weights_)
    {
      weight_map.second->initWeight();
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
          node_map.second->getOutputMutable()->operator()(j, memory_step) = values(j);
        }
        else if (value_type == "error")
        {
          node_map.second->getErrorMutable()->operator()(j, memory_step) = values(j);
        }
        else if (value_type == "dt")
        {
          node_map.second->getDtMutable()->operator()(j, memory_step) = values(j);
        }
				if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
					node_map.second->setStatus(status_update);
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
    else if (nodes_.at(node_names[0])->getOutput().dimension(0) != values.dimension(0))
    {
      std::cout << "The number of input samples and the node batch size does not match." << std::endl;
      return;
    }
    // assumes the node exists
    else if (nodes_.at(node_names[0])->getOutput().dimension(1) <= memory_step)
    {
      std::cout << "The memory_step is greater than the memory_size." << std::endl;
      return;
    }

    // // infer the memory size from the node output size
    // const int memory_size = nodes_.at(node_names[0])->getOutput().dimension(1);

    // copy over the input values
    for (int i=0; i<node_names.size(); ++i)
    {
      for (int j=0; j<values.dimension(0); ++j)
      {
        if (value_type == "output")
        {
          // SANITY CHECK:
          // std::cout << "i" << i << " j" << j << " values: " << values.data()[i*values.dimension(0) + j] << std::endl;
          // nodes_.at(node_names[i])->getOutputPointer()[j + values.dimension(0)*memory_step] = std::move(values.data()[i*values.dimension(0) + j]);
          // nodes_.at(node_names[i])->getOutputPointer()[j + values.dimension(0)*memory_step] = values(j, i);
          nodes_.at(node_names[i])->getOutputMutable()->operator()(j, memory_step) = values(j, i);
        }
        else if (value_type == "error")
        {
          nodes_.at(node_names[i])->getErrorMutable()->operator()(j, memory_step) = values(j, i);
        }
        else if (value_type == "dt")
        {
          nodes_.at(node_names[i])->getDtMutable()->operator()(j, memory_step) = values(j, i);
        }
				if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
					nodes_.at(node_names[i])->setStatus(status_update);
      }
    }
  }
  
  void Model::mapValuesToNode(
    const Eigen::Tensor<float, 1>& values,
    const int& memory_step,
    const std::string& node_name,
    const NodeStatus& status_update,
    const std::string& value_type)
  {
    // check dimension mismatches
    // assumes the node exists
    if (nodes_.at(node_name)->getOutput().dimension(0) != values.dimension(0))
    {
      std::cout << "The number of input samples and the node batch size does not match." << std::endl;
      return;
    }
    
    // // copy over the input values
    // for (int j=0; j<values.dimension(0); ++j)
    // {
    //   if (value_type == "output")
    //   {
    //     nodes_.at(node_name)->getOutputMutable()->operator()(j, memory_step) = values(j);
    //   }
    //   else if (value_type == "error")
    //   {
    //     nodes_.at(node_name)->getErrorMutable()->operator()(j, memory_step) = values(j);
    //   }
    //   else if (value_type == "derivative")
    //   {
    //     nodes_.at(node_name)->getDerivativeMutable()->operator()(j, memory_step) = values(j);
    //   }
    //   else if (value_type == "dt")
    //   {
    //     nodes_.at(node_name)->getDtMutable()->operator()(j, memory_step) = values(j);
    //   }
    // }

    // copy over the input values
    if (value_type == "output")
    {
      nodes_.at(node_name)->getOutputMutable()->chip(memory_step, 1) = values;
    }
    else if (value_type == "error")
    {
      nodes_.at(node_name)->getErrorMutable()->chip(memory_step, 1) = values;
    }
    else if (value_type == "derivative")
    {
      nodes_.at(node_name)->getDerivativeMutable()->chip(memory_step, 1) = values;
    }
    else if (value_type == "dt")
    {
      nodes_.at(node_name)->getDtMutable()->chip(memory_step, 1) = values;
    }

    // update the status
		if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
			nodes_.at(node_name)->setStatus(status_update);
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
    else if (nodes_.at(node_names[0])->getOutput().dimension(0) != values.dimension(0))
    {
      printf("The number of input samples %d and the node batch size %d does not match.\n", (int)values.dimension(0), (int)nodes_.at(node_names[0])->getOutput().dimension(0));
      return;
    }
    else if (nodes_.at(node_names[0])->getOutput().dimension(1) != values.dimension(1))
    {
      printf("The number of input time steps %d and the node memory size %d does not match.\n", (int)values.dimension(1), (int)nodes_.at(node_names[0])->getOutput().dimension(1));
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
            // nodes_.at(node_names[i])->getOutputPointer()[k*values.dimension(0) + j] = values(j, k, i);
            nodes_.at(node_names[i])->getOutputMutable()->operator()(j, k) = values(j, k, i);
          }
          else if (value_type == "error")
          {
            nodes_.at(node_names[i])->getErrorMutable()->operator()(j, k) = values(j, k, i);
          }
					else if (value_type == "derivative")
					{
						nodes_.at(node_names[i])->getDerivativeMutable()->operator()(j, k) = values(j, k, i);
					}
          else if (value_type == "dt")
          {
            nodes_.at(node_names[i])->getDtMutable()->operator()(j, k) = values(j, k, i);
          }
					if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
						nodes_.at(node_names[i])->setStatus(status_update);
        }
      }
    }
  }
  
  void Model::getNextInactiveLayer(
      std::map<std::string, int>& FP_operations_map,
      std::vector<OperationList>& FP_operations)
  {

    // get all links where the source node is active and the sink node is inactive
    // except for biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias &&
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated && 
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized)
      {
        OperationArguments arguments;
				//std::cout<<"Link source node name: "<< link_map.second->getSourceNodeName() <<std::endl
        arguments.source_node = nodes_.at(link_map.second->getSourceNodeName());
				//std::cout << "Link weight name: " << link_map.second->getWeightName() << std::endl;
        arguments.weight = weights_.at(link_map.second->getWeightName());
        arguments.time_step = 0;

        // std::cout<<"Addres of model source node: "<<&nodes_.at(link_map.second->getSourceNodeName())<<std::endl;
        // std::cout<<"Addres of arguments source node: "<<arguments.source_node<<std::endl;
        
        auto found = FP_operations_map.emplace(link_map.second->getSinkNodeName(), (int)FP_operations.size());
        if (!found.second)
        {
          FP_operations[FP_operations_map.at(link_map.second->getSinkNodeName())].arguments.push_back(arguments);
        }
        else
        {
          OperationList operation_list;
          OperationResult result;
          result.sink_node = nodes_.at(link_map.second->getSinkNodeName());
          operation_list.result = result;
          operation_list.arguments.push_back(arguments);
          FP_operations.push_back(operation_list);
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
      if (nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias &&
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated && 
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized)
      {
        std::vector<std::string> links = {link_map.second->getName()};
        auto found = sink_links_map.emplace(link_map.second->getSinkNodeName(), links);        
        if (!found.second)
        {
          sink_links_map[link_map.second->getSinkNodeName()].push_back(link_map.second->getName());
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
      if (nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias &&
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated && 
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized)
      {
        // std::cout << "Model::getNextInactiveLayer() link_name: " << link_map.second->getName() << std::endl;
        // std::cout << "Model::getNextInactiveLayer() source_node_name: " << link_map.second->getSourceNodeName() << std::endl;
        // std::cout << "Model::getNextInactiveLayer() sink_node_name: " << link_map.second->getSinkNodeName() << std::endl;
        links.push_back(link_map.second->getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSourceNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second->getSourceNodeName());
        }
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second->getSinkNodeName()) == 0)
        {
          sink_nodes.push_back(link_map.second->getSinkNodeName());
        }
      }
    }
  }
  
  void Model::getNextInactiveLayerBiases(
    std::map<std::string, int>& FP_operations_map,
    std::vector<OperationList>& FP_operations,
    std::vector<std::string>& sink_nodes_with_biases)
  {

    // get all the biases for the sink nodes
    for (auto& link_map : links_)
    {
      if (        
        // does not allow for cycles
        nodes_.at(link_map.second->getSourceNodeName())->getType() == NodeType::bias && 
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
        FP_operations_map.count(link_map.second->getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        OperationArguments arguments;
        arguments.source_node = nodes_.at(link_map.second->getSourceNodeName());
        arguments.weight = weights_.at(link_map.second->getWeightName());
        arguments.time_step = 0;
        FP_operations[FP_operations_map.at(link_map.second->getSinkNodeName())].arguments.push_back(arguments);
        if (std::count(sink_nodes_with_biases.begin(), sink_nodes_with_biases.end(), link_map.second->getSinkNodeName()) == 0)
        {
          sink_nodes_with_biases.push_back(link_map.second->getSinkNodeName());
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
        nodes_.at(link_map.second->getSourceNodeName())->getType() == NodeType::bias && 
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
        sink_links_map.count(link_map.second->getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        sink_links_map[link_map.second->getSinkNodeName()].push_back(link_map.second->getName());
        if (std::count(sink_nodes_with_biases.begin(), sink_nodes_with_biases.end(), link_map.second->getSinkNodeName()) == 0)
        {
          sink_nodes_with_biases.push_back(link_map.second->getSinkNodeName());
        }
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
        nodes_.at(link_map.second->getSourceNodeName())->getType() == NodeType::bias && 
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
        std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second->getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        // std::cout << "Model::getNextInactiveLayerBiases() link_name: " << link_map.second->getName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerBiases() source_node_name: " << link_map.second->getSourceNodeName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerBiases() sink_node_name: " << link_map.second->getSinkNodeName() << std::endl;
        links.push_back(link_map.second->getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSourceNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second->getSourceNodeName());
        }
        if (std::count(sink_nodes_with_biases.begin(), sink_nodes_with_biases.end(), link_map.second->getSinkNodeName()) == 0)
        {
          sink_nodes_with_biases.push_back(link_map.second->getSinkNodeName());
        }
      }
    }
  }
  
  void Model::getNextInactiveLayerCycles(
    std::map<std::string, int>& FP_operations_map,
    std::vector<OperationList>& FP_operations,
    std::vector<std::string>& sink_nodes_with_cycles)
  {

    // get cyclic source nodes
    for (auto& link_map : links_)
    {
      if (
        (nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::initialized) &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
        FP_operations_map.count(link_map.second->getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        OperationArguments arguments;
        arguments.source_node = nodes_.at(link_map.second->getSourceNodeName());
        arguments.weight = weights_.at(link_map.second->getWeightName());

        // [PARRALLEL] can we check that we will not over exceed the memory
        //             and take appropriate measures here?
        // e.g.
        // memory_size = arguments.source_node->getOutput().dimension(1);
        // if (time_step + 1 >= memory_size) ...
        arguments.time_step = 1;
        FP_operations[FP_operations_map.at(link_map.second->getSinkNodeName())].arguments.push_back(arguments);
        sink_nodes_with_cycles.push_back(link_map.second->getSinkNodeName());
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
        (nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::initialized) &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
        sink_links_map.count(link_map.second->getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        sink_links_map[link_map.second->getSinkNodeName()].push_back(link_map.second->getName());
        sink_nodes_with_cycles.push_back(link_map.second->getSinkNodeName());
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
        (nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::initialized) &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
        std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second->getSinkNodeName()) != 0 // sink node has already been identified
      )
      {
        // std::cout << "Model::getNextInactiveLayerCycles() link_name: " << link_map.second->getName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerCycles() source_node_name: " << link_map.second->getSourceNodeName() << std::endl;
        // std::cout << "Model::getNextInactiveLayerCycles() sink_node_name: " << link_map.second->getSinkNodeName() << std::endl;
        links.push_back(link_map.second->getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSourceNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second->getSourceNodeName());
        }
        if (std::count(sink_nodes_with_cycles.begin(), sink_nodes_with_cycles.end(), link_map.second->getSinkNodeName()) == 0)
        {
          sink_nodes_with_cycles.push_back(link_map.second->getSinkNodeName());
        }
      }
    }
  }

  Eigen::Tensor<float, 1> Model::calculateNodeInput_(
    OperationArguments* arguments, 
    const int& batch_size,
    const int& memory_size,
    const int& time_step)
  {
    std::lock_guard<std::mutex> lock(calculateNodeInput_mutex);

    Eigen::Tensor<float, 1> sink_tensor(batch_size);
    sink_tensor.setConstant(0.0f);
		Eigen::Tensor<float, 1> weight_tensor(batch_size);
		weight_tensor.setConstant(arguments->weight->getWeight());
		if (arguments->time_step == 0 || time_step + arguments->time_step < memory_size)
		{
		  sink_tensor = weight_tensor * arguments->source_node->getOutput().chip(time_step + arguments->time_step, 1);
		}
		else
		{
		  //std::cout<<"time_step exceeded memory size in forwardPropogateLayerNetInput."<<std::endl;
		}
    return sink_tensor;
  }
  
  bool Model::calculateNetNodeInput_(
    OperationList* operations,  
    const int& batch_size,
    const int& memory_size,
    const int& time_step,
    int n_threads)
  {
    std::lock_guard<std::mutex> lock(calculateNetNodeInput_mutex);

    std::vector<std::future<Eigen::Tensor<float, 1>>> task_results;
    int thread_cnt = 0;
    
    Eigen::Tensor<float, 1> sink_tensor(batch_size);
		if (operations->result.sink_node->getIntegration() == NodeIntegration::Sum)
			sink_tensor.setConstant(0.0f);
		else if (operations->result.sink_node->getIntegration() == NodeIntegration::Product)
			sink_tensor.setConstant(1.0f);
		else if (operations->result.sink_node->getIntegration() == NodeIntegration::Max)
			sink_tensor.setConstant(0.0f);    

    // for (const std::string& link : sink_links)
    for (int i=0; i<operations->arguments.size(); ++i)
    {
      std::packaged_task<Eigen::Tensor<float, 1> // encapsulate in a packaged_task
        (OperationArguments*, int, int, int
        )> task(Model::calculateNodeInput_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &operations->arguments[i], std::ref(batch_size), std::ref(memory_size), std::ref(time_step));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || i == operations->arguments.size() - 1)
      {
        for (auto& task_result: task_results)
        {
          if (task_result.valid())
          {
            try
            {
			  // [TESTS: add tests for Sum, Product, or Max NodeIntegration]
			  if (operations->result.sink_node->getIntegration() == NodeIntegration::Sum)
				  sink_tensor += task_result.get(); 
			  else if (operations->result.sink_node->getIntegration() == NodeIntegration::Product)
				  sink_tensor *= task_result.get();
			  else if (operations->result.sink_node->getIntegration() == NodeIntegration::Max)
				  sink_tensor = sink_tensor.cwiseMax(task_result.get());
            }            
            catch (std::exception& e)
            {
              printf("Exception: %s", e.what());
            }
          }
        }
        task_results.clear();
        thread_cnt = 0;
      }
      else
      {
        ++thread_cnt;
      } 
    }

    // calculate the output and the derivative
    const NodeType sink_node_type = operations->result.sink_node->getType();
    const NodeActivation sink_node_activation = operations->result.sink_node->getActivation();
    Eigen::Tensor<float, 1> output = calculateActivation(
      sink_node_type, sink_node_activation, sink_tensor,
      operations->result.sink_node->getDt().chip(time_step, 1),
      1);
    Eigen::Tensor<float, 1> derivative = calculateDerivative(
      sink_node_type, sink_node_activation, output, 1);
    
    operations->result.sink_node->setStatus(NodeStatus::activated);
		operations->result.sink_node->getInputMutable()->chip(time_step, 1) = sink_tensor; // [TESTS: update tests]
    operations->result.sink_node->getOutputMutable()->chip(time_step, 1) = output;
    operations->result.sink_node->getDerivativeMutable()->chip(time_step, 1) = derivative;

    return true;
  }

  void Model::forwardPropogateLayerNetInput(
      std::vector<OperationList>& FP_operations,
    const int& time_step, int n_threads)
  {

    // get all the information needed to construct the tensors
    int batch_size = 0;
    int memory_size = 0;
    for (const auto& FP_operation : FP_operations)
    {
      batch_size = FP_operation.result.sink_node->getOutput().dimension(0);
      memory_size = FP_operation.result.sink_node->getOutput().dimension(1);
      break;
    }

    // iterate through each sink node and calculate the net input
    // invoke the activation function once the net input is calculated
    std::vector<std::future<bool>> task_results;
    int thread_cnt = 0;
    const int threads_per_sub_process = 1; // [TODO: how to best divide up the allowable threads?]
    int operations_cnt = 0;
    for (auto& FP_operation : FP_operations)
    {
      std::packaged_task<bool // encapsulate in a packaged_task
        (OperationList*, int, int, int, int
        )> task(Model::calculateNetNodeInput_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &FP_operation, std::ref(batch_size), std::ref(memory_size), std::ref(time_step),
        std::ref(threads_per_sub_process));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || operations_cnt == FP_operations.size() - 1)
      {
        for (auto& task_result: task_results)
        {
          if (task_result.valid())
          {
            try
            {
              bool success = task_result.get();
              // Eigen::Tensor<float, 1> model_output(batch_size);
              // model_output = nodes_.at(FP_operation.result.sink_node->getName()).getOutput().chip(time_step, 1);
              // Eigen::Tensor<float, 1> result_output(batch_size);
              // result_output = FP_operation.result.sink_node->getOutput().chip(time_step, 1);
              // std::cout<<"Model output: "<<model_output<<std::endl;
              // std::cout<<"FP operation result: "<<result_output<<std::endl;
            }
            catch (std::exception& e)
            {
              printf("Exception: %s", e.what());
            }
          }
        }
        task_results.clear();
        thread_cnt = 0;
      }
      else
      {
        thread_cnt += threads_per_sub_process;
      } 
      // std::cout<<"thread_count"<<thread_cnt<<std::endl;
      // std::cout<<"operations_cnt"<<operations_cnt<<std::endl;
      ++operations_cnt;
    }
  }

	// [DEPRECATED]
  void Model::forwardPropogateLayerNetInput(
    std::map<std::string, std::vector<std::string>>& sink_links_map,
    const int& time_step, int n_threads)
  {

    // get all the information needed to construct the tensors
    int batch_size = 0;
    int memory_size = 0;
    for (const auto& sink_links : sink_links_map)
    {
      batch_size = nodes_.at(sink_links.first)->getOutput().dimension(0);
      memory_size = nodes_.at(sink_links.first)->getOutput().dimension(1);
      break;
    }

    // iterate through each sink node and calculate the net input
    // invoke the activation function once the net input is calculated
    for (const auto& sink_links : sink_links_map)
    {
      Eigen::Tensor<float, 1> sink_tensor(batch_size);
      sink_tensor.setConstant(0.0f);
      Eigen::Tensor<float, 1> weight_tensor(batch_size);
      for (const std::string& link : sink_links.second)
      {
        weight_tensor.setConstant(weights_.at(links_.at(link)->getWeightName())->getWeight());
        if (nodes_.at(links_.at(link)->getSourceNodeName())->getStatus() == NodeStatus::activated)
        {
          sink_tensor = sink_tensor + weight_tensor * nodes_.at(links_.at(link)->getSourceNodeName())->getOutput().chip(time_step, 1); //current time-step
        }
        else if (nodes_.at(links_.at(link)->getSourceNodeName())->getStatus() == NodeStatus::initialized)
        {
          if (time_step + 1 < memory_size)
          {
            sink_tensor = sink_tensor + weight_tensor * nodes_.at(links_.at(link)->getSourceNodeName())->getOutput().chip(time_step + 1, 1); //previous time-step
          }
          else
          {
            std::cout<<"time_step exceeded memory size in forwardPropogateLayerNetInput."<<std::endl;
          }
        }
      }

      // calculate the output and the derivative
      const NodeType sink_node_type = nodes_.at(sink_links.first)->getType();
      const NodeActivation sink_node_activation = nodes_.at(sink_links.first)->getActivation();
      Eigen::Tensor<float, 1> output = calculateActivation(
        sink_node_type, sink_node_activation, sink_tensor,
        nodes_.at(sink_links.first)->getDt().chip(time_step, 1),
        1);
      Eigen::Tensor<float, 1> derivative = calculateDerivative(
        sink_node_type, sink_node_activation, output, 1);

      // update the node
      mapValuesToNode(output, time_step, sink_links.first, NodeStatus::activated, "output");
      mapValuesToNode(derivative, time_step, sink_links.first, NodeStatus::activated, "derivative");      
    }
  }

  // [DEPRECATED]
  void Model::forwardPropogateLayerNetInput(
    const std::vector<std::string>& links,
    const std::vector<std::string>& source_nodes,
    const std::vector<std::string>& sink_nodes,
    const int& time_step)
  {
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(source_nodes[0])->getOutput().dimension(0);
    const int memory_size = nodes_.at(source_nodes[0])->getOutput().dimension(1);

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
        if (nodes_.at(source_nodes[i])->getStatus() == NodeStatus::activated)
        {
          source_tensor(j, i) = nodes_.at(source_nodes[i])->getOutput()(j, time_step); //current time-step
        }
        else if (nodes_.at(source_nodes[i])->getStatus() == NodeStatus::initialized)
        {
          if (time_step + 1 < memory_size)
          {
            source_tensor(j, i) = nodes_.at(source_nodes[i])->getOutput()(j, time_step + 1); //previous time-step
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
          if (links_.at(link)->getSinkNodeName() == sink_nodes[i] &&
          links_.at(link)->getSourceNodeName() == source_nodes[j])
          {
            weight_tensor(j, i) = weights_.at(links_.at(link)->getWeightName())->getWeight();
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
  
  void Model::forwardPropogate(const int& time_step, bool cache_FP_steps, bool use_cache, int n_threads)
  { 
    if (use_cache)
    {
      for (auto& FP_operations : FP_operations_cache_)
        forwardPropogateLayerNetInput(FP_operations, time_step, n_threads);
    }
    else
    {
      const int max_iters = 1e6;
      for (int iter=0; iter<max_iters; ++iter)
      { 
        // get the next hidden layer
        std::map<std::string, int> FP_operations_map;
        std::vector<OperationList> FP_operations_list;
        getNextInactiveLayer(FP_operations_map, FP_operations_list);

        // get biases,
        std::vector<std::string> sink_nodes_with_biases;
        getNextInactiveLayerBiases(FP_operations_map, FP_operations_list, sink_nodes_with_biases);
        
        // get cycles
        std::vector<std::string> sink_nodes_cycles;
        getNextInactiveLayerCycles(FP_operations_map, FP_operations_list, sink_nodes_cycles);

        // std::cout<<"sink nodes cycles size "<<sink_nodes_cycles.size()<<std::endl;
        // std::cout<<"FP operations list size "<<FP_operations_list.size()<<std::endl;
        if (sink_nodes_cycles.size() > 0 && 
          sink_nodes_cycles.size() != FP_operations_list.size())
        { // not all forward propogation steps have caught up
          // need to remove sink nodes with cycles
          std::vector<OperationList> FP_operations_list_nocycles;
          for (const std::string& sink_node: sink_nodes_cycles)
          {
            FP_operations_list_nocycles.push_back(FP_operations_list[FP_operations_map.at(sink_node)]);
          }
          FP_operations_list = FP_operations_list_nocycles;
        }

        // check if all nodes have been activated
        if (FP_operations_list.size() == 0)
        {
          break;
        }

        if (cache_FP_steps)
          FP_operations_cache_.push_back(FP_operations_list);

        // calculate the net input
        forwardPropogateLayerNetInput(FP_operations_list, time_step, n_threads);
      }
    }
  }
  
  // [DEPRECATED]
  // void Model::forwardPropogate(const int& time_step, bool cache_FP_steps, bool use_cache, int n_threads)
  // { 
  //   if (use_cache)
  //   {
  //     for (auto& sink_link : FP_sink_link_cache_)
  //       forwardPropogateLayerNetInput(sink_link, time_step, n_threads);
  //   }
  //   else
  //   {
  //     const int max_iters = 1e6;
  //     for (int iter=0; iter<max_iters; ++iter)
  //     { 
  //       // get the next hidden layer
  //       std::map<std::string, std::vector<std::string>> sink_links_map;
  //       getNextInactiveLayer(sink_links_map);

  //       // get biases,
  //       std::vector<std::string> sink_nodes_with_biases;
  //       getNextInactiveLayerBiases(sink_links_map, sink_nodes_with_biases);
        
  //       // get cycles
  //       std::map<std::string, std::vector<std::string>> sink_links_map_cycles = sink_links_map;
  //       std::vector<std::string> sink_nodes_cycles;
  //       getNextInactiveLayerCycles(sink_links_map_cycles, sink_nodes_cycles);

  //       if (sink_links_map_cycles.size() == sink_links_map.size())
  //       { // all forward propogation steps have caught up
  //         // add sink nodes with cycles to the forward propogation step
  //         sink_links_map = sink_links_map_cycles;
  //       }

  //       // check if all nodes have been activated
  //       if (sink_links_map.size() == 0)
  //       {
  //         break;
  //       }

  //       if (cache_FP_steps)
  //         FP_sink_link_cache_.push_back(sink_links_map);

  //       // calculate the net input
  //       forwardPropogateLayerNetInput(sink_links_map, time_step, n_threads);
  //     }
  //   }
  // }
  
  // [DEPRECATED]
  // void Model::forwardPropogate(const int& time_step)
  // {
  //   const int max_iters = 1e6;
  //   for (int iter=0; iter<max_iters; ++iter)
  //   {      
  //     // std::cout<<"Model::forwardPropogate() iter: "<<iter<<std::endl;

  //     // get the next hidden layer
  //     std::vector<std::string> links, source_nodes, sink_nodes;
  //     getNextInactiveLayer(links, source_nodes, sink_nodes);
  //     // std::cout<<"Model::forwardPropogate() getNextInactiveLayer links, source, and sink sizes "<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

  //     // get biases,
  //     std::vector<std::string> sink_nodes_with_biases;
  //     getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);
  //     // std::cout<<"Model::forwardPropogate() getNextInactiveLayerBiases links, source, and sink sizes "<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;
      
  //     // get cycles
  //     std::vector<std::string> links_cycles, source_nodes_cycles, sink_nodes_cycles;
  //     getNextInactiveLayerCycles(links_cycles, source_nodes_cycles, sink_nodes, sink_nodes_cycles);
  //     // std::cout<<"Model::forwardPropogate() getNextInactiveLayerCycles links, source, and sink sizes "<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() sink_nodes_cycles: "<<sink_nodes_cycles.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() links: "<<links.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

  //     if (sink_nodes_cycles.size() == sink_nodes.size())
  //     { // all forward propogation steps have caught up
  //       // add sink nodes with cycles to the forward propogation step
  //       links.insert( links.end(), links_cycles.begin(), links_cycles.end() );
  //       source_nodes.insert( source_nodes.end(), source_nodes_cycles.begin(), source_nodes_cycles.end() );
  //     }
  //     else
  //     { // remove source/sink nodes with cycles from the forward propogation step
  //       for (const std::string node_name : sink_nodes_cycles)
  //       {
  //         sink_nodes.erase(std::remove(sink_nodes.begin(), sink_nodes.end(), node_name), sink_nodes.end());
  //       }
  //     }

  //     // check if all nodes have been activated
  //     if (links.size() == 0)
  //     {
  //       break;
  //     }      
  //     // std::cout<<"Model::forwardPropogate() final links, source, and sink sizes "<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
  //     // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

  //     // calculate the net input
  //     forwardPropogateLayerNetInput(links, source_nodes, sink_nodes, time_step);

  //     // calculate the activation
  //     forwardPropogateLayerActivation(sink_nodes, time_step);
  //   }
  // }

  void Model::FPTT(const int& time_steps, 
    const Eigen::Tensor<float, 3>& values,
    const std::vector<std::string> node_names,
    const Eigen::Tensor<float, 2>& dt,
    bool cache_FP_steps, bool use_cache, int n_threads)
  {
    // check time_steps vs memory_size
    int max_steps = time_steps;
    if (time_steps > nodes_.begin()->second->getOutput().dimension(1))
    {
      std::cout<<"Time_steps will be scaled back to the memory_size."<<std::endl;
      max_steps = nodes_.begin()->second->getOutput().dimension(1);
    }

    for (int time_step=0; time_step<max_steps; ++time_step)
    {
			const int time_step_cur = max_steps - 1 - time_step;

      // std::cout<<"Model::FPTT() time_step: "<<time_step<<std::endl;
      if (time_step>0)
      {
        // move to the next memory step
        for (auto& node_map: nodes_)
        {      
          if (std::count(node_names.begin(), node_names.end(), node_map.first) == 0)
          {
            node_map.second->setStatus(NodeStatus::initialized); // reinitialize non-input nodes
          }   
          // std::cout<<"Model::FPTT() output: "<<node_map.second->getOutput()<<" for node_name: "<<node_map.first<<std::endl;
        }
      }

      // initialize nodes for the next time-step
      const Eigen::Tensor<float, 1> dt_values = dt.chip(time_step, 1);
      mapValuesToNodes(dt_values, time_step_cur, NodeStatus::deactivated, "dt"); // [TESTS: setting this to "initialized" caused one hell of a headache to debug...]
      const Eigen::Tensor<float, 2> active_values = values.chip(time_step, 1);
       //std::cout<<"Model::FPTT() active_values: "<<active_values<<std::endl;
      mapValuesToNodes(active_values, time_step_cur, node_names, NodeStatus::activated, "output");

      if (cache_FP_steps && time_step == 0)
        forwardPropogate(time_step_cur, true, false, n_threads);
      else if (cache_FP_steps && time_step > 0)
        forwardPropogate(time_step_cur, false, true, n_threads);
      else
        forwardPropogate(time_step_cur, cache_FP_steps, use_cache, n_threads); // always working at the current head of memory
    }
  }

	Eigen::Tensor<float, 1> Model::calculateModelError_(
		Node* output_node,
		const Eigen::Tensor<float, 1>& expected,
		LossFunctionOp<float>* loss_function,
		const int& batch_size,
		const int& time_step
	){
		std::lock_guard<std::mutex> lock(calculateModelError_mutex);

		Eigen::Tensor<float, 1> model_error(batch_size);
		model_error = loss_function->operator()(output_node->getOutput().chip(time_step, 1), expected);
		return model_error;
	};

	bool Model::calculateOutputNodeError_(
		Node* output_node,
		const Eigen::Tensor<float, 1>& expected,
		LossFunctionGradOp<float>* loss_function_grad,
		const int& time_step
	){
		std::lock_guard<std::mutex> lock(calculateOutputNodeError_mutex);

		// [BUG previous incorrect implementation below]
		//output_node->getErrorMutable()->chip(time_step, 1) = loss_function_grad->operator()(
		//	output_node->getOutput().chip(time_step, 1), expected);
		// [CORRECT implementation below]
		output_node->getErrorMutable()->chip(time_step, 1) = loss_function_grad->operator()(
			output_node->getOutput().chip(time_step, 1), expected) *
			output_node->getDerivative().chip(time_step, 1);
		//std::cout << "expected: " << expected << std::endl;
		//std::cout << "derivative: " << output_node->getDerivative().chip(time_step, 1) << std::endl;
		//std::cout << "output: " << output_node->getOutput().chip(time_step, 1) << std::endl;
		//std::cout << "error: " << output_node->getError().chip(time_step, 1) << std::endl;
		output_node->setStatus(NodeStatus::corrected);
		return true;
	};

	void Model::calculateError(
		const Eigen::Tensor<float, 2>& values, const std::vector<std::string>& node_names,
		const int& time_step, bool cache_output_nodes, bool use_cache,
		int n_threads)
	{
		//TODO: encapsulate into a seperate method
		// infer the batch size from the first source node
		const int batch_size = nodes_.at(node_names[0])->getOutput().dimension(0);

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

		// get the model loss function and loss function gradients
		std::shared_ptr<LossFunctionOp<float>> operation_ptr;
		std::shared_ptr<LossFunctionGradOp<float>> gradient_ptr;
		switch (loss_function_)
		{
			case ModelLossFunction::EuclideanDistance:
			{
				operation_ptr.reset(new EuclideanDistanceOp<float>);
				gradient_ptr.reset(new EuclideanDistanceGradOp<float>);
				break;
			}
			case ModelLossFunction::L2Norm:
			{
				operation_ptr.reset(new L2NormOp<float>);
				gradient_ptr.reset(new L2NormGradOp<float>);
				break;
			}
			case ModelLossFunction::CrossEntropy:
			{
				operation_ptr.reset(new CrossEntropyOp<float>);
				gradient_ptr.reset(new CrossEntropyGradOp<float>);
				break;
			}
			case ModelLossFunction::NegativeLogLikelihood:
			{
				operation_ptr.reset(new NegativeLogLikelihoodOp<float>);
				gradient_ptr.reset(new NegativeLogLikelihoodGradOp<float>);
				break;
			}
			case ModelLossFunction::MSE:
			{
				operation_ptr.reset(new MSEOp<float>);
				gradient_ptr.reset(new MSEGradOp<float>);
				break;
			}
			default:
			{
				std::cout << "Loss Function not supported." << std::endl;
				break;
			}
		}

		// collect the output nodes
		std::vector<std::shared_ptr<Node>> output_nodes;
		if (use_cache)
		{ 
			output_nodes = output_node_cache_;
		}
		else
		{
			for (int i = 0; i < node_names.size(); ++i)
			{
				std::shared_ptr<Node> output_node = nodes_.at(node_names[i]);
				if (cache_output_nodes)
					output_node_cache_.push_back(output_node);
				output_nodes.push_back(output_node);
			}
		}
		
		// loop over all nodes and calculate the error for the model
		std::vector<std::future<Eigen::Tensor<float, 1>>> model_error_task_results;
		Eigen::Tensor<float, 1> model_error(batch_size);
		model_error.setConstant(0.0f);
		int thread_cnt = 0;
		for (int i = 0; i<node_names.size(); ++i)
		{
			// encapsulate in a packaged_task
			std::packaged_task<Eigen::Tensor<float, 1> 
				(Node*, Eigen::Tensor<float, 1>, LossFunctionOp<float>*,
					int, int
					)> task(Model::calculateModelError_);

			// launch the thread
			model_error_task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				output_nodes[i].get(), values.chip(i, 1), operation_ptr.get(), std::ref(batch_size), std::ref(time_step));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || i == node_names.size() - 1)
			{
				for (auto& task_result : model_error_task_results)
				{
					if (task_result.valid())
					{
						try
						{
							model_error += task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				model_error_task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				++thread_cnt;
			}
		}
		error_.chip(time_step, 1) = model_error; // asign the model_error

		// loop over all nodes and calculate the error for the nodes
		std::vector<std::future<bool>> output_node_error_task_results;
		thread_cnt = 0;
		for (int i = 0; i<node_names.size(); ++i)
		{
			// encapsulate in a packaged_task
			std::packaged_task<bool
			(Node*, Eigen::Tensor<float, 1>, LossFunctionGradOp<float>*,
				int
				)> task(Model::calculateOutputNodeError_);

			// launch the thread
			output_node_error_task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				output_nodes[i].get(), values.chip(i, 1), gradient_ptr.get(), std::ref(time_step));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || i == node_names.size() - 1)
			{
				for (auto& task_result : output_node_error_task_results)
				{
					if (task_result.valid())
					{
						try
						{
							bool result = task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				output_node_error_task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				++thread_cnt;
			}
		}
	}
 

  void Model::CETT(const Eigen::Tensor<float, 3>& values, const std::vector<std::string>& node_names, const int & time_steps,
	bool cache_output_nodes, bool use_cache, int n_threads)
  {
	// check time_steps vs memory_size
	int max_steps = time_steps;
	if (time_steps > nodes_.begin()->second->getOutput().dimension(1))
	{
	  std::cout << "Time_steps will be scaled back to the memory_size." << std::endl;
	  max_steps = nodes_.begin()->second->getOutput().dimension(1);
	}

	// NOTE: the output are stored [Tmax, Tmax - 1, ..., T=0]
	//	     while the expected output (values) are stored [T=0, T=1, ..., Tmax]
	for (int i=0; i<max_steps; ++i)
	{
	  int next_time_step = values.dimension(1) - 1 - i;
	  // [TESTS: Test for the expected output error at each time step]
	  //std::cout<<"Expected output for time point "<< i << " is " << values.chip(next_time_step, 1)<<std::endl;

	  // calculate the error for each batch of memory
	  if (cache_output_nodes && i == 0)
		calculateError(values.chip(next_time_step, 1), node_names, i, true, false, n_threads);
	  else if (cache_output_nodes && i > 0)
		calculateError(values.chip(next_time_step, 1), node_names, i, false, true, n_threads);
	  else
		calculateError(values.chip(next_time_step, 1), node_names, i, cache_output_nodes, use_cache, n_threads);
	  //calculateError(values.chip(i, 1), node_names, i);
	}
  }
  
  void Model::getNextUncorrectedLayer(
    std::map<std::string, int>& BP_operations_map,
    std::vector<OperationList>& BP_operations,
    std::vector<std::string>& source_nodes)
  {
    // get all links where the source node is corrected and the sink node is active
    // including biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected && 
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated)
      {        
        OperationArguments arguments;
        arguments.source_node = nodes_.at(link_map.second->getSinkNodeName());
        arguments.weight = weights_.at(link_map.second->getWeightName());
        arguments.time_step = 0;

        // std::cout<<"Addres of model source node: "<<&nodes_.at(link_map.second->getSourceNodeName())<<std::endl;
        // std::cout<<"Addres of arguments source node: "<<arguments.source_node<<std::endl;
        
        auto found = BP_operations_map.emplace(link_map.second->getSourceNodeName(), (int)BP_operations.size());
        if (!found.second)
        {
          BP_operations[BP_operations_map.at(link_map.second->getSourceNodeName())].arguments.push_back(arguments);
        }
        else
        {
          OperationList operation_list;
          OperationResult result;
          result.sink_node = nodes_.at(link_map.second->getSourceNodeName());
          operation_list.result = result;
          operation_list.arguments.push_back(arguments);
          BP_operations.push_back(operation_list);
        }

        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSinkNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second->getSinkNodeName());
        }
      }
    }
  }
  
  void Model::getNextUncorrectedLayer(
    std::map<std::string, std::vector<std::string>>& sink_links_map,
    std::vector<std::string>& source_nodes)
  {
    // get all links where the source node is corrected and the sink node is active
    // including biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected && 
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated)
      {
        std::vector<std::string> links = {link_map.second->getName()};
        auto found = sink_links_map.emplace(link_map.second->getSourceNodeName(), links);        
        if (!found.second)
        {
          sink_links_map[link_map.second->getSourceNodeName()].push_back(link_map.second->getName());
        }
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSinkNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second->getSinkNodeName());
        }
      }
    }
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
      if (nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected && 
        nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated)
      {
        links.push_back(link_map.second->getName());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSinkNodeName()) == 0)
        {
          source_nodes.push_back(link_map.second->getSinkNodeName());
        }
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second->getSourceNodeName()) == 0)
        {
          sink_nodes.push_back(link_map.second->getSourceNodeName());
        }
      }
    }
  }
  
  void Model::getNextUncorrectedLayerCycles(
    std::map<std::string, int>& BP_operations_map,
    std::vector<OperationList>& BP_operations,
    std::vector<std::string>& source_nodes,
    std::vector<std::string>& source_nodes_with_cycles)
  {

    // allows for cycles
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::corrected &&
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected &&
        std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSinkNodeName()) != 0 // source node has already been identified)
      ) 
      {
        OperationArguments arguments;
        arguments.source_node = nodes_.at(link_map.second->getSinkNodeName());
        arguments.weight = weights_.at(link_map.second->getWeightName());
        arguments.time_step = 0;        
        
        auto found = BP_operations_map.emplace(link_map.second->getSourceNodeName(), (int)BP_operations.size());
        if (!found.second)
        {
          BP_operations[BP_operations_map.at(link_map.second->getSourceNodeName())].arguments.push_back(arguments);
        }
        else
        {
          OperationList operation_list;
          OperationResult result;
          result.sink_node = nodes_.at(link_map.second->getSourceNodeName());
          result.time_step = 1;
          operation_list.result = result;
          operation_list.arguments.push_back(arguments);
          BP_operations.push_back(operation_list);
        }
        
        if (std::count(source_nodes_with_cycles.begin(), source_nodes_with_cycles.end(), link_map.second->getSinkNodeName()) == 0)
        {
          source_nodes_with_cycles.push_back(link_map.second->getSinkNodeName());
        }
      }
    }
  }
  
  void Model::getNextUncorrectedLayerCycles(
    std::map<std::string, std::vector<std::string>>& sink_links_map,
    const std::vector<std::string>& source_nodes,
    std::vector<std::string>& source_nodes_with_cycles)
  {

    // allows for cycles
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::corrected &&
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected &&
        std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSinkNodeName()) != 0 // source node has already been identified)
      ) 
      {
        sink_links_map[link_map.second->getSourceNodeName()].push_back(link_map.second->getName());
        if (std::count(source_nodes_with_cycles.begin(), source_nodes_with_cycles.end(), link_map.second->getSinkNodeName()) == 0)
        {
          source_nodes_with_cycles.push_back(link_map.second->getSinkNodeName());
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
      if (nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::corrected && 
        std::count(links.begin(), links.end(), link_map.second->getName()) == 0 && // unique links 
        nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected &&
        std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSinkNodeName()) != 0 // source node has already been identified)
      ) 
      {
        links.push_back(link_map.second->getName());
        // could use std::set instead to check for duplicates
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second->getSourceNodeName()) == 0)
        {
          sink_nodes.push_back(link_map.second->getSourceNodeName());
        }
        if (std::count(source_nodes_with_cycles.begin(), source_nodes_with_cycles.end(), link_map.second->getSinkNodeName()) == 0)
        {
          source_nodes_with_cycles.push_back(link_map.second->getSinkNodeName());
        }
      }
    }
  }  

  Eigen::Tensor<float, 1> Model::calculateNodeError_(
    OperationArguments* arguments, 
    const int& batch_size,
    const int& memory_size,
    const int& time_step
  )
  {
    std::lock_guard<std::mutex> lock(calculateNodeError_mutex);

    Eigen::Tensor<float, 1> sink_tensor(batch_size);
    Eigen::Tensor<float, 1> weight_tensor(batch_size);
    weight_tensor.setConstant(arguments->weight->getWeight());
		if (arguments->source_node->getIntegration() == NodeIntegration::Sum)
			sink_tensor = weight_tensor * arguments->source_node->getError().chip(time_step, 1);
		else if (arguments->source_node->getIntegration() == NodeIntegration::Product)
			sink_tensor = arguments->source_node->getInput().chip(time_step, 1) * arguments->source_node->getError().chip(time_step, 1); // missing the division by the source node output
		else if (arguments->source_node->getIntegration() == NodeIntegration::Max)
			sink_tensor = weight_tensor * arguments->source_node->getError().chip(time_step, 1); // [TODO: update with correct formulat]
    // std::cout<<"Weight tensor: "<<weight_tensor<<std::endl;
    // std::cout<<"Source tensor: "<<arguments->source_node->getError().chip(time_step, 1)<<std::endl;
    // std::cout<<"Sink tensor: "<<sink_tensor<<std::endl;
    return sink_tensor;
  }

  bool Model::calculateNetNodeError_(
    OperationList* operations, 
    const int& batch_size,
    const int& memory_size,
    const int& time_step,
    int n_threads
  )
  {
    std::lock_guard<std::mutex> lock(calculateNetNodeError_mutex);

    std::vector<std::future<Eigen::Tensor<float, 1>>> task_results;
    int thread_cnt = 0;
    
    Eigen::Tensor<float, 1> sink_tensor(batch_size);
    sink_tensor.setConstant(0.0f);

    // for (const std::string& link : sink_links)
    for (int i=0; i<operations->arguments.size(); ++i)
    {
      std::packaged_task<Eigen::Tensor<float, 1> // encapsulate in a packaged_task
        (OperationArguments*, int, int, int
        )> task(Model::calculateNodeError_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &operations->arguments[i], std::ref(batch_size), std::ref(memory_size), std::ref(time_step));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || i == operations->arguments.size() - 1)
      {
        for (auto& task_result: task_results) 
        {       
          if (task_result.valid())
          {
            try
            {
							if (operations->arguments[i].source_node->getIntegration() == NodeIntegration::Sum)
								sink_tensor += task_result.get();
							else if (operations->arguments[i].source_node->getIntegration() == NodeIntegration::Product)
								sink_tensor += (task_result.get() / operations->result.sink_node->getOutput().chip(time_step, 1)
									).unaryExpr(std::ptr_fun(checkNanInf<float>)); // apply the missing division by the source node output
							else if (operations->arguments[i].source_node->getIntegration() == NodeIntegration::Max)
								sink_tensor += task_result.get(); // [TODO: update with correct formula]
            }
            catch (std::exception& e)
            {
              printf("Exception: %s", e.what());
            }
          }
        }
        task_results.clear();
        thread_cnt = 0;
      }
      else
      {
        ++thread_cnt;
      } 
    }
   
    if (operations->result.time_step == 0 || time_step + operations->result.time_step < memory_size)
    { // [PARALLEL: could add a dummy time step with output 0 so as not to need a check for the memory size being exceeded]

      // scale the error by the derivative and add in any residual error
      sink_tensor = sink_tensor * operations->result.sink_node->getDerivative().chip(time_step + operations->result.time_step, 1) + operations->result.sink_node->getError().chip(time_step + operations->result.time_step, 1);

      // update the node error
      operations->result.sink_node->setStatus(NodeStatus::corrected);
      operations->result.sink_node->getErrorMutable()->chip(time_step + operations->result.time_step, 1) = sink_tensor;
    }
    else
    {
      //std::cout<<"time_step exceeded memory size in backwardPropogateLayerError."<<std::endl;
    }

    return true;
  }

  void Model::backPropogateLayerError(
    std::vector<OperationList>& BP_operations,
    const int& time_step, int n_threads)
  {

    // get all the information needed to construct the tensors
    int batch_size = 0;
    int memory_size = 0;
    for (const auto& BP_operation : BP_operations)
    {
      batch_size = BP_operation.result.sink_node->getOutput().dimension(0);
      memory_size = BP_operation.result.sink_node->getOutput().dimension(1);
      break;
    }

    if (time_step >= memory_size)
    {
      std::cout<<"time step: "<<time_step<<" exceeds the memory_size!"<<std::endl;
      return;
    }

    // iterate through each sink node and calculate the error
    std::vector<std::future<bool>> task_results;
    int thread_cnt = 0;
    const int threads_per_sub_process = 1; // [TODO: how to best divide up the allowable threads?]
    int operations_cnt = 0;
    for (auto& BP_operation : BP_operations)
    {
      std::packaged_task<bool // encapsulate in a packaged_task
        (OperationList*, int, int, int, int
        )> task(Model::calculateNetNodeError_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &BP_operation, std::ref(batch_size), std::ref(memory_size), std::ref(time_step),
        std::ref(threads_per_sub_process));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || operations_cnt == BP_operations.size() - 1)
      {
        for (auto& task_result: task_results)
        {
          if (task_result.valid())
          {
            try
            {
              bool success = task_result.get();
              // Eigen::Tensor<float, 1> model_output(batch_size);
              // model_output = nodes_.at(BP_operation.result.sink_node->getName())->getError().chip(time_step, 1);
              // Eigen::Tensor<float, 1> result_error(batch_size);
              // result_error = BP_operation.result.sink_node->getError().chip(time_step, 1);
              // std::cout<<"Model error: "<<model_output<<std::endl;
              // std::cout<<"BP operation result: "<<result_error<<std::endl;
            }
            catch (std::exception& e)
            {
              printf("Exception: %s", e.what());
            }
          }
        }
        task_results.clear();
        thread_cnt = 0;
      }
      else
      {
        thread_cnt += threads_per_sub_process;
      } 
      // std::cout<<"thread_count"<<thread_cnt<<std::endl;
      // std::cout<<"operations_cnt"<<operations_cnt<<std::endl;
      ++operations_cnt;
    }
  }

  void Model::backPropogateLayerError(
    const std::map<std::string, std::vector<std::string>>& sink_links_map,
    const int& time_step, int n_threads)
  {

    // get all the information needed to construct the tensors
    int batch_size = 0;
    int memory_size = 0;
    for (const auto& sources_links : sink_links_map)
    {
      batch_size = nodes_.at(sources_links.first)->getOutput().dimension(0);
      memory_size = nodes_.at(sources_links.first)->getOutput().dimension(1);
      break;
    }

    if (time_step >= memory_size)
    {
      std::cout<<"time step: "<<time_step<<" exceeds the memory_size!"<<std::endl;
      return;
    }

    // iterate through each sink node and calculate the error
    for (const auto& sinks_links : sink_links_map)
    {
      Eigen::Tensor<float, 1> sink_tensor(batch_size);
      sink_tensor.setConstant(0.0f);
      Eigen::Tensor<float, 1> weight_tensor(batch_size);

      // calculate the total incoming error
      for (const std::string& link : sinks_links.second)
      {
        weight_tensor.setConstant(weights_.at(links_.at(link)->getWeightName())->getWeight());
        sink_tensor = sink_tensor + weight_tensor * nodes_.at(links_.at(link)->getSinkNodeName())->getError().chip(time_step, 1);
      }
      
      // scale the error by the derivative
      if (nodes_.at(sinks_links.first)->getStatus() == NodeStatus::activated)
      {
        sink_tensor = sink_tensor * nodes_.at(sinks_links.first)->getDerivative().chip(time_step, 1); // current time-step
      }
      else if (nodes_.at(sinks_links.first)->getStatus() == NodeStatus::corrected)
      {
        // std::cout << "Model::backPropogateLayerError() Previous derivative (batch_size, Sink) " << j << "," << i << std::endl;
        if (time_step + 1 < memory_size)
        {
          sink_tensor = sink_tensor * nodes_.at(sinks_links.first)->getDerivative().chip(time_step + 1, 1); // previous time-step
        }  
      }  

      // update the errors      
      if (nodes_.at(sinks_links.first)->getStatus() == NodeStatus::activated)
      {
        mapValuesToNode(sink_tensor, time_step, sinks_links.first, NodeStatus::corrected, "error");
      }
      else if (nodes_.at(sinks_links.first)->getStatus() == NodeStatus::corrected)
      {
        if (time_step + 1 < memory_size)
        {
          mapValuesToNode(sink_tensor, time_step + 1, sinks_links.first, NodeStatus::corrected, "error");
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
    const int batch_size = nodes_.at(source_nodes[0])->getOutput().dimension(0);
    const int memory_size = nodes_.at(source_nodes[0])->getOutput().dimension(1);

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
        source_tensor(j, i) = nodes_.at(source_nodes[i])->getError()(j, time_step); // current time-step
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
          if (links_.at(link)->getSourceNodeName() == sink_nodes[i] &&
          links_.at(link)->getSinkNodeName() == source_nodes[j])
          {
            weight_tensor(j, i) = weights_.at(links_.at(link)->getWeightName())->getWeight();
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
        if (nodes_.at(sink_nodes[i])->getStatus() == NodeStatus::activated)
        {
          derivative_tensor(j, i) = nodes_.at(sink_nodes[i])->getDerivative()(j, time_step); // current time-step
        }
        else if (nodes_.at(sink_nodes[i])->getStatus() == NodeStatus::corrected)
        {
          // std::cout << "Model::backPropogateLayerError() Previous derivative (batch_size, Sink) " << j << "," << i << std::endl;
          if (time_step + 1 < memory_size)
          {
            derivative_tensor(j, i) = nodes_.at(sink_nodes[i])->getDerivative()(j, time_step + 1); // previous time-step
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
  
  std::vector<std::string> Model::backPropogate(const int& time_step, bool cache_BP_steps, bool use_cache, int n_threads)
  {
    if (use_cache)
    {
      for (auto& BP_operations: BP_operations_cache_)
        backPropogateLayerError(BP_operations, time_step, n_threads);
      return BP_cyclic_nodes_cache_;
    }
    else
    {
      std::vector<std::string> node_names_with_cycles;
      const int max_iters = 1e6;
      for (int iter=0; iter<max_iters; ++iter)
      {
        // get the next uncorrected layer
        std::map<std::string, int> BP_operations_map;
        std::vector<OperationList> BP_operations_list;
        std::vector<std::string> source_nodes;
        getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);  

        // std::cout<<"getNextUncorrectedLayer"<<std::endl;
        // for (auto& operation: BP_operations_list)
        // {
        //   std::cout<<"Sink node: "<<operation.result.sink_node->getName()<<std::endl;
        //   for (auto& argument: operation.arguments)
        //   {
        //     std::cout<<"Source node: "<<argument.source_node->getName()<<std::endl;
        //     std::cout<<"Weight: "<<argument.weight->getName()<<std::endl;
        //   }
        // }

        // get cycles
        std::map<std::string, int> BP_operations_map_cycles = BP_operations_map;
        std::vector<OperationList> BP_operations_list_cycles = BP_operations_list;
        std::vector<std::string> source_nodes_cycles;
        getNextUncorrectedLayerCycles(BP_operations_map_cycles, BP_operations_list_cycles, source_nodes, source_nodes_cycles);

        // std::cout<<"getNextUncorrectedLayerCycles"<<std::endl;
        // for (auto& operation: BP_operations_list_cycles)
        // {
        //   std::cout<<"Sink node: "<<operation.result.sink_node->getName()<<std::endl;
        //   for (auto& argument: operation.arguments)
        //   {
        //     std::cout<<"Source node: "<<argument.source_node->getName()<<std::endl;
        //     std::cout<<"Weight: "<<argument.weight->getName()<<std::endl;
        //   }
        // }

        if (source_nodes_cycles.size() == source_nodes.size())
        { // all backward propogation steps have caught up
          // add source nodes with cycles to the backward propogation step
          for (const auto& sink_operation : BP_operations_map_cycles)
          {
            if (BP_operations_map.count(sink_operation.first) == 0)
            {
              if (cache_BP_steps) // track sink nodes with cycles
                BP_cyclic_nodes_cache_.push_back(sink_operation.first);
              else
                node_names_with_cycles.push_back(sink_operation.first);
            }
          }
          BP_operations_list = BP_operations_list_cycles;
        }

        // check if all nodes have been corrected
        if (BP_operations_list.size() == 0)
        {
          break;
        }

        // calculate the net input
        backPropogateLayerError(BP_operations_list, time_step, n_threads);

        if (cache_BP_steps)
          BP_operations_cache_.push_back(BP_operations_list);
      }
      if (cache_BP_steps)
        return BP_cyclic_nodes_cache_;
      else
        return node_names_with_cycles;
    }
  }
  
  // // [DEPRECATED]  
  // std::vector<std::string> Model::backPropogate(const int& time_step, bool cache_BP_steps, bool use_cache, int n_threads)
  // {
  //   if (use_cache)
  //   {
  //     for (auto const& sink_links_map: BP_sink_link_cache_)
  //       backPropogateLayerError(sink_links_map, time_step, n_threads);
  //     return BP_cyclic_nodes_cache_;
  //   }
  //   else
  //   {
  //     std::vector<std::string> node_names_with_cycles;
  //     const int max_iters = 1e6;
  //     for (int iter=0; iter<max_iters; ++iter)
  //     {
  //       // get the next uncorrected layer
  //       std::map<std::string, std::vector<std::string>> sink_links_map;
  //       std::vector<std::string> source_nodes;
  //       getNextUncorrectedLayer(sink_links_map, source_nodes);  

  //       // get cycles
  //       std::map<std::string, std::vector<std::string>> sink_links_map_cycles = sink_links_map;
  //       std::vector<std::string> source_nodes_cycles;
  //       getNextUncorrectedLayerCycles(sink_links_map_cycles, source_nodes, source_nodes_cycles);

  //       if (source_nodes_cycles.size() == source_nodes.size())
  //       { // all backward propogation steps have caught up
  //         // add source nodes with cycles to the backward propogation step
  //         for (const auto& sink_link : sink_links_map_cycles)
  //         {
  //           if (sink_links_map.count(sink_link.first) == 0)
  //           {
  //             if (cache_BP_steps)
  //               BP_cyclic_nodes_cache_.push_back(sink_link.first);
  //             else
  //               node_names_with_cycles.push_back(sink_link.first);
  //           }
  //         }
  //         sink_links_map = sink_links_map_cycles;
  //       }

  //       // check if all nodes have been corrected
  //       if (sink_links_map.size() == 0)
  //       {
  //         break;
  //       }

  //       // calculate the net input
  //       backPropogateLayerError(sink_links_map, time_step, n_threads);

  //       if (cache_BP_steps)
  //         BP_sink_link_cache_.push_back(sink_links_map);
  //     }
  //     if (cache_BP_steps)
  //       return BP_cyclic_nodes_cache_;
  //     else
  //       return node_names_with_cycles;
  //   }
  // }
  
  // // [DEPRECATED]
  // std::vector<std::string> Model::backPropogate(const int& time_step)
  // {
  //   std::vector<std::string> node_names_with_cycles;
  //   const int max_iters = 1e6;
  //   for (int iter=0; iter<max_iters; ++iter)
  //   {
  //     // std::cout<<"Model::backPropogate() iter :"<<iter<<::std::endl;

  //     // std::cout<<"Model::backPropogate() NodeStatuses :";
  //     // for (const auto& node_map : nodes_)
  //     // {
  //     //   if (node_map.second->getStatus() == NodeStatus::activated) std::cout<<"Node status for Node ID: "<<node_map.first<<" is Activated"<<std::endl;
  //     //   if (node_map.second->getStatus() == NodeStatus::corrected) std::cout<<"Node status for Node ID: "<<node_map.first<<" is Corrected"<<std::endl;
  //     // }

  //     // get the next uncorrected layer
  //     std::vector<std::string> links, source_nodes, sink_nodes;
  //     getNextUncorrectedLayer(links, source_nodes, sink_nodes);
  //     // std::cout<<"link size "<<links.size()<<::std::endl;

  //     // get cycles
  //     std::vector<std::string> links_cycles, source_nodes_cycles, sink_nodes_cycles;
  //     getNextUncorrectedLayerCycles(links_cycles, source_nodes, sink_nodes_cycles, source_nodes_cycles);

  //     // std::cout << "Back Propogate cycles found: " << source_nodes_cycles.size() << std::endl;
  //     if (source_nodes_cycles.size() == source_nodes.size())
  //     { // all backward propogation steps have caught up
  //       // add source nodes with cycles to the backward propogation step
  //       links.insert( links.end(), links_cycles.begin(), links_cycles.end() );
  //       sink_nodes.insert( sink_nodes.end(), sink_nodes_cycles.begin(), sink_nodes_cycles.end() );
  //       node_names_with_cycles.insert(node_names_with_cycles.end(), sink_nodes_cycles.begin(), sink_nodes_cycles.end() );
  //     }
  //     else
  //     { // remove source/sink nodes with cycles from the backward propogation step
  //       for (const std::string node_name : source_nodes_cycles)
  //       {
  //         source_nodes.erase(std::remove(source_nodes.begin(), source_nodes.end(), node_name), source_nodes.end());
  //       }
  //     }

  //     // std::cout<<"Model::backPropogate() links.size()[after cycles] :"<<links.size()<<::std::endl;
  //     // check if all nodes have been corrected
  //     if (links.size() == 0)
  //     {
  //       break;
  //     }

  //     // calculate the net input
  //     backPropogateLayerError(links, source_nodes, sink_nodes, time_step);
  //   }
  //   return node_names_with_cycles;
  // }

  void Model::TBPTT(const int& time_steps, bool cache_BP_steps, bool use_cache, int n_threads)
  {
    // check time_steps vs memory_size
    int max_steps = time_steps;
    if (time_steps >= nodes_.begin()->second->getOutput().dimension(1))
    {
      std::cout<<"Time_steps will be scaled back to the memory_size - 1."<<std::endl;
      max_steps = nodes_.begin()->second->getOutput().dimension(1) - 1;
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
            node_map.second->setStatus(NodeStatus::activated); // reinitialize cyclic nodes
            // std::cout<<"Model::TBPTT() cyclic nodes: "<<node_map.first<<std::endl;
          }   
          // std::cout<<"Model::TBPTT() output: "<<node_map.second->getError()<<" for node_name: "<<node_map.first<<std::endl;
        }
      }

      // calculate the error for each batch of memory
      if (cache_BP_steps && time_step == 0)
        node_names = backPropogate(time_step, true, false, n_threads);
      else if (cache_BP_steps && time_step > 0)
        node_names = backPropogate(time_step, false, true, n_threads);
      else
        node_names = backPropogate(time_step, cache_BP_steps, use_cache, n_threads);
    }
    // for (auto& node_map: nodes_)
    // {
    //   std::cout<<"Model::TBPTT() error: "<<node_map.second->getError()<<" for node_name: "<<node_map.first<<std::endl;
    // }
  }

  void Model::updateWeights(const int& time_steps)
  {
    // check time_steps vs memory_size
    int max_steps = time_steps;
    if (time_steps > nodes_.begin()->second->getOutput().dimension(1))
    {
      std::cout<<"Time_steps will be scaled back to the memory_size."<<std::endl;
      max_steps = nodes_.begin()->second->getOutput().dimension(1);
    }

    std::map<std::string, float> weight_derivatives;

    // calculate the average derivative for all weights
		// sum the average derivative for all time steps
		// and sum the average derivate for all time steps across shared weights
    for (const auto& link_map : links_)
    {
      if (nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected)      
      {
        // Sum the error from current and previous time-steps
        // [PARALLEL: implement threads here]
        float error_sum = 0.0;
        for (int i=0; i<max_steps; ++i)
        {
          // [PARALLEL: move to threadPool/CUDA implementations]
					// [Tests: update tests accordingly]
					Eigen::Tensor<float, 1> error_tensor = nodes_.at(link_map.second->getSinkNodeName())->getError().chip(i, 1);

					Eigen::Tensor<float, 1> output_tensor;
					if (nodes_.at(link_map.second->getSinkNodeName())->getIntegration() == NodeIntegration::Sum)
						output_tensor = nodes_.at(link_map.second->getSourceNodeName())->getOutput().chip(i, 1);
					else if (nodes_.at(link_map.second->getSinkNodeName())->getIntegration() == NodeIntegration::Product)
						output_tensor = (nodes_.at(link_map.second->getSourceNodeName())->getInput().chip(i, 1)/
							weights_.at(link_map.second->getWeightName())->getWeight()
							).unaryExpr(std::ptr_fun(checkNanInf<float>));
					else if (nodes_.at(link_map.second->getSinkNodeName())->getIntegration() == NodeIntegration::Max)
						output_tensor = nodes_.at(link_map.second->getSourceNodeName())->getOutput().chip(i, 1); // [TODO: update with correct formula]
					else
						std::cout<<"NodeIntegration type is not supported."<<std::endl; // should throw an error!

          Eigen::Tensor<float, 0> derivative_mean_tensor = (- error_tensor * output_tensor).mean(); // average derivative
          error_sum += derivative_mean_tensor(0);
        } 
        // [PARALELL: collect threads here sum the error]
				auto found = weight_derivatives.emplace(link_map.second->getWeightName(), error_sum);
				if (!found.second)
				{
					weight_derivatives.at(link_map.second->getWeightName()) += error_sum;
				}         
      }    
    }

    // update the weights
    for (const auto& weight_derivative : weight_derivatives)
      weights_.at(weight_derivative.first)->updateWeight(weight_derivative.second);
  }

  void Model::reInitializeNodeStatuses()
  {
    for (auto& node_map : nodes_)
    {
      node_map.second->setStatus(NodeStatus::initialized);
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

	bool Model::checkCompleteInputToOutput(
		const std::vector<std::string>& input_nodes, 
		const std::vector<std::string>& output_nodes,
		int n_threads)
	{
		// check that all input/output nodes exist!
		if (!checkNodeNames(input_nodes) || !checkNodeNames(output_nodes))
			return false;

		// infer the batch and memory size
		// [BUG: modifying the batch_size or memory_size causes a memory corruption error when
		//			 using the training the population after replicating and modifying the models
		//			 potential cause: the batch/memory sizes are not updated during training?]
		int batch_size_cur = nodes_.at(input_nodes[0])->getOutput().dimension(0);
		int memory_size_cur = nodes_.at(input_nodes[0])->getOutput().dimension(1);

		// check for uninitialized nodes
		int batch_size = 2;
		int memory_size = 2;
		if (batch_size_cur != 0)
			batch_size = batch_size_cur;
		if (memory_size_cur != 0)
			memory_size = memory_size_cur;
			
		// set all node outputs to zero except for the input
		// set all node derivatives to one
		// set all node errors to zero except for the output
		Eigen::Tensor<float, 2> zero(batch_size, memory_size);
		zero.setConstant(0.0f);
		Eigen::Tensor<float, 2> one(batch_size, memory_size);
		one.setConstant(1.0f);
		for (auto& node_map: nodes_)
		{
			if (std::count(input_nodes.begin(), input_nodes.end(), node_map.second->getName()) != 0)
			{
				node_map.second->setOutput(one);
				node_map.second->setError(zero);
				node_map.second->setDerivative(one);
				node_map.second->setDt(one);
			}
			else if (std::count(output_nodes.begin(), output_nodes.end(), node_map.second->getName()) != 0)
			{
				node_map.second->setOutput(zero);
				node_map.second->setError(one);
				node_map.second->setDerivative(one);
				node_map.second->setDt(one);
			}
			else
			{
				node_map.second->setOutput(zero);
				node_map.second->setError(zero);
				node_map.second->setDerivative(one);
				node_map.second->setDt(one);
			}
			node_map.second->setStatus(NodeStatus::initialized);
			node_map.second->setActivation(NodeActivation::Linear);  // safer but requires setting
																															 // the node activation back to its original value
		}

		// set all weights to 1
		for (auto& weight_map : weights_)
			weight_map.second->setWeight(1.0f);

		// Forward propogate
		for (const std::string& node_name : input_nodes)
			nodes_.at(node_name)->setStatus(NodeStatus::activated);
		forwardPropogate(0, false, false, n_threads);

		// check that all output nodes are greater than 0
		for (const std::string& node_name: output_nodes)
		{
			Eigen::Tensor<float, 0> output = nodes_.at(node_name)->getOutput().sum();
			if (output(0) == 0.0f)
				return false;
		}

		// backward propagation
		for (const std::string& node_name : output_nodes)
			nodes_.at(node_name)->setStatus(NodeStatus::corrected);
		backPropogate(0, false, false, n_threads);

		// check that all input nodes are greater than 0
		for (const std::string& node_name : input_nodes)
		{
			Eigen::Tensor<float, 0> error = nodes_.at(node_name)->getError().sum();
			if (error(0) == 0.0f)
				return false;
		}

		return true;
	}

	bool Model::checkLinksNodeAndWeightNames(std::vector<std::string>& nodes_not_found, std::vector<std::string>& weights_not_found)
	{
		bool link_names_check = true;
		for (const auto& link_map : links_)
		{
			if (!checkNodeNames({ link_map.second->getSourceNodeName() }))
			{
				link_names_check = false;
				nodes_not_found.push_back(link_map.second->getSourceNodeName());
			}
			if (!checkNodeNames({ link_map.second->getSinkNodeName() }))
			{
				link_names_check = false;
				nodes_not_found.push_back(link_map.second->getSinkNodeName());
			}
			if (!checkWeightNames({ link_map.second->getWeightName() }))
			{
				link_names_check = false;
				weights_not_found.push_back(link_map.second->getWeightName());
			}
		}
		return link_names_check;
	}

	bool Model::removeIsolatedNodes()
	{
		// key/value pair of node name and source/sink count pair
		std::map<std::string, std::pair<int, int>> node_counts;

		// count all sink/source connections for each node
		for (const auto& link_map: links_)
		{
			// source
			if (nodes_.at(link_map.second->getSourceNodeName())->getType() == NodeType::hidden)
			{
				auto found = node_counts.emplace(link_map.second->getSourceNodeName(), std::make_pair(1, 0));
				if (!found.second)
				{
					node_counts[link_map.second->getSourceNodeName()].first += 1;
				}
			}

			// sink
			if (nodes_.at(link_map.second->getSinkNodeName())->getType() == NodeType::hidden
				&& nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias)
			{
				auto found = node_counts.emplace(link_map.second->getSinkNodeName(), std::make_pair(0, 1));
				if (!found.second)
				{
					node_counts[link_map.second->getSinkNodeName()].second += 1;
				}
			}
		}

		bool dead_end_node_found = false;
		for (const auto& node_count: node_counts)
		{
			if (node_count.second.first == 0 || node_count.second.second == 0)
			{
				removeNodes({node_count.first});
				dead_end_node_found = true;
			}
		}
		return dead_end_node_found;
	}

	void Model::clearCache()
  {
    FP_operations_cache_.clear();
    BP_operations_cache_.clear();
		BP_cyclic_nodes_cache_.clear();
		output_node_cache_.clear();

    // [DEPRECATED]
    FP_sink_link_cache_.clear();
    BP_sink_link_cache_.clear();
  }
}