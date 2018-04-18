/**TODO:  Add copyright*/

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Operation.h>

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
    for (Node const& node: nodes)
    {
      // check for duplicate nodes (by id)
      if (nodes_.count(node.getId()) == 0)
      {
        nodes_.emplace(node.getId(), node);
        // nodes_[node.getId()] = node;
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
    if (!nodes_.empty() && nodes_.count(node_id) != 0)
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


  void Model::addWeights(const std::vector<Weight>& weights)
  { 
    for (Weight const& weight: weights)
    {
      // check for duplicate weights (by id)
      if (weights_.count(weight.getId()) == 0)
      {
        weights_.emplace(weight.getId(), weight);
        // weights_[weight.getId()] = weight;
      }
      else
      {
        // TODO: move to debug log
        std::cout << "Weight id " << weight.getId() << " already exists!" << std::endl;
      }
    }
  }

  Weight Model::getWeight(const int& weight_id) const
  {
    if (!weights_.empty() && weights_.count(weight_id) != 0)
    {
      return weights_.at(weight_id);
    }
    else
    {
      // TODO: move to debug log
      std::cout << "Weight id " << weight_id << " not found!" << std::endl;
    }
  }

  void Model::removeWeights(const std::vector<int>& weight_ids)
  { 
    for (int const& weight_id: weight_ids)
    {
      // check for duplicate weights (by id)
      if (weights_.count(weight_id) != 0)
      {
        weights_.erase(weight_id);
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
        links_.emplace(link.getId(), link);
        // links_[link.getId()] = link;
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
    pruneWeights();
  }

  Link Model::getLink(const int& link_id) const
  {
    if (!links_.empty() && links_.count(link_id) != 0)
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

  void Model::pruneWeights()
  {
    std::vector<int> weight_ids;
    if (weights_.empty()) { return; }
    for (const auto& weight : weights_)
    {
      bool found = false;
      if (links_.empty()) { return; }
      for (const auto& link: links_)
      {
        if (weight.second.getId() == link.second.getWeightId())
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        weight_ids.push_back(weight.first);
      }
    }
    if (weight_ids.size() != 0) { removeWeights(weight_ids); }    
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
      // if (weights_.empty()) { return; }
      // for (const auto& weight : weights_)
      // {
      //   if (weight.second.getId() == link.second.getWeightId())
      //   {
      //     found = true;
      //     break;
      //   }
      // }
      if (!found)
      {
        link_ids.push_back(link.first);
      }
    }
    if (link_ids.size() != 0) { removeLinks(link_ids); }
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
    const Eigen::Tensor<float, 2>& values,
    const int& memory_step,
    const std::vector<int>& node_ids,
    const NodeStatus& status_update)
  {
    // check dimension mismatches
    if (node_ids.size() != values.dimension(1))
    {
      std::cout << "The number of input features and the number of nodes do not match." << std::endl;
      return;
    }
    // assumes the node exists
    else if (nodes_.at(node_ids[0]).getOutput().dimension(0) != values.dimension(0))
    {
      std::cout << "The number of input samples and the node batch size does not match." << std::endl;
      return;
    }

    // // infer the memory size from the node output size
    // const int memory_size = nodes_.at(node_ids[0]).getOutput().dimension(1);

    // copy over the input values
    for (int i=0; i<node_ids.size(); ++i)
    {
      for (int j=0; j<values.dimension(0); ++j)
      {
        if (status_update == NodeStatus::activated)
        {
          // SANITY CHECK:
          // std::cout << "i" << i << " j" << j << " values: " << values.data()[i*values.dimension(0) + j] << std::endl;
          // nodes_.at(node_ids[i]).getOutputPointer()[j + values.dimension(0)*memory_step] = std::move(values.data()[i*values.dimension(0) + j]);
          // nodes_.at(node_ids[i]).getOutputPointer()[j + values.dimension(0)*memory_step] = values(j, i);
          nodes_.at(node_ids[i]).getOutputMutable()->operator()(j, memory_step) = values(j, i);
          nodes_.at(node_ids[i]).setStatus(NodeStatus::activated);
        }
        else if (status_update == NodeStatus::corrected)
        {
          // SANITY CHECK:
          // std::cout << "i" << i << " j" << j << " values: " << values.data()[i*values.dimension(0) + j] << std::endl;
          // nodes_.at(node_ids[i]).getErrorPointer()[j + values.dimension(0)*memory_step] = std::move(values.data()[i*values.dimension(0) + j]);
          // nodes_.at(node_ids[i]).getErrorPointer()[j + values.dimension(0)*memory_step] = values(j, i);
          nodes_.at(node_ids[i]).getErrorMutable()->operator()(j, memory_step) = values(j, i);
          nodes_.at(node_ids[i]).setStatus(NodeStatus::corrected);
        }
      }
    }
  }
  
  void Model::mapValuesToNodes(
    const Eigen::Tensor<float, 3>& values,
    const std::vector<int>& node_ids,
    const NodeStatus& status_update)
  {
    // check dimension mismatches
    if (node_ids.size() != values.dimension(2))
    {
      std::cout << "The number of input features and the number of nodes do not match." << std::endl;
      return;
    }
    // assumes the node exists
    else if (nodes_.at(node_ids[0]).getOutput().dimension(0) != values.dimension(0))
    {
      std::cout << "The number of input samples and the node batch size does not match." << std::endl;
      return;
    }
    else if (nodes_.at(node_ids[0]).getOutput().dimension(1) != values.dimension(1))
    {
      std::cout << "The number of input time steps and the node memory size does not match." << std::endl;
      return;
    }

    // copy over the input values
    for (int i=0; i<node_ids.size(); ++i)
    {
      for (int k=0; k<values.dimension(1); ++k)
      {
        for (int j=0; j<values.dimension(0); ++j)
        {
          if (status_update == NodeStatus::activated)
          {
            // nodes_.at(node_ids[i]).getOutputPointer()[k*values.dimension(0) + j] = values(j, k, i);
            nodes_.at(node_ids[i]).getOutputMutable()->operator()(j, k) = values(j, k, i);
            nodes_.at(node_ids[i]).setStatus(NodeStatus::activated);
          }
          else if (status_update == NodeStatus::corrected)
          {
            nodes_.at(node_ids[i]).getErrorPointer()[k*values.dimension(0) + j] = values(j, k, i);
            nodes_.at(node_ids[i]).getErrorMutable()->operator()(j, k) = values(j, k, i);
            nodes_.at(node_ids[i]).setStatus(NodeStatus::corrected);
          }
        }
      }
    }
  }
  
  void Model::getNextInactiveLayer(
    std::vector<int>& links,
    std::vector<int>& source_nodes,
    std::vector<int>& sink_nodes)
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
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::initialized)
      {
        // std::cout << "Model::getNextInactiveLayer() link_id: " << link_map.first << std::endl;
        links.push_back(link_map.second.getId());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSourceNodeId()) == 0)
        {
          source_nodes.push_back(link_map.second.getSourceNodeId());
        }
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSinkNodeId()) == 0)
        {
          sink_nodes.push_back(link_map.second.getSinkNodeId());
        }
      }
    }
  }
  
  void Model::getNextInactiveLayerBiases(
    std::vector<int>& links,
    std::vector<int>& source_nodes,
    const std::vector<int>& sink_nodes,
    std::vector<int>& sink_nodes_with_biases)
  {

    // get all the biases for the sink nodes
    for (auto& link_map : links_)
    {
      if (        
        // does not allow for cycles
        nodes_.at(link_map.second.getSourceNodeId()).getType() == NodeType::bias && 
        nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::activated &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::initialized &&
        std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSinkNodeId()) != 0 // sink node has already been identified
      )
      {
        links.push_back(link_map.second.getId());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSourceNodeId()) == 0)
        {
          source_nodes.push_back(link_map.second.getSourceNodeId());
        }
        if (std::count(sink_nodes_with_biases.begin(), sink_nodes_with_biases.end(), link_map.second.getSinkNodeId()) == 0)
        {
          sink_nodes_with_biases.push_back(link_map.second.getSinkNodeId());
        }
      }
    }
  }
  
  void Model::getNextInactiveLayerCycles(
    std::vector<int>& links,
    std::vector<int>& source_nodes,
    const std::vector<int>& sink_nodes,
    std::vector<int>& sink_nodes_with_cycles)
  {

    // get cyclic source nodes
    for (auto& link_map : links_)
    {
      if (
        (nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::initialized) &&
        // required regardless if cycles are or are not allowed
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::initialized &&
        std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSinkNodeId()) != 0 // sink node has already been identified
      )
      {
        links.push_back(link_map.second.getId());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSourceNodeId()) == 0)
        {
          source_nodes.push_back(link_map.second.getSourceNodeId());
        }
        if (std::count(sink_nodes_with_cycles.begin(), sink_nodes_with_cycles.end(), link_map.second.getSinkNodeId()) == 0)
        {
          sink_nodes_with_cycles.push_back(link_map.second.getSinkNodeId());
        }
      }
    }
  }

  // Failed attempt to link Node memory directly with the tensor used for computation
  //=================================================================================
  // void Model::forwardPropogateLayerNetInput(
  //   const std::vector<int>& links,
  //   const std::vector<int>& source_nodes,
  //   const std::vector<int>& sink_nodes)
  // {
  //   // infer the batch size from the first source node
  //   const int batch_size = nodes_.at(source_nodes[0]).getOutput().dimension(0);

  //   // concatenate the source and weight tensors
  //   // using col-major ordering where rows are the batch vectors
  //   // and cols are the nodes

  //   // source_ptr
  //   float source_ptr [source_nodes.size() * batch_size];
  //   for (int i=0; i<source_nodes.size(); ++i)
  //   {
  //     for (int j=0; j<batch_size; ++j)
  //     {
  //       if (nodes_.at(source_nodes[i]).getStatus() == NodeStatus::activated)
  //       {
  //         source_ptr[i*batch_size + j] = nodes_.at(source_nodes[i]).getOutputPointer()[j]; //current time-step
  //       }
  //       else if (nodes_.at(source_nodes[i]).getStatus() == NodeStatus::initialized)
  //       {
  //         source_ptr[i*batch_size + j] = nodes_.at(source_nodes[i]).getOutputPointer()[batch_size + j]; //previous time-step
  //       }
  //     }
  //   }

  //   // weight_ptr
  //   float weight_ptr [source_nodes.size() * sink_nodes.size()];
  //   for (int i=0; i<sink_nodes.size(); ++i)
  //   {
  //     for (int j=0; j<source_nodes.size(); ++j)
  //     {
  //       for (const int& link : links)
  //       {
  //         if (links_.at(link).getSinkNodeId() == sink_nodes[i] &&
  //         links_.at(link).getSourceNodeId() == source_nodes[j])
  //         {
  //           weight_ptr[i*source_nodes.size() + j] = weights_.at(links_.at(link).getWeightId()).getWeight();
  //           break;
  //         }
  //       }
  //     }
  //   }
  //   float sink_ptr [sink_nodes.size() * batch_size];
  //   // // Not necessary because the underlying data will not be linked with the tensor
  //   // // in the current implementation
  //   // for (int i=0; i<sink_nodes.size(); ++i)
  //   // {
  //   //   for (int j=0; j<batch_size; ++j)
  //   //   {
  //   //     sink_ptr[i*batch_size + j] = nodes_.at(sink_nodes[i]).getOutputPointer()[j]; //current time-step
  //   //   }
  //   // }

  //   // construct the source and weight tensors
  //   Eigen::TensorMap<Eigen::Tensor<float, 2>> source_tensor(source_ptr, batch_size, source_nodes.size());
  //   // std::cout << "source_tensor " << source_tensor << std::endl;
  //   Eigen::TensorMap<Eigen::Tensor<float, 2>> weight_tensor(weight_ptr, source_nodes.size(), sink_nodes.size());
  //   // std::cout << "weight_tensor " << weight_tensor << std::endl;
  //   Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_tensor(sink_ptr, batch_size, sink_nodes.size());

  //   // compute the output tensor
  //   Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
  //   sink_tensor = source_tensor.contract(weight_tensor, product_dims);
  //   // std::cout << "sink_tensor " << sink_tensor << std::endl;

  //   // update the sink nodes
  //   mapValuesToNodes(sink_tensor, 0, sink_nodes, NodeStatus::activated);
  //   // std::cout<<&sink_ptr[0]<<std::endl;
  //   // std::cout<<sink_ptr[0]<<std::endl;
  //   // std::cout<<&nodes_.at(sink_nodes[0]).getOutputPointer()[0]<<std::endl;
  //   // std::cout<<nodes_.at(sink_nodes[0]).getOutputPointer()[0]<<std::endl;
  // }
  //=================================================================================

  void Model::forwardPropogateLayerNetInput(
    const std::vector<int>& links,
    const std::vector<int>& source_nodes,
    const std::vector<int>& sink_nodes,
    const int& time_step)
  {
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(source_nodes[0]).getOutput().dimension(0);

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
          // std::cout<<"Model::forwardPropogateLayerNetInput() source_node prev: "<<source_nodes[i]<<std::endl;
          source_tensor(j, i) = nodes_.at(source_nodes[i]).getOutput()(j, time_step + 1); //previous time-step
        }
      }
    }

    Eigen::Tensor<float, 2> weight_tensor(source_nodes.size(), sink_nodes.size());
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<source_nodes.size(); ++j)
      {
        for (const int& link : links)
        {
          if (links_.at(link).getSinkNodeId() == sink_nodes[i] &&
          links_.at(link).getSourceNodeId() == source_nodes[j])
          {
            weight_tensor(j, i) = weights_.at(links_.at(link).getWeightId()).getWeight();
            break;
          }
        }
      }
    }

    // compute the output tensor
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<float, 2> sink_tensor = source_tensor.contract(weight_tensor, product_dims);

    // update the sink nodes
    mapValuesToNodes(sink_tensor, time_step, sink_nodes, NodeStatus::activated);
  }
  
  void Model::forwardPropogateLayerActivation(
    const std::vector<int>& sink_nodes,
      const int& time_step)
  {
    for (const int& node : sink_nodes)
    {
      nodes_.at(node).calculateActivation(time_step);
      nodes_.at(node).calculateDerivative(time_step);
    }
  }
  
  void Model::forwardPropogate(const int& time_step)
  {
    const int max_iters = 1e6;
    for (int iter; iter<max_iters; ++iter)
    {      
      // std::cout<<"Model::forwardPropogate() iter: "<<iter<<std::endl;

      // get the next hidden layer
      std::vector<int> links, source_nodes, sink_nodes;
      getNextInactiveLayer(links, source_nodes, sink_nodes);

      // get biases,
      std::vector<int> sink_nodes_with_biases;
      getNextInactiveLayerBiases(links, source_nodes, sink_nodes, sink_nodes_with_biases);
      
      // get cycles
      std::vector<int> links_cycles, source_nodes_cycles, sink_nodes_cycles;
      getNextInactiveLayerCycles(links_cycles, source_nodes_cycles, sink_nodes, sink_nodes_cycles);

      // std::cout<<"Model::forwardPropogate() sink_nodes_cycles.size(): "<<sink_nodes_cycles.size()<<std::endl;
      if (sink_nodes_cycles.size() == sink_nodes.size())
      { // all forward propogation steps have caught up
        // add sink nodes with cycles to the forward propogation step
        links.insert( links.end(), links_cycles.begin(), links_cycles.end() );
        source_nodes.insert( source_nodes.end(), source_nodes_cycles.begin(), source_nodes_cycles.end() );
      }
      else
      { // remove source/sink nodes with cycles from the forward propogation step
        for (const int node_id : sink_nodes_cycles)
        {
          sink_nodes.erase(std::remove(sink_nodes.begin(), sink_nodes.end(), node_id), sink_nodes.end());
        }
      }

      // check if all nodes have been activated
      if (links.size() == 0)
      {
        break;
      }      
      // std::cout<<"Model::forwardPropogate() links.size(): "<<links.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() source nodes: "<<source_nodes.size()<<std::endl;
      // std::cout<<"Model::forwardPropogate() sink nodes: "<<sink_nodes.size()<<std::endl;

      // calculate the net input
      forwardPropogateLayerNetInput(links, source_nodes, sink_nodes, time_step);

      // calculate the activation
      forwardPropogateLayerActivation(sink_nodes, time_step);
    }
  }

  void Model::FPTT(const int& time_steps, 
    const Eigen::Tensor<float, 3>& values,
    const std::vector<int> node_ids)
  {
    for (int time_step=0; time_step<time_steps; ++time_step)
    {
      // std::cout<<"Model::FPTT() time_step: "<<time_step<<std::endl;
      if (time_step>0)
      {
        // move to the next memory step
        for (auto& node_map: nodes_)
        {          
          node_map.second.saveCurrentOutput();
          node_map.second.saveCurrentDerivative();
          if (std::count(node_ids.begin(), node_ids.end(), node_map.first) == 0)
          {
            node_map.second.setStatus(NodeStatus::initialized); // reinitialize non-input nodes
          }   
          // std::cout<<"Model::FPTT() output: "<<node_map.second.getOutput()<<" for node_id: "<<node_map.first<<std::endl;
        }
      }

      // initialize nodes for the next time-step
      const Eigen::Tensor<float, 2> active_values = values.chip(time_step, 1);
      // std::cout<<"Model::FPTT() active_values: "<<active_values<<std::endl;
      mapValuesToNodes(active_values, 0, node_ids, NodeStatus::activated);

      forwardPropogate(0); // always working at the current head of memory
    }
  }
  
  void Model::calculateError(
    const Eigen::Tensor<float, 2>& values, const std::vector<int>& node_ids)
  {
    //TODO: encapsulate into a seperate method
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(node_ids[0]).getOutput().dimension(0);

    //TODO: encapsulate into a seperate method
    // check dimension mismatches
    if (node_ids.size() != values.dimension(1))
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
    // float node_ptr [node_ids.size() * batch_size];
    // for (int i=0; i<node_ids.size(); ++i)
    // {
    //   for (int j=0; j<batch_size; ++j)
    //   {
    //     node_ptr[i*batch_size + j] = nodes_.at(node_ids[i]).getOutputPointer()[j];
    //   }
    // }
    // Eigen::TensorMap<Eigen::Tensor<float, 2>> node_tensor(node_ptr, batch_size, node_ids.size());
    Eigen::Tensor<float, 2> node_tensor(batch_size, node_ids.size());
    for (int i=0; i<node_ids.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        node_tensor(j, i) = nodes_.at(node_ids[i]).getOutput()(j, 0); // current time-step
      }
    }

    // calculate the model error wrt the expected model output
    Eigen::Tensor<float, 2> error_tensor(batch_size, node_ids.size());
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
    mapValuesToNodes(error_tensor, 0, node_ids, NodeStatus::corrected);
  }
  
  void Model::getNextUncorrectedLayer(
    std::vector<int>& links,
    std::vector<int>& source_nodes,
    std::vector<int>& sink_nodes)
  {
    links.clear();
    source_nodes.clear();
    sink_nodes.clear();

    // get all links where the source node is corrected and the sink node is active
    // including biases
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::corrected && 
        nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::activated)
      {
        links.push_back(link_map.second.getId());
        // could use std::set instead to check for duplicates
        if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSinkNodeId()) == 0)
        {
          source_nodes.push_back(link_map.second.getSinkNodeId());
        }
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSourceNodeId()) == 0)
        {
          sink_nodes.push_back(link_map.second.getSourceNodeId());
        }
      }
    }
  }
  
  void Model::getNextUncorrectedLayerCycles(
    std::vector<int>& links,
    const std::vector<int>& source_nodes,
    std::vector<int>& sink_nodes,
    std::vector<int>& source_nodes_with_cycles)
  {

    // allows for cycles
    for (auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::corrected && 
        std::count(links.begin(), links.end(), link_map.second.getId()) == 0 && // unique links 
        nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::corrected &&
        std::count(source_nodes.begin(), source_nodes.end(), link_map.second.getSinkNodeId()) != 0 // sink node has already been identified)
      ) 
      {
        links.push_back(link_map.second.getId());
        // could use std::set instead to check for duplicates
        if (std::count(sink_nodes.begin(), sink_nodes.end(), link_map.second.getSourceNodeId()) == 0)
        {
          sink_nodes.push_back(link_map.second.getSourceNodeId());
        }
        if (std::count(source_nodes_with_cycles.begin(), source_nodes_with_cycles.end(), link_map.second.getSinkNodeId()) == 0)
        {
          source_nodes_with_cycles.push_back(link_map.second.getSinkNodeId());
        }
      }
    }
  }

  // Failed attempt to link Node memory direction with computational tensor
  //=======================================================================
  // void Model::backPropogateLayerError(
  //   const std::vector<int>& links,
  //   const std::vector<int>& source_nodes,
  //   const std::vector<int>& sink_nodes)
  // {
  //   // infer the batch size from the first source node
  //   const int batch_size = nodes_.at(source_nodes[0]).getOutput().dimension(0);

  //   // concatenate the source and weight tensors
  //   // using col-major ordering where rows are the batch vectors
  //   // and cols are the nodes

  //   // source_ptr
  //   float source_ptr [source_nodes.size() * batch_size];
  //   for (int i=0; i<source_nodes.size(); ++i)
  //   {
  //     for (int j=0; j<batch_size; ++j)
  //     {
  //       source_ptr[i*batch_size + j] = nodes_.at(source_nodes[i]).getErrorPointer()[j];
  //     }
  //   }
  //   // weight_ptr
  //   float weight_ptr [source_nodes.size() * sink_nodes.size()];
  //   for (int i=0; i<sink_nodes.size(); ++i)
  //   {
  //     for (int j=0; j<source_nodes.size(); ++j)
  //     {
  //       for (const int& link : links)
  //       {
  //         if (links_.at(link).getSourceNodeId() == sink_nodes[i] &&
  //         links_.at(link).getSinkNodeId() == source_nodes[j])
  //         {
  //           weight_ptr[i*source_nodes.size() + j] = weights_.at(links_.at(link).getWeightId()).getWeight();
  //           break;
  //         }
  //       }
  //     }
  //   }
  //   // derivative_ptr
  //   float derivative_ptr [sink_nodes.size() * batch_size];
  //   for (int i=0; i<sink_nodes.size(); ++i)
  //   {
  //     for (int j=0; j<batch_size; ++j)
  //     {
  //       derivative_ptr[i*batch_size + j] = nodes_.at(sink_nodes[i]).getDerivativePointer()[j];
  //     }
  //   }
  //   float sink_ptr [sink_nodes.size() * batch_size];
  //   for (int i=0; i<sink_nodes.size(); ++i)
  //   {
  //     for (int j=0; j<batch_size; ++j)
  //     {
  //       sink_ptr[i*batch_size + j] = nodes_.at(sink_nodes[i]).getOutputPointer()[j];
  //     }
  //   }

  //   // construct the source and weight tensors
  //   Eigen::TensorMap<Eigen::Tensor<float, 2>> source_tensor(source_ptr, batch_size, source_nodes.size());
  //   // std::cout << "source_tensor " << source_tensor << std::endl;
  //   Eigen::TensorMap<Eigen::Tensor<float, 2>> weight_tensor(weight_ptr, source_nodes.size(), sink_nodes.size());
  //   // std::cout << "weight_tensor " << weight_tensor << std::endl;
  //   Eigen::TensorMap<Eigen::Tensor<float, 2>> derivative_tensor(derivative_ptr, batch_size, sink_nodes.size());
  //   // std::cout << "derivative_tensor " << derivative_tensor << std::endl;
  //   Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_tensor(sink_ptr, batch_size, sink_nodes.size());

  //   // compute the output tensor
  //   Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
  //   sink_tensor = source_tensor.contract(weight_tensor, product_dims) * derivative_tensor;
  //   // std::cout << "sink_tensor " << sink_tensor << std::endl;

  //   // update the sink nodes
  //   mapValuesToNodes(sink_tensor, 0, sink_nodes, NodeStatus::corrected);
  // }
  //=======================================================================

  void Model::backPropogateLayerError(
    const std::vector<int>& links,
    const std::vector<int>& source_nodes,
    const std::vector<int>& sink_nodes,
    const int& time_step)
  {
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(source_nodes[0]).getOutput().dimension(0);

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
        for (const int& link : links)
        {
          if (links_.at(link).getSourceNodeId() == sink_nodes[i] &&
          links_.at(link).getSinkNodeId() == source_nodes[j])
          {
            weight_tensor(j, i) = weights_.at(links_.at(link).getWeightId()).getWeight();
            break;
          }
        }
      }
    }
    
    // construct the derivative tensors
    std::vector<int> sink_nodes_prev;
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
          std::cout << "Model::backPropogateLayerError() Previous derivative (batch_size, Sink) " << j << "," << i << std::endl;
          derivative_tensor(j, i) = nodes_.at(sink_nodes[i]).getDerivative()(j, time_step + 1); // previous time-step
          if (std::count(sink_nodes_prev.begin(), sink_nodes_prev.end(), i) == 0) sink_nodes_prev.push_back(i);
        }        
      }
    }

    // compute the output tensor
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<float, 2> sink_tensor = source_tensor.contract(weight_tensor, product_dims) * derivative_tensor;

    std::vector<int> sink_nodes_cur;
    if (sink_nodes_prev.size()>0)
    {
      std::cout<<"Model::backPropogateLayerError() sink_nodes_prev.size(): "<<sink_nodes_prev.size()<<std::endl;
      std::cout<<"Model::backPropogateLayerError() sink_nodes.size(): "<<sink_nodes.size()<<std::endl;
      // split the sink_tensor into current and previous
      Eigen::Tensor<float, 2> sink_tensor_cur(batch_size, sink_nodes.size() - sink_nodes_prev.size());
      Eigen::Tensor<float, 2> sink_tensor_prev(batch_size, sink_nodes_prev.size());
      int sink_tensor_cur_iter = 0;
      int sink_tensor_prev_iter = 0;
      for (int i=0; i<sink_nodes.size(); ++i)
      {
        if (std::count(sink_nodes_prev.begin(), sink_nodes_prev.end(), i) != 0)
        {
          // std::cout<<"Model::backPropogateLayerError() sink_tensor_prev_iter: "<<sink_tensor_prev_iter<<std::endl;
          for (int j=0; j<batch_size; ++j)
          {
            sink_tensor_prev(j, sink_tensor_prev_iter) = sink_tensor(j, i);
          }
          sink_tensor_prev_iter += 1;
        }
        else
        {
          // std::cout<<"Model::backPropogateLayerError() sink_tensor_cur_iter: "<<sink_tensor_cur_iter<<std::endl;
          for (int j=0; j<batch_size; ++j)
          {
            sink_tensor_cur(j, sink_tensor_cur_iter) = sink_tensor(j, i);            
          }
          sink_tensor_cur_iter += 1;
          sink_nodes_cur.push_back(i);
        }
      }
      std::cout<<"Model::backPropogateLayerError() sink_tensor_cur: "<<sink_tensor_cur<<std::endl;
      std::cout<<"Model::backPropogateLayerError() sink_tensor_prev: "<<sink_tensor_prev<<std::endl;

      // update the sink nodes errors for the current time-step
      mapValuesToNodes(sink_tensor_cur, time_step, sink_nodes_cur, NodeStatus::corrected);

      // update the sink nodes errors for the previous time-step
      mapValuesToNodes(sink_tensor_prev, time_step + 1, sink_nodes_prev, NodeStatus::corrected);
    }
    else
    {
      // update the sink nodes errors for the current time-step
      mapValuesToNodes(sink_tensor, time_step, sink_nodes, NodeStatus::corrected);
    }
  }
  
  void Model::backPropogate(const int& time_step)
  {
    std::vector<int> node_ids_with_cycles;
    // const int max_iters = 1e6;
    const int max_iters = 5;
    for (int iter; iter<max_iters; ++iter)
    {
      std::cout<<"Model::backPropogate() iter :"<<iter<<::std::endl;
      // get the next uncorrected layer
      std::vector<int> links, source_nodes, sink_nodes;
      getNextUncorrectedLayer(links, source_nodes, sink_nodes);

      // get cycles
      std::vector<int> links_cycles, source_nodes_cycles, sink_nodes_cycles;
      getNextUncorrectedLayerCycles(links_cycles, source_nodes, sink_nodes_cycles, source_nodes_cycles);

      // std::cout << "Back Propogate cycles found: " << source_nodes_cycles.size() << std::endl;
      if (source_nodes_cycles.size() == source_nodes.size())
      { // all backward propogation steps have caught up
        // add source nodes with cycles to the backward propogation step
        links.insert( links.end(), links_cycles.begin(), links_cycles.end() );
        sink_nodes.insert( sink_nodes.end(), sink_nodes_cycles.begin(), sink_nodes_cycles.end() );
        node_ids_with_cycles.insert(node_ids_with_cycles.end(), source_nodes_cycles.begin(), source_nodes_cycles.end() );
      }
      else
      { // remove source/sink nodes with cycles from the backward propogation step
        for (const int node_id : source_nodes_cycles)
        {
          source_nodes.erase(std::remove(source_nodes.begin(), source_nodes.end(), node_id), source_nodes.end());
        }
      }

      // check if all nodes have been corrected
      if (links.size() == 0)
      {
        break;
      }

      // calculate the net input
      backPropogateLayerError(links, source_nodes, sink_nodes, time_step);
    }
  }

  void Model::TBPTT(const int& time_steps,
    const std::vector<int> node_ids)
  {
    for (int time_step=0; time_step<time_steps; ++time_step)
    {
      std::cout<<"Model::TBPTT() time_step: "<<time_step<<std::endl;
      if (time_step>0)
      {
        for (auto& node_map: nodes_)
        {
          if (std::count(node_ids.begin(), node_ids.end(), node_map.first) == 0)
          {
            node_map.second.setStatus(NodeStatus::activated); // reinitialize non-output nodes
          }   
          // std::cout<<"Model::TBPTT() output: "<<node_map.second.getError()<<" for node_id: "<<node_map.first<<std::endl;
        }
      }

      // calculate the error for each batch of memory
      backPropogate(time_step);
    }
  }

  void Model::updateWeights()
  {

    std::map<int, std::vector<float>> weight_derivatives;  
    // initalize the map
    for (const auto& weight_map: weights_)  
    {
      const std::vector<float> derivatives;
      weight_derivatives.emplace(weight_map.first, derivatives);
    }

    // collect the derivative for all weights
    for (const auto& link_map : links_)
    {
      if (nodes_.at(link_map.second.getSinkNodeId()).getStatus() == NodeStatus::corrected)
      {

        Eigen::Tensor<float, 1> error_tensor = nodes_.at(link_map.second.getSinkNodeId()).getError().chip(0, 1); // first time-step
        Eigen::Tensor<float, 1> output_tensor = nodes_.at(link_map.second.getSourceNodeId()).getOutput().chip(0, 1);  // first time-step
        // auto derivative_tensor = - error_tensor * output_tensor; // derivative of the weight wrt the error
        // Eigen::Tensor<float, 0> derivative_mean_tensor = derivative_tensor.mean(); // average derivative
        Eigen::Tensor<float, 0> derivative_mean_tensor = (- error_tensor * output_tensor).mean(); // average derivative
        // std::cout<<"derivative_mean_tensor "<<derivative_mean_tensor(0)<<std::endl;
        weight_derivatives.at(link_map.second.getWeightId()).push_back(derivative_mean_tensor(0));
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
}