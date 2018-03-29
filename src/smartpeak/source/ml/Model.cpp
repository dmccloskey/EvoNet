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

    // get all the biases for the sink nodes
    for (auto& link_map : links_)
    {
      if (        
        // does not allow for cycles
        // nodes_.at(link_map.second.getSourceNodeId()).getType() == NodeType::bias && 
        // nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::activated && 
        // allows for cycles
        (nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::activated || 
          nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::initialized) && 
        std::count(links.begin(), links.end(), link_map.second.getId()) == 0 && // unique links\
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
      }
    }
  }

  void Model::initNodes(const int& batch_size)
  {
    for (auto& node_map : nodes_)
    {
      node_map.second.initNode(batch_size);
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
    else if (nodes_.at(node_ids[0]).getOutput().size() != values.dimension(0))
    {
      std::cout << "The number of input samples and the node batch size does not match." << std::endl;
      return;
    }

    // copy over the input values
    for (int i=0; i<node_ids.size(); ++i)
    {
      if (status_update == NodeStatus::activated)
      {
        for (int j=0; j<nodes_.at(node_ids[i]).getOutput().size(); ++j)
        {
          // SANITY CHECK:
          // std::cout << "i" << i << " j" << j << " values: " << values.data()[i*values.dimension(0) + j] << std::endl;
          nodes_.at(node_ids[i]).getOutputPointer()[j] = values.data()[i*values.dimension(0) + j];
          nodes_.at(node_ids[i]).setStatus(NodeStatus::activated);
        }
      }
      else if (status_update == NodeStatus::corrected)
      {
        for (int j=0; j<nodes_.at(node_ids[i]).getError().size(); ++j)
        {
          // SANITY CHECK:
          // std::cout << "i" << i << " j" << j << " values: " << values.data()[i*values.dimension(0) + j] << std::endl;
          nodes_.at(node_ids[i]).getErrorPointer()[j] = values.data()[i*values.dimension(0) + j];
          nodes_.at(node_ids[i]).setStatus(NodeStatus::corrected);
        }
      }
    }
  }

  void Model::forwardPropogateLayerNetInput(
    const std::vector<int>& links,
    const std::vector<int>& source_nodes,
    const std::vector<int>& sink_nodes)
  {
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(source_nodes[0]).getOutput().size();

    // concatenate the source and weight tensors
    // using col-major ordering where rows are the batch vectors
    // and cols are the nodes

    // source_ptr
    float source_ptr [source_nodes.size() * batch_size];
    for (int i=0; i<source_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        source_ptr[i*batch_size + j] = nodes_.at(source_nodes[i]).getOutputPointer()[j];
      }
    }

    // weight_ptr
    float weight_ptr [source_nodes.size() * sink_nodes.size()];
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<source_nodes.size(); ++j)
      {
        for (const int& link : links)
        {
          if (links_.at(link).getSinkNodeId() == sink_nodes[i] &&
          links_.at(link).getSourceNodeId() == source_nodes[j])
          {
            weight_ptr[i*source_nodes.size() + j] = weights_.at(links_.at(link).getWeightId()).getWeight();
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
        sink_ptr[i*batch_size + j] = nodes_.at(sink_nodes[i]).getOutputPointer()[j];
      }
    }

    // construct the source and weight tensors
    Eigen::TensorMap<Eigen::Tensor<float, 2>> source_tensor(source_ptr, batch_size, source_nodes.size());
    // std::cout << "source_tensor " << source_tensor << std::endl;
    Eigen::TensorMap<Eigen::Tensor<float, 2>> weight_tensor(weight_ptr, source_nodes.size(), sink_nodes.size());
    // std::cout << "weight_tensor " << weight_tensor << std::endl;
    Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_tensor(sink_ptr, batch_size, sink_nodes.size());

    // compute the output tensor
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    sink_tensor = source_tensor.contract(weight_tensor, product_dims);
    // std::cout << "sink_tensor " << sink_tensor << std::endl;

    // update the sink nodes
    mapValuesToNodes(sink_tensor, sink_nodes, NodeStatus::activated);
    // std::cout<<&sink_ptr[0]<<std::endl;
    // std::cout<<sink_ptr[0]<<std::endl;
    // std::cout<<&nodes_.at(sink_nodes[0]).getOutputPointer()[0]<<std::endl;
    // std::cout<<nodes_.at(sink_nodes[0]).getOutputPointer()[0]<<std::endl;
  }
  
  void Model::forwardPropogateLayerActivation(
    const std::vector<int>& sink_nodes)
  {
    for (const int& node : sink_nodes)
    {
      nodes_.at(node).calculateActivation();
      nodes_.at(node).calculateDerivative();
    }
  }
  
  void Model::forwardPropogate()
  {
    const int max_iters = 1e6;
    for (int iter; iter<max_iters; ++iter)
    {
      // get the next hidden layer
      std::vector<int> links, source_nodes, sink_nodes;
      getNextInactiveLayer(links, source_nodes, sink_nodes);

      // check if all nodes have been activated
      if (links.size() == 0)
      {
        break;
      }

      // calculate the net input
      forwardPropogateLayerNetInput(links, source_nodes, sink_nodes);

      // calculate the activation
      forwardPropogateLayerActivation(sink_nodes);
    }
  }
  
  void Model::calculateError(
    const Eigen::Tensor<float, 2>& values, const std::vector<int>& node_ids)
  {
    //TODO: encapsulate into a seperate method
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(node_ids[0]).getOutput().size();

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
    float node_ptr [node_ids.size() * batch_size];
    for (int i=0; i<node_ids.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        node_ptr[i*batch_size + j] = nodes_.at(node_ids[i]).getOutputPointer()[j];
      }
    }
    Eigen::TensorMap<Eigen::Tensor<float, 2>> node_tensor(node_ptr, batch_size, node_ids.size());

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
    mapValuesToNodes(error_tensor, node_ids, NodeStatus::corrected);
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
    // except for biases
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

    // allows for cycles
    for (auto& link_map : links_)
    {
      if ((nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::corrected || 
          nodes_.at(link_map.second.getSourceNodeId()).getStatus() == NodeStatus::activated) && 
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
      }
    }
  }

  void Model::backPropogateLayerError(
    const std::vector<int>& links,
    const std::vector<int>& source_nodes,
    const std::vector<int>& sink_nodes)
  {
    // infer the batch size from the first source node
    const int batch_size = nodes_.at(source_nodes[0]).getOutput().size();

    // concatenate the source and weight tensors
    // using col-major ordering where rows are the batch vectors
    // and cols are the nodes

    // source_ptr
    float source_ptr [source_nodes.size() * batch_size];
    for (int i=0; i<source_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        source_ptr[i*batch_size + j] = nodes_.at(source_nodes[i]).getErrorPointer()[j];
      }
    }
    // weight_ptr
    float weight_ptr [source_nodes.size() * sink_nodes.size()];
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<source_nodes.size(); ++j)
      {
        for (const int& link : links)
        {
          if (links_.at(link).getSourceNodeId() == sink_nodes[i] &&
          links_.at(link).getSinkNodeId() == source_nodes[j])
          {
            weight_ptr[i*source_nodes.size() + j] = weights_.at(links_.at(link).getWeightId()).getWeight();
            break;
          }
        }
      }
    }
    // derivative_ptr
    float derivative_ptr [sink_nodes.size() * batch_size];
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        derivative_ptr[i*batch_size + j] = nodes_.at(sink_nodes[i]).getDerivativePointer()[j];
      }
    }
    float sink_ptr [sink_nodes.size() * batch_size];
    for (int i=0; i<sink_nodes.size(); ++i)
    {
      for (int j=0; j<batch_size; ++j)
      {
        sink_ptr[i*batch_size + j] = nodes_.at(sink_nodes[i]).getOutputPointer()[j];
      }
    }

    // construct the source and weight tensors
    Eigen::TensorMap<Eigen::Tensor<float, 2>> source_tensor(source_ptr, batch_size, source_nodes.size());
    // std::cout << "source_tensor " << source_tensor << std::endl;
    Eigen::TensorMap<Eigen::Tensor<float, 2>> weight_tensor(weight_ptr, source_nodes.size(), sink_nodes.size());
    // std::cout << "weight_tensor " << weight_tensor << std::endl;
    Eigen::TensorMap<Eigen::Tensor<float, 2>> derivative_tensor(derivative_ptr, batch_size, sink_nodes.size());
    // std::cout << "derivative_tensor " << derivative_tensor << std::endl;
    Eigen::TensorMap<Eigen::Tensor<float, 2>> sink_tensor(sink_ptr, batch_size, sink_nodes.size());

    // compute the output tensor
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    sink_tensor = source_tensor.contract(weight_tensor, product_dims) * derivative_tensor;
    // std::cout << "sink_tensor " << sink_tensor << std::endl;

    // update the sink nodes
    mapValuesToNodes(sink_tensor, sink_nodes, NodeStatus::corrected);
  }
  
  void Model::backPropogate()
  {
    const int max_iters = 1e6;
    for (int iter; iter<max_iters; ++iter)
    {
      // std::cout<<"iter # "<<iter<<::std::endl;
      // get the next hidden layer
      std::vector<int> links, source_nodes, sink_nodes;
      getNextUncorrectedLayer(links, source_nodes, sink_nodes);
      // std::cout<<"link size "<<links.size()<<::std::endl;

      // check if all nodes have been corrected
      if (links.size() == 0)
      {
        break;
      }

      // calculate the net input
      backPropogateLayerError(links, source_nodes, sink_nodes);
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
        const int batch_size = nodes_.at(link_map.second.getSinkNodeId()).getError().size(); // infer the batch_size
        Eigen::TensorMap<Eigen::Tensor<float, 1>> error_tensor(nodes_.at(link_map.second.getSinkNodeId()).getErrorPointer(), batch_size);
        Eigen::TensorMap<Eigen::Tensor<float, 1>> output_tensor(nodes_.at(link_map.second.getSourceNodeId()).getOutputPointer(), batch_size);
        auto derivative_tensor = - error_tensor * output_tensor; // derivative of the weight wrt the error
        Eigen::Tensor<float, 0> derivative_mean_tensor = derivative_tensor.mean(); // average derivative
        // std::cout<<"derivative_mean_tensor "<<derivative_mean_tensor(0)<<std::endl;
        weight_derivatives[link_map.second.getWeightId()].push_back(derivative_mean_tensor(0));
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