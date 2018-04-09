/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Operation.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <tuple>

namespace SmartPeak
{

  enum class ModelLossFunction
  {
    MSE = 0,
    L2Norm = 1,
    EuclideanDistance = 2,
    CrossEntropy = 3,
    NegativeLogLikelihood = 4
  };

  /**
    @brief Directed Network Model
  */
  class Model
  {
public:
    Model(); ///< Default constructor
    Model(const int& id); ///< Explicit constructor  
    ~Model(); ///< Default destructor

    inline bool operator==(const Model& other) const
    {
      return
        std::tie(
          id_,
          links_,
          nodes_,
          weights_
        ) == std::tie(
          other.id_,
          other.links_,
          other.nodes_,
          other.weights_
        )
      ;
    }

    inline bool operator!=(const Model& other) const
    {
      return !(*this == other);
    }

    /**
      @brief Initialize all link weights
    */ 
    void initWeights();

    /**
      @brief Initialize all node output to zero.
        The node statuses are then changed to NodeStatus::deactivated

      @param[in] batch_size Batch size of the output, error, and derivative node vectors
      @param[in] memory_size Memory size of the output, error, and derivative node vectors
    */ 
    void initNodes(const int& batch_size, const int& memory_size);

    /**
      @brief Assigns output or error values to the nodes.
        The node statuses are then changed accordingly (i.e.,
        status_update of "activated" will update the output values
        of the node and status_update of "corrected" will update
        the error values of the node.

      dimensions of batch size by memory size by nodes

      @param[in] values Values to assign to the node
      @param[in] node_ids 
      @param[in] status_update
    */ 
    void mapValuesToNodes(
      const Eigen::Tensor<float, 3>& values,
      const std::vector<int>& node_ids,
      const NodeStatus& status_update);

    /**
      @brief Assigns output or error values to the nodes.
        The node statuses are then changed accordingly (i.e.,
        status_update of "activated" will update the output values
        of the node and status_update of "corrected" will update
        the error values of the node.

      dimensions of batch size by nodes

      @param[in] values Values to assign to the node
      @param[in] node_ids 
      @param[in] status_update
    */ 
    void mapValuesToCurrentNodes(
      const Eigen::Tensor<float, 2>& values,
      const std::vector<int>& node_ids,
      const NodeStatus& status_update);
 
    /**
      @brief A prelude to a forward propogation step. Returns a vector of links
        and associated nodes that satisfy the following conditions:
        1. all sink output values are unknown (i.e. inactive),
        2. all source node output values are known (i.e. active).
        3. all nodes need not be the same type

      @param[out] Links
      @param[out] source_nodes
      @param[out] sink_nodes
    */ 
    void getNextInactiveLayer(
      std::vector<int>& links,
      std::vector<int>& source_nodes,
      std::vector<int>& sink_nodes);
 
    /**
      @brief A prelude to a forward propogation step. Computes the net
        input into all nodes composing the next layer:
        1. all sink output values are unknown (i.e. inactive),
        2. all source node output values are known (i.e. active).

      Note that nodes need not be the same type.

      @param[out] Links
      @param[out] source_nodes
      @param[out] sink_nodes

      OPTIMIZATION:
      pass memory to tensors so that when the tensors compute the matrices
      the underlying node values are automatically updated
    */ 
    void forwardPropogateLayerNetInput(
      const std::vector<int>& links,
      const std::vector<int>& source_nodes,
      const std::vector<int>& sink_nodes);
 
    /**
      @brief Completion of a forward propogation step. Computes the net
        activation for all nodes in the tensor layer.

      Note before computing the activation, the layer tensor will be split
        according to the node type, and the corresponding activation
        function will be applied

      @param[in] sink_nodes
    */ 
    void forwardPropogateLayerActivation(
      const std::vector<int>& sink_nodes);
 
    /**
      @brief Foward propogation of the network model.
        All node outputs and derivatives are calculating
        starting from the input nodes.  Each node status is
        changed from "initialized" to "activated" when the
        outputs and derivatives are calculated.
    */ 
    void forwardPropogate();    
 
    /**
      @brief Foward propogation through time (FPTT) of the network model.
        All node outputs and derivatives are calculating
        starting from the input nodes.  Each node status is
        changed from "initialized" to "activated" when the
        outputs and derivatives are calculated.  This is repeated
        for n_time steps without weight updates.

      @param[in] time_steps The number of time_steps forward to 
        continuously accumulate errors.
    */ 
    void FPTT(const int& time_steps);
 
    /**
      @brief Calculates the error of the model with respect to
        the expected values

      @param[in] values Expected node output values
      @param[in] node_ids Output nodes
    */ 
    void calculateError(const Eigen::Tensor<float, 2>& values, const std::vector<int>& node_ids);
 
    /**
      @brief Calculates the error of the model through time (CETT)
        with respect to the expected values

      @param[in] values Expected node output values
      @param[in] node_ids Output nodes
    */ 
    void CETT(const Eigen::Tensor<float, 3>& values, const std::vector<int>& node_ids);
 
    /**
      @brief A prelude to a back propogation step.  Returns a vector of links
        and associated nodes that satisfy the following conditions:
        1. all sink error values are unknown (i.e. active),
        2. all source error values are known (i.e. corrected).
        3. all nodes need not be the same type

      @param[out] Links
      @param[out] source_nodes
      @param[out] sink_nodes
    */ 
    void getNextUncorrectedLayer(
      std::vector<int>& links,
      std::vector<int>& source_nodes,
      std::vector<int>& sink_nodes);
 
    /**
      @brief A back propogation step. Computes the net
        error into all nodes composing the next layer:
        1. all sink error values are unknown (i.e. active),
        2. all source error values are known (i.e. corrected).

      Note that nodes need not be the same type.

      @param[out] Links
      @param[out] source_nodes
      @param[out] sink_nodes

      OPTIMIZATION:
      pass memory to tensors so that when the tensors compute the matrices
      the underlying node values are automatically updated
    */ 
    void backPropogateLayerError(
      const std::vector<int>& links,
      const std::vector<int>& source_nodes,
      const std::vector<int>& sink_nodes);
 
    /**
      @brief Back propogation of the network model.
        All node errors are calculating starting from the output nodes.  
        Each node status is changed from "activated" to "corrected" when the
        outputs and derivatives are calculated.
    */ 
    void backPropogate();  
 
    /**
      @brief Truncated Back Propogation Through Time (TBPTT) of the network model.
        All node errors are calculating starting from the output nodes.  
        Each node status is changed from "activated" to "corrected" when the
        outputs and derivatives are calculated.

      @param[in] time_steps The number of time_steps backwards to 
        unfold the network model.
    */ 
    void TBPTT(const int& time_steps);  
 
    /**
      @brief Recurrent Real Time Learning (RTRL) of the network model.
        All node errors are calculating starting from the output nodes.  
        Each node status is changed from "activated" to "corrected" when the
        outputs and derivatives are calculated.

      @param[in] time_steps The number of time_steps backwards to 
        unfold the network model.
    */ 
    void RTRL(const int& time_steps);  
 
    /**
      @brief Update the weights
      
    */ 
    void updateWeights();
 
    /**
      @brief Reset the node statuses back to inactivated
      
    */ 
    void reInitializeNodeStatuses();

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setError(const Eigen::Tensor<float, 1>& error); ///< error setter
    Eigen::Tensor<float, 1> getError() const; ///< error getter

    void setLossFunction(const SmartPeak::ModelLossFunction& loss_function); ///< loss_function setter
    SmartPeak::ModelLossFunction getLossFunction() const; ///< loss_function getter
 
    /**
      @brief Add new links to the model.

      @param[in] links Links to add to the model
    */ 
    void addLinks(const std::vector<Link>& links);
    Link getLink(const int& link_id) const; ///< link getter
 
    /**
      @brief Remove existing links from the model.

      @param[in] Link_ids Links to remove from the model
    */ 
    void removeLinks(const std::vector<int>& link_ids);
 
    /**
      @brief Add new nodes to the model.

      @param[in] nodes Nodes to add to the model
    */ 
    void addNodes(const std::vector<Node>& nodes);
    Node getNode(const int& node_id) const; ///< node getter
    std::vector<int> getNodeIDs(const NodeStatus& node_status) const; ///< node getter (TODO)
    std::vector<int> getNodeIDs(const NodeType& node_type) const; ///< node getter (TODO)
 
    /**
      @brief Remove existing nodes from the model.

      @param[in] node_ids Nodes to remove from the model
    */ 
    void removeNodes(const std::vector<int>& node_ids);
 
    /**
      @brief Add new weights to the model.

      @param[in] weights Weights to add to the model
    */ 
    void addWeights(const std::vector<Weight>& weights);
    Weight getWeight(const int& weight_id) const; ///< weight getter
 
    /**
      @brief Remove existing weights from the model.

      @param[in] weight_ids Weights to remove from the model
    */ 
    void removeWeights(const std::vector<int>& weight_ids);
 
    /**
      @brief Removes nodes from the model that no longer
        have an associated link.
    */ 
    void pruneNodes();
 
    /**
      @brief Removes links from the model that no longer
        have associated nodes.
    */ 
    void pruneLinks();    
 
    /**
      @brief Removes weights from the model that no longer
        have associated links.
    */ 
    void pruneWeights(); 

private:
    int id_; ///< Model ID
    std::map<int, Link> links_; ///< Model links
    std::map<int, Node> nodes_; ///< Model nodes
    std::map<int, Weight> weights_; ///< Model nodes
    Eigen::Tensor<float, 1> error_; ///< Model error
    // Eigen::Tensor<float, 2> error_; ///< Model error
    SmartPeak::ModelLossFunction loss_function_; ///< Model loss function

  };
}

#endif //SMARTPEAK_MODEL_H