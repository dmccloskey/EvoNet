/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Weight.h>

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

    Assumptions about the model structure:
    1. Inputs can only be sources
    2. Outputs can only be sinks (will break back propogation algorithm)
  */
  class Model
  {
public:
    Model(); ///< Default constructor
    Model(const Model& other); ///< Copy constructor // [TODO: add test]
    Model(const int& id); ///< Explicit constructor  
    ~Model(); ///< Default destructor

    inline bool operator==(const Model& other) const
    {
      return
        std::tie(
          id_,
          name_,
          links_,
          nodes_,
          weights_
        ) == std::tie(
          other.id_,
          other.name_,
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

    inline Model& operator=(const Model& other)
    { // [TODO: add test]
      id_ = other.id_;
      name_ = other.name_;
      links_ = other.links_;
      nodes_ = other.nodes_;
      weights_ = other.weights_;
      error_ = other.error_;
      loss_function_ = other.loss_function_;
      return *this;
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
      @param[in] node_names 
      @param[in] status_update
      @param[in] value_type ("output", "derivative", "error", or "dt")
    */ 
    void mapValuesToNodes(
      const Eigen::Tensor<float, 3>& values,
      const std::vector<std::string>& node_names,
      const NodeStatus& status_update,
      const std::string& value_type);

    /**
      @brief Assigns output or error values to the nodes at a specific
        place in memory.
        The node statuses are then changed accordingly (i.e.,
        status_update of "activated" will update the output values
        of the node and status_update of "corrected" will update
        the error values of the node.

      dimensions of batch size by nodes

      @param[in] values Values to assign to the node
      @param[in] memory_step The memory step to add values to 
      @param[in] node_names 
      @param[in] status_update
      @param[in] value_type String of "output", "derivative", or "error"
    */ 
    void mapValuesToNodes(
      const Eigen::Tensor<float, 2>& values,
      const int& memory_step,
      const std::vector<std::string>& node_names,
      const NodeStatus& status_update,
      const std::string& value_type);

    /**
      @brief Assigns output or error values to a single node at a specific
        place in memory.

      dimensions of batch size

      @param[in] values Values to assign to the node
      @param[in] memory_step The memory step to add values to 
      @param[in] node_name
      @param[in] status_update
      @param[in] value_type String of "output", "derivative", or "error"
    */ 
    void mapValuesToNode(
      const Eigen::Tensor<float, 1>& values,
      const int& memory_step,
      const std::string& node_name,
      const NodeStatus& status_update,
      const std::string& value_type);

    /**
      @brief Assigns output or error values to all nodes at a specific
        place in memory.
        The node statuses are also updated according to status_update.

      dimensions of batch size by nodes

      @param[in] values Values to assign to the node
      @param[in] memory_step The memory step to add values to 
      @param[in] status_update
      @param[in] value_type String of "output", "derivative", or "error"
    */       
    void mapValuesToNodes(
      const Eigen::Tensor<float, 1>& values,
      const int& memory_step,
      const NodeStatus& status_update,
      const std::string& value_type);
 
    /**
      @brief A prelude to a forward propogation step. Returns a vector of links
        and associated nodes that satisfy the following conditions:
        1. all sink output values are unknown (i.e. inactive),
        2. all source node output values are known (i.e. active).
        3. all nodes need not be the same type
      
      [TODO: check that only sink nodes where ALL sources are active are identified]

      @param[out] Links
      @param[out] source_nodes
      @param[out] sink_nodes
    */ 
    void getNextInactiveLayer(
      std::vector<std::string>& links,
      std::vector<std::string>& source_nodes,
      std::vector<std::string>& sink_nodes);
    void getNextInactiveLayer(
      std::map<std::string, std::vector<std::string>>& sink_links_map);
 
    /**
      @brief Continuation of the forward propogation step that identifies all biases
        for the identified sink nodes. Returns a vector of links
        and associated nodes that satisfy the following conditions:
        1. all sink output values are unknown (i.e. inactive),
        2. all source node output values are known (i.e. active) and biases.

      @param[out] Links
      @param[out] source_nodes
      @param[in] sink_nodes

      @param[in, out] sink_links_map Map of sink nodes (keys) to a vector of links (values)
      @param[out] sink_nodes_with_biases
    */ 
    void getNextInactiveLayerBiases(
      std::vector<std::string>& links,
      std::vector<std::string>& source_nodes,
      const std::vector<std::string>& sink_nodes,
      std::vector<std::string>& sink_nodes_with_biases);
    void getNextInactiveLayerBiases(
      std::map<std::string, std::vector<std::string>>& sink_links_map,
      std::vector<std::string>& sink_nodes_with_biases
      );
 
    /**
      @brief Continuation of the forward propogation step that identifies 
        all cyclic source nodes for the identified sink nodes. Returns a vector of links
        and associated nodes that satisfy the following conditions:
        1. all sink output values are unknown (i.e. inactive),
        2. all source node output values are unknown (i.e. inactive).

      @param[out] Links
      @param[out] source_nodes
      @param[in] sink_nodes

      @param[in, out] sink_links_map Map of sink nodes (keys) to a vector of links (values)
      @param[out] sink_nodes_with_cycles
    */ 
    void getNextInactiveLayerCycles(
      std::vector<std::string>& links,
      std::vector<std::string>& source_nodes,
      const std::vector<std::string>& sink_nodes,
      std::vector<std::string>& sink_nodes_with_cycles);
    void getNextInactiveLayerCycles(
      std::map<std::string, std::vector<std::string>>& sink_links_map,
      std::vector<std::string>& sink_nodes_with_cycles);

    /**
    @brief Allocate tensor memory for all forward
      propogation tensors.

    Note that nodes need not be the same type.

    @param[in] time_step Time step to activate.
    */
    void allocateForwardPropogationLayerTensors(const int& time_step);
 
    /**
    @brief A prelude to a forward propogation step. Computes the net
      input into all nodes composing the next layer:
      1. all sink output values are unknown (i.e. inactive),
      2. all source node output values are known (i.e. active).

    Note that nodes need not be the same type.

    @param[out] Links
    @param[out] source_nodes
    @param[out] sink_nodes
    @param[in] time_step Time step to activate.

    [OPTIMIZATION:
      pass memory to tensors so that when the tensors compute the matrices
      the underlying node values are automatically updated?]
    [PARALLEL: allow for parallelization of iteration of sink nodes]
    [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
    */ 
    void forwardPropogateLayerNetInput(
      const std::vector<std::string>& links,
      const std::vector<std::string>& source_nodes,
      const std::vector<std::string>& sink_nodes,
      const int& time_step);
    void forwardPropogateLayerNetInput(
      std::map<std::string, std::vector<std::string>>& sink_links_map,
      const int& time_step);
 
    /**
    @brief Completion of a forward propogation step. Computes the net
      activation for all nodes in the tensor layer.

    [DEPRECATED]

    Note before computing the activation, the layer tensor will be split
      according to the node type, and the corresponding activation
      function will be applied

    @param[in] sink_nodes
    @param[in] time_step Time step to activate.
    */ 
    void forwardPropogateLayerActivation(
      const std::vector<std::string>& sink_nodes,
      const int& time_step);
 
    /**
    @brief Foward propogation of the network model.
      All node outputs and derivatives are calculating
      starting from the input nodes.  Each node status is
      changed from "initialized" to "activated" when the
      outputs and derivatives are calculated.
    
    [TODO: add tests for caching]

    @param[in] time_step Time step to forward propogate.
    @param[in] cache_FP_steps Whether to save the FP steps
      for faster iteration next epoch.
    @param[in] use_cache Whether to use the cached FP steps.
    */ 
    void forwardPropogate(const int& time_step, bool cache_FP_steps = false, bool use_cache = false);     
 
    /**
    @brief Foward propogation through time (FPTT) of the network model.
      All node outputs and derivatives are calculating
      starting from the input nodes.  Each node status is
      changed from "initialized" to "activated" when the
      outputs and derivatives are calculated.  This is repeated
      for n_time steps without weight updates.
    
    NOTE: The implementation assumes that the output values for
      all input and biases have already been set.  

    @param[in] time_steps The number of time_steps forward to 
      continuously calculate node outputs and node derivatives.
    @param[in] values Input values at each time step where
      dim0: batch_size, dim1: time_step, and dim2: nodes.
    @param[in] node_names
    @param[in] dt Node time resolution 
    */ 
    void FPTT(const int& time_steps, 
      const Eigen::Tensor<float, 3>& values,
      const std::vector<std::string> node_names,
      const Eigen::Tensor<float, 2>& dt);
 
    /**
    @brief Calculates the error of the model with respect to
      the expected values

    @param[in] values Expected node output values
    @param[in] node_names Output nodes
    */ 
    void calculateError(const Eigen::Tensor<float, 2>& values, const std::vector<std::string>& node_names);
 
    /**
    @brief Calculates the error of the model through time (CETT)
      with respect to the expected values

    @param[in] values Expected node output values
    @param[in] node_names Output nodes
    */ 
    void CETT(const Eigen::Tensor<float, 3>& values, const std::vector<std::string>& node_names);
 
    /**
    @brief A prelude to a back propogation step.  Returns a vector of links
      and associated nodes that satisfy the following conditions:
      1. all sink error values are unknown (i.e. active),
      2. all source error values are known (i.e. corrected).
      3. all nodes need not be the same type
    
    [TODO: check that only sink nodes where ALL sources are corrected are identified]

    @param[out] Links
    @param[out] source_nodes
    @param[out] sink_nodes
    */ 
    void getNextUncorrectedLayer(
      std::vector<std::string>& links,
      std::vector<std::string>& source_nodes,
      std::vector<std::string>& sink_nodes);
    void getNextUncorrectedLayer(
      std::map<std::string, std::vector<std::string>>& sink_links_map,
      std::vector<std::string>& source_nodes);      
 
    /**
    @brief A continuation of a back propogation step.  Returns a vector of links
      and associated nodes that satisfy the following conditions:
      1. all sink error values are known (i.e. corrected),
      2. all source error values are known (i.e. corrected).
      3. all nodes need not be the same type

    @param[out] Links
    @param[out] source_nodes
    @param[out] sink_nodes
    @param[out] source_nodes_with_cycles
    */ 
    void getNextUncorrectedLayerCycles(
      std::vector<std::string>& links,
      const std::vector<std::string>& source_nodes,
      std::vector<std::string>& sink_nodes,
      std::vector<std::string>& source_nodes_with_cycles);
    void getNextUncorrectedLayerCycles(
      std::map<std::string, std::vector<std::string>>& sink_links_map,
      const std::vector<std::string>& source_nodes,
      std::vector<std::string>& source_nodes_with_cycles); 
 
    /**
    @brief A back propogation step. Computes the net
      error into all nodes composing the next layer:
      1. all sink error values are unknown (i.e. active),
      2. all source error values are known (i.e. corrected).

    Note that nodes need not be the same type.

    @param[out] Links
    @param[out] source_nodes
    @param[out] sink_nodes
    @param[in] time_step Time step to forward propogate.

    [OPTIMIZATION:
    pass memory to tensors so that when the tensors compute the matrices
    the underlying node values are automatically updated]

    [PARALLEL: allow for parallelization of iteration of sink nodes]
    */ 
    void backPropogateLayerError(
      const std::vector<std::string>& links,
      const std::vector<std::string>& source_nodes,
      const std::vector<std::string>& sink_nodes,
      const int& time_step);
    void backPropogateLayerError(
      const std::map<std::string, std::vector<std::string>>& sink_links_map,
      const int& time_step);
 
    /**
    @brief Back propogation of the network model.
      All node errors are calculating starting from the output nodes.  
      Each node status is changed from "activated" to "corrected" when the
      outputs and derivatives are calculated.

    [TODO: add tests for caching]

    @param[in] time_step Time step to forward propogate.
    @param[in] cache_BP_steps Whether to save the BP steps
      for faster iteration next epoch.
    @param[in] use_cache Whether to use the cached BP steps.

    @returns Vector of cyclic sink node IDs
    */ 
    std::vector<std::string> backPropogate(const int& time_step, bool cache_BP_steps = false, bool use_cache = false);  
 
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
    void updateWeights(const int& time_steps);
 
    /**
    @brief Reset the node statuses back to inactivated
      
    */ 
    void reInitializeNodeStatuses();

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< name setter
    std::string getName() const; ///< name getter

    void setError(const Eigen::Tensor<float, 1>& error); ///< error setter
    Eigen::Tensor<float, 1> getError() const; ///< error getter

    void setLossFunction(const SmartPeak::ModelLossFunction& loss_function); ///< loss_function setter
    SmartPeak::ModelLossFunction getLossFunction() const; ///< loss_function getter
 
    /**
      @brief Add new links to the model.

      @param[in] links Links to add to the model
    */ 
    void addLinks(const std::vector<Link>& links);
    Link getLink(const std::string& link_name) const; ///< link getter
    std::vector<Link> getLinks() const;  ///< links getter
 
    /**
      @brief Remove existing links from the model.

      @param[in] Link_names Links to remove from the model
    */ 
    void removeLinks(const std::vector<std::string>& link_names);
 
    /**
      @brief Add new nodes to the model.

      @param[in] nodes Nodes to add to the model
    */ 
    void addNodes(const std::vector<Node>& nodes);
    Node getNode(const std::string& node_name) const; ///< node getter
    std::vector<Node> getNodes() const; ///< nodes getter
    std::vector<std::string> getNodeIDs(const NodeStatus& node_status) const; ///< node getter (TODO)
    std::vector<std::string> getNodeIDs(const NodeType& node_type) const; ///< node getter (TODO)
 
    /**
      @brief Remove existing nodes from the model.

      @param[in] node_names Nodes to remove from the model
    */ 
    void removeNodes(const std::vector<std::string>& node_names);
 
    /**
      @brief Add new weights to the model.

      @param[in] weights Weights to add to the model
    */ 
    void addWeights(const std::vector<Weight>& weights);
    Weight getWeight(const std::string& weight_name) const; ///< weight getter
    std::vector<Weight> getWeights() const;  ///< weights getter
 
    /**
      @brief Remove existing weights from the model.

      @param[in] weight_names Weights to remove from the model
    */ 
    void removeWeights(const std::vector<std::string>& weight_names);
 
    /**
      @brief Removes nodes from the model that no longer
        have an associated link.

      @returns True if nodes were removed, False otherwise
    */ 
    bool pruneNodes();
 
    /**
      @brief Removes links from the model that no longer
        have associated nodes.

      @returns True if links were removed, False otherwise
    */ 
    bool pruneLinks();    
 
    /**
      @brief Removes weights from the model that no longer
        have associated links.

      @returns True if weights were removed, False otherwise
    */ 
    bool pruneWeights(); 
 
    /**
      @brief Removes dangling links, weights, and nodes 
        recursively until there are no more dangling
        model components or the number of user specified
        iterations has been reached.

      @param[in] iterations The number of recursive iterations to prune
    */ 
    void pruneModel(int iterations = 1e3); 
 
    /**
      @brief Check to ensure that the nodes are in the model

      @param[in] node_names 
    */ 
    bool checkNodeNames(const std::vector<std::string> node_names); 
 
    /**
      @brief Check to ensure that the links are in the model

      @param[in] link_names 
    */ 
    bool checkLinkNames(const std::vector<std::string> link_names); 
 
    /**
      @brief Check to ensure that the weights are in the model

      @param[in] weight_names 
    */ 
    bool checkWeightNames(const std::vector<std::string> weight_names);

    void clearCache(); ///< clear the FP and BP caches

private:
    int id_; ///< Model ID
    std::string name_; ///< Model Name
    std::map<std::string, Link> links_; ///< Model links
    std::map<std::string, Node> nodes_; ///< Model nodes
    std::map<std::string, Weight> weights_; ///< Model nodes
    Eigen::Tensor<float, 1> error_; ///< Model error
    // Eigen::Tensor<float, 2> error_; ///< Model error

    // TODO: will most likely need to expand to a derived class model (e.g., SolverOp)
    SmartPeak::ModelLossFunction loss_function_; ///< Model loss function

    // Internal structures to allow for caching of the different FP and BP layers
    std::vector<std::map<std::string, std::vector<std::string>>> FP_sink_link_cache_; 
    std::vector<std::map<std::string, std::vector<std::string>>> BP_sink_link_cache_;
    std::vector<std::string> BP_cyclic_nodes_cache_;

    // Internal structures to allow for efficient multi-threading
    // and off-loading of computation from host to devices


  };
}

#endif //SMARTPEAK_MODEL_H