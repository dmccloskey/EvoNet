/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

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
          error_
        ) == std::tie(
          other.id_,
          other.links_,
          other.nodes_,
          other.error_
        )
      ;
    }

    inline bool operator!=(const Model& other) const
    {
      return !(*this == other);
    }

    /**
      @brief Initialize all link weights

      input and node Link weights will be initialized using the method of He, et al 2015
      bias Link weight will be initialized as a constant

      TODO: make a new class called Weights.  Replace weight with weight_id in Link.  
        change initLinks() to initWeights()
    */ 
    void initLinks() const;  //TODO

    /**
      @brief Initialize all node output to zero.
        The node statuses are then changed to NodeStatus::deactivated

      @param[in] batch_size Size of the output, error, and derivative node vectors
    */ 
    void initNodes(const int& batch_size);

    /**
      @brief Assigns output values to the input nodes.
        The node statuses are then changed to NodeStatus::activated

      dimensions of batch size by nodes

      @param[in] input
    */ 
    void mapValuesToNodes(const Eigen::Tensor<float, 2>& values, const std::vector<int>& node_ids);
 
    /**
      @brief A prelude to a forward propogation step. Returns a vector of links
        and associated nodes that satisfy the following conditions:
        1. all sink output values are unknown (i.e. inactive),
        2. all source node output values are known (i.e. active).
        3. all nodes need not be the same type

      @param[out] Links
      @param[out] source_nodes
      @param[out] sink_nodes

      @returns layer vector of links
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

      @returns layer vector of links

      OPTIMIZATION:
      pass memory to tensors so that when the tensors compute the matrices
      the underlying node values are automatically updated
    */ 
    void forwardPropogateLayerNetInput(
      const std::vector<int>& links,
      const std::vector<int>& source_nodes,
      const std::vector<int>& sink_nodes);
 
    /**
      @brief Completion of the forward propogation step. Computes the net
        activation for all nodes in the tensor layer.

      Note before computing the activation, the layer tensor will be split
        according to the node type, and the corresponding activation
        function will be applied

      @param[in] sink_nodes

      @returns layer vector of links
    */ 
    void forwardPropogateLayerActivation(
      const std::vector<int>& sink_nodes);
 
    /**
      @brief Calculate the error of the model with respect to
        expected values

      @param[in] values Expected node output values
      @param[in] node_ids Output nodes
    */ 
    void calculateError(const Eigen::Tensor<float, 2>& values, const std::vector<int>& node_ids);
 
    /**
      @brief A back propogation step.  Returns a vector of links where
        all sink error values are unknown (i.e. active),
        but all source node error values are known (i.e. inactive).

      If multiple vectors of links satisfy the above
        criteria, only the first vector of links will
        be returned.  All others will be returned
        on subsequent calls.

      @param[in] x_I Input value

      @returns layer vector of links
    */ 
    void getNextUncorrectedLayer() const;

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setError(const double& error); ///< error setter
    double getError() const; ///< error getter

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
      @brief Removes nodes from the model that no longer
        have an associated link.
    */ 
    void pruneNodes();
 
    /**
      @brief Removes links from the model that no longer
        have associated nodes.
    */ 
    void pruneLinks();    

private:
    int id_; ///< Model ID
    std::map<int, Link> links_; ///< Model links
    std::map<int, Node> nodes_; ///< Model nodes
    double error_; ///< Model error
    SmartPeak::ModelLossFunction loss_function_; ///< Model loss function

  };
}

#endif //SMARTPEAK_MODEL_H