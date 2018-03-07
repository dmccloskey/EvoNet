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
    */ 
    void initLinks() const;  //TODO

    /**
      @brief Initialize all node output to zero.
        The node statuses are then changed to NodeStatus::deactivated

      @param[in] batch_size Size of the output, error, and derivative node vectors
    */ 
    void initNodes(const int& batch_size);  //TODO

    /**
      @brief Assigns output values to the specified nodes.
        The node statuses are then changed to NodeStatus::activated

      dimensions of Node x batch size

      @param[in]
    */ 
    void setNodeOutput(const Eigen::Tensor<float, 1>& values, const std::vector<int>& node_ids) const;  //TODO

    /**
      @brief Assigns error values to the specified nodes.
        The node statuses are then changed to NodeStatus::corrected

      dimensions of Node x batch size

      @param[in]
    */ 
    void setNodeError(const Eigen::Tensor<float, 1>& values, const std::vector<int>& node_ids) const;  //TODO
 
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
      std::vector<Link>& links,
      std::vector<Node>& source_nodes,
      std::vector<Node>& sink_nodes) const;
 
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
    */ 
    void forwardPropogateLayerNetInput(
      std::vector<Link>& links,
      std::vector<Node>& source_nodes,
      std::vector<Node>& sink_nodes) const;
 
    /**
      @brief Completion of the forward propogation step. Computes the net
        activation for all nodes in the tensor layer.

      Note before computing the activation, the layer tensor will be split
        according to the node type, and the corresponding activation
        function will be applied

      @param[out] Links
      @param[out] source_nodes
      @param[out] sink_nodes

      @returns layer vector of links
    */ 
    void forwardPropogateLayerActivation(
      std::vector<Link>& links,
      std::vector<Node>& source_nodes,
      std::vector<Node>& sink_nodes) const;
 
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

  };
}

#endif //SMARTPEAK_MODEL_H