/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELREPLICATOR_H
#define SMARTPEAK_MODELREPLICATOR_H

#include <SmartPeak/ml/Model.h>

#include <vector>
#include <string>

namespace SmartPeak
{

  /**
    @brief Replicates a model with or without modification (i.e., mutation)
  */
  class ModelReplicator
  {
public:
    ModelReplicator(); ///< Default constructor
    ~ModelReplicator(); ///< Default destructor

    void setBatchSize(const int& n_node_additions); ///< n_node_additions setter
    void setMemorySize(const int& n_link_additions); ///< n_link_additions setter
    void setNEpochs(const int& n_node_deletions); ///< n_node_deletions setter
    void setNLinkDeletions(const int& n_link_deletions); ///< n_link_deletions setter
    void setNWeightChanges(const int& n_weight_changes); ///< n_weight_changes setter
    void setWeightChangeStDev(const float& weight_change_stdev); ///< weight_change_stdev setter

    int getBatchSize() const; ///< n_node_additions setter
    int getMemorySize() const; ///< n_link_additions setter
    int getNEpochs() const; ///< n_node_deletions setter
    int getNLinkDeletions() const; ///< n_link_deletions setter
    int getNWeightChanges() const; ///< n_weight_changes setter
    float getWeightChangeStDev() const; ///< weight_change_stdev setter
 
    /**
      @brief Make a new baseline model where all layers are fully connected

      @param n_input_nodes The number of input nodes the model should have
      @param n_hidden_nodes The number of hidden nodes the model should have
      @param n_output_nodes The number of output nodes the model should have
      @param hidden_node_type The type of hidden node to create
      @param output_node_type The type of output node to create
      @param [TODO: docstrings for new params]

      @returns A baseline model
    */ 
    Model makeBaselineModel(const int& n_input_nodes, const int& n_hidden_nodes, const int& n_output_nodes,
      const NodeType& hidden_node_type, const NodeType& output_node_type,
      const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver);
 
    /**
      @brief Modify (i.e., mutate) as existing model

      @param model The model to modify
    */ 
    void modifyModel(Model& model);
 
    /**
      @brief Modify (i.e., mutate) as existing model

      @param model The model to modify
      @param with_modifications Should modifications be done to the replicated model?

      @returns A replicated model with or without modifications
    */ 
    Model replicateModel(const Model& model, const bool& with_modifications);
 
    /**
      @brief Select random node given the following conditions:
        1. the Node is not of NodeType input nor bias
        2. more distant nodes (as based on the number of links seperating them)
          have a lower probability of being selected
        3. directionality of the node

      @param node Previous node selected (for distance calculation)
      @param node_type_exclude Node types to exclude
      @param node_type_include Node types to include
      @param direction Source to Sink node direction; options are "forward, reverse"

      @returns A node
    */ 
    Node selectNodeRandom(const Node& node, 
      const std::vector<NodeType>& node_type_exclude,
      const std::vector<NodeType>& node_type_include,
      const std::string& direction);

    void addNode(Model& model); ///< add node to the model
    void addLink(Model& model); ///< add link to the model
    void deleteNode(Model& model); ///< delete node to the model
    void deleteLink(Model& model); ///< delete link to the model
    void modifyWeight(Model& model); ///< modify weights in the model

private:
    // modification parameters
    int n_node_additions_; ///< new nodes to add to the model (nodes are created through replication)
    int n_link_additions_; ///< new links to add to the model
    int n_node_deletions_; ///< nodes to remove from the model
    int n_link_deletions_; ///< links to remove from the model
    int n_weight_changes_; ///< the number of weights to change in the model
    float weight_change_stdev_; ///< the standard deviation to change the weights in the model
  };
}

#endif //SMARTPEAK_MODELREPLICATOR_H