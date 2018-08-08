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

    void setNNodeAdditions(const int& n_node_additions); ///< n_node_additions setter
    void setNLinkAdditions(const int& n_link_additions); ///< n_link_additions setter
    void setNNodeDeletions(const int& n_node_deletions); ///< n_node_deletions setter
    void setNLinkDeletions(const int& n_link_deletions); ///< n_link_deletions setter
		void setNNodeActivationChanges(const int& n_node_activation_changes); ///< n_node_activation_changes setter
		void setNNodeIntegrationChanges(const int& n_node_integration_changes); ///< n_node_integration_changes setter
		void setNodeActivations(const std::vector<NodeActivation>& node_activations); ///< node_activations setter
		void setNodeIntegrations(const std::vector<NodeIntegration>& node_integrations); ///< node_integrations setter
		void setNModuleAdditions(const int& n_module_additions); ///< n_module_additions setter
		void setNModuleDeletions(const int& n_module_deletions); ///< n_module_deletions setter

    int getNNodeAdditions() const; ///< n_node_additions setter
    int getNLinkAdditions() const; ///< n_link_additions setter
    int getNNodeDeletions() const; ///< n_node_deletions setter
    int getNLinkDeletions() const; ///< n_link_deletions setter
		int getNNodeActivationChanges() const; ///< n_node_activation_changes setter
		int getNNodeIntegrationChanges() const; ///< n_node_integration_changes setter
		std::vector<NodeActivation> getNodeActivations() const; ///< node_activations setter
		std::vector<NodeIntegration> getNodeIntegrations() const; ///< node_integrations setter
		int getNModuleAdditions() const; ///< n_module_additions setter
		int getNModuleDeletions() const; ///< n_module_deletions setter

		void setNNodeCopies(const int& n_node_copies); ///< n_node_copies setter
		void setNWeightChanges(const int& n_weight_changes); ///< n_weight_changes setter
		void setWeightChangeStDev(const float& weight_change_stdev); ///< weight_change_stdev setter

		int getNNodeCopies() const; ///< n_node_copies setter
		int getNWeightChanges() const; ///< n_weight_changes setter
    float getWeightChangeStDev() const; ///< weight_change_stdev setter
 
    /**
      @brief Make a new baseline model where all layers are fully connected

      @param n_input_nodes The number of input nodes the model should have
      @param n_hidden_nodes The number of hidden nodes the model should have
      @param n_output_nodes The number of output nodes the model should have
      @param hidden_node_activation The activation function of the hidden node to create
      @param hidden_node_integration The integration function of the hidden node to create
      @param output_node_activation The activation function of the output node to create
      @param output_node_integration The integration function of the output node to create
      @param weight_init Weight init operator to use for hidden and output nodes
      @param solver Solver operator to use for hidden and output nodes
      @param error_function Model loss function
      @param unique_str Optional string to make the model name unique

      @returns A baseline model
    */ 
    Model makeBaselineModel(const int& n_input_nodes, const int& n_hidden_nodes, const int& n_output_nodes,
			const NodeActivation& hidden_node_activation, const NodeIntegration& hidden_node_integration,
			const NodeActivation& output_node_activation, const NodeIntegration& output_node_integration,
      const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
      const ModelLossFunction& error_function, std::string unique_str = "");
 
    /**
      @brief Modify (i.e., mutate) an existing model in place

      @param[in, out] model The model to modify
    */ 
    void modifyModel(Model& model, std::string unique_str = "");
 
    /**
      @brief Select nodes given a set of conditions

      @param[in, out] model The model
      @param node_type_exclude Node types to exclude
      @param node_type_include Node types to include

      @returns A node name
    */ 
    std::vector<std::string> selectNodes(
      const Model& model,
      const std::vector<NodeType>& node_type_exclude,
      const std::vector<NodeType>& node_type_include);
 
    /**
      @brief Select random node given a set of conditions

      @param[in, out] model The model
      @param node_type_exclude Node types to exclude
      @param node_type_include Node types to include
      @param node Previous node selected (for distance calculation)
      @param distance_weight Probability weighting to punish more "distant" nodes
      @param direction Source to Sink node direction; options are "forward, reverse"

      @returns A node name
    */ 
    std::string selectRandomNode(
      const Model& model,
      const std::vector<NodeType>& node_type_exclude,
      const std::vector<NodeType>& node_type_include,
      const Node& node, 
      const float& distance_weight,
      const std::string& direction);
    std::string selectRandomNode(
      const Model& model,
      const std::vector<NodeType>& node_type_exclude,
      const std::vector<NodeType>& node_type_include);
 
    /**
      @brief Select random link given a set of conditions

      @param[in, out] model The model
      @param source_node_type_exclude Source node types to exclude
      @param source_node_type_include Source node types to include
      @param sink_node_type_exclude Sink node types to exclude
      @param sink_node_type_include Sink node types to include
      @param direction Source to Sink node direction; options are "forward, reverse"

      @returns A link name
    */ 
    std::string selectRandomLink(
      const Model& model,
      const std::vector<NodeType>& source_node_type_exclude,
      const std::vector<NodeType>& source_node_type_include,
      const std::vector<NodeType>& sink_node_type_exclude,
      const std::vector<NodeType>& sink_node_type_include,
      const std::string& direction);
    std::string selectRandomLink(
      const Model& model,
      const std::vector<NodeType>& source_node_type_exclude,
      const std::vector<NodeType>& source_node_type_include,
      const std::vector<NodeType>& sink_node_type_exclude,
      const std::vector<NodeType>& sink_node_type_include);

    // Model modification operators

    /**
      @brief Copy a node in the model (Layer expansion to the left or right)

      @param[in, out] model The model
    */ 
    void copyNode(Model& model);

    /**
      @brief Add node to the model (Layer injection up or down).
        The method utilizes a modified version of the NEAT algorithm whereby a random
        link is chosen and bifurcated with a new node.  Instead, new nodes are added
        using the following procedure:
        1. an existing node is randomly chosen from the model.
        2. a randomly connected input link to the node is chosen.
          Note that an input link is chose because it is easier
          to exclude input nodes than output nodes.
        3. the chosen node is copied and a new link is added
          between the new node and the existing node.
        4. the new link becomes the input link of the existing node and the output link of the new node, 
          and existing link becomes the input link of the new node.

      References:
        Kenneth O. Stanley & Risto Miikkulainen (2002). "Evolving Neural Networks Through Augmenting Topologies". 
        Evolutionary Computation. 10 (2): 99–127. doi:10.1162/106365602320169811

      @param[in, out] model The model
    */ 
    void addNode(Model& model, std::string unique_str = "");

    /**
      @brief add link to the model

      @param[in, out] model The model
    */ 
    void addLink(Model& model, std::string unique_str = "");

		/**
		@brief Add a new module to the model

		@param[in, out] model The model
		*/
		void addModule(Model& model, std::string unique_str = "");

    /**
      @brief delete node to the model

      @param[in, out] model The model
      @param[in] prune_iterations The number of model recursive prune iterations
    */ 
    void deleteNode(Model& model, int prune_iterations = 1e6);

    /**
      @brief delete link to the model

      @param[in, out] model The model
      @param[in] prune_iterations The number of model recursive prune iterations
    */ 
    void deleteLink(Model& model, int prune_iterations = 1e6);

		/**
		@brief delete module in the model

		@param[in, out] model The model
		@param[in] prune_iterations The number of model recursive prune iterations
		*/
		void deleteModule(Model& model, int prune_iterations = 1e6);

		/**
		@brief change node activation

		@param[in, out] model The model
		*/
		void changeNodeActivation(Model& model, std::string unique_str = "");

		/**
		@brief change node integration

		@param[in, out] model The model
		*/
		void changeNodeIntegration(Model& model, std::string unique_str = "");

    /**
      @brief modify weights in the model

      @param[in, out] model The model
    */ 
    void modifyWeight(Model& model);

    /**
      @brief Make a unique time stampped hash of the form
        left_str + right_str + timestamp

      @param[in] left_str
      @param[in] right_str

      @returns A unique string hash
    */ 
    std::string makeUniqueHash(const std::string& left_str, const std::string& right_str);    

    /**
      @brief randomly order the mutations

      @returns A random list of mutations types
    */ 
    std::vector<std::string> makeRandomModificationOrder();

		/**
		@brief set random model modification parameters

		@param[in] node_additions lower/upper bound for the number of potential node additions
		@param[in] link_additions lower/upper bound for the number of potential link additions
		@param[in] node_deletions lower/upper bound for the number of potential node deletions
		@param[in] link_deletions lower/upper bound for the number of potential link deletions
		*/
		void setRandomModifications(
			const std::pair<int, int>& node_additions,
			const std::pair<int, int>& link_additions,
			const std::pair<int, int>& node_deletions,
			const std::pair<int, int>& link_deletions,
			const std::pair<int, int>& node_activation_changes,
			const std::pair<int, int>& node_integration_changes,
			const std::pair<int, int>& module_additions,
			const std::pair<int, int>& module_deletions);

		/**
		@brief make random model modification parameters
		*/
		void makeRandomModifications();

		/**
		@brief Entry point for users to code their adaptive scheduler
		to modify model modification parameters based on a given trigger

		@param[in] n_generations The number of evolution generations
		@param[in] models The models in the population
		@param[in] model_errors The trace of models errors from validation at each generation
		*/
		virtual void adaptiveReplicatorScheduler(
			const int& n_generations,
			std::vector<Model>& models,
			std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations) = 0;

private:
    // modification parameters
    int n_node_additions_ = 0; ///< new nodes to add to the model (nodes are created through replication)
    int n_link_additions_ = 0; ///< new links to add to the model
    int n_node_deletions_ = 0; ///< nodes to remove from the model
    int n_link_deletions_ = 0; ///< links to remove from the model
		int n_node_activation_changes_ = 0; ///< nodes to change the activation
		int n_node_integration_changes_ = 0; ///< nodes to change the activation
		int n_module_additions_ = 0; ///< new modules added to the model (modules are created through replication)
		int n_module_deletions_ = 0; ///< modules to remove from the model

		// random modification parameters
		std::pair<int, int> node_additions_ = std::make_pair(0, 0);
		std::pair<int, int> link_additions_ = std::make_pair(0, 0);
		std::pair<int, int> node_deletions_ = std::make_pair(0, 0);
		std::pair<int, int> link_deletions_ = std::make_pair(0, 0);
		std::pair<int, int> node_activation_changes_ = std::make_pair(0, 0);
		std::pair<int, int> node_integration_changes_ = std::make_pair(0, 0);
		std::pair<int, int> module_additions_ = std::make_pair(0, 0);
		std::pair<int, int> module_deletions_ = std::make_pair(0, 0);
		std::vector<NodeActivation> node_activations_;
		std::vector<NodeIntegration> node_integrations_;
		

		// not yet implemented...
		int n_node_copies_ = 0; ///< nodes to duplicate in the model (with a random source and sink connection) [TODO: names should be swapped at some point]
    int n_weight_changes_ = 0; ///< the number of weights to change in the model
    float weight_change_stdev_ = 0; ///< the standard deviation to change the weights in the model
  };
}

#endif //SMARTPEAK_MODELREPLICATOR_H