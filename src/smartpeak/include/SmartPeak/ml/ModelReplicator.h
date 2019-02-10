/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELREPLICATOR_H
#define SMARTPEAK_MODELREPLICATOR_H

// .h
#include <SmartPeak/ml/Model.h>
#include <vector>
#include <string>

// .cpp
#include <SmartPeak/core/Preprocessing.h>
#include <random> // random number generator
#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <ctime> // time format
#include <chrono> // current time
#include <set>

namespace SmartPeak
{

  /**
    @brief Replicates a model with or without modification (i.e., mutation)
  */
	template<typename TensorT>
  class ModelReplicator
  {
public:
    ModelReplicator() = default; ///< Default constructor
    ~ModelReplicator() = default; ///< Default destructor

    void setNNodeDownAdditions(const int& n_node_additions); ///< n_node_additions setter
    void setNLinkAdditions(const int& n_link_additions); ///< n_link_additions setter
    void setNNodeDeletions(const int& n_node_deletions); ///< n_node_deletions setter
    void setNLinkDeletions(const int& n_link_deletions); ///< n_link_deletions setter
		void setNNodeActivationChanges(const int& n_node_activation_changes); ///< n_node_activation_changes setter
		void setNNodeIntegrationChanges(const int& n_node_integration_changes); ///< n_node_integration_changes setter
		void setNodeActivations(const std::vector<std::pair<std::shared_ptr<ActivationOp<TensorT>>, std::shared_ptr<ActivationOp<TensorT>>>>& node_activations); ///< node_activations setter
		void setNodeIntegrations(const std::vector<std::tuple<std::shared_ptr<IntegrationOp<TensorT>>, std::shared_ptr<IntegrationErrorOp<TensorT>>, std::shared_ptr<IntegrationWeightGradOp<TensorT>>>>& node_integrations); ///< node_integrations setter
		void setNModuleAdditions(const int& n_module_additions); ///< n_module_additions setter
		void setNModuleDeletions(const int& n_module_deletions); ///< n_module_deletions setter

    int getNNodeDownAdditions() const; ///< n_node_additions setter
    int getNLinkAdditions() const; ///< n_link_additions setter
    int getNNodeDeletions() const; ///< n_node_deletions setter
    int getNLinkDeletions() const; ///< n_link_deletions setter
		int getNNodeActivationChanges() const; ///< n_node_activation_changes setter
		int getNNodeIntegrationChanges() const; ///< n_node_integration_changes setter
		std::vector<std::pair<std::shared_ptr<ActivationOp<TensorT>>, std::shared_ptr<ActivationOp<TensorT>>>> getNodeActivations() const; ///< node_activations setter
		std::vector<std::tuple<std::shared_ptr<IntegrationOp<TensorT>>, std::shared_ptr<IntegrationErrorOp<TensorT>>, std::shared_ptr<IntegrationWeightGradOp<TensorT>>>> getNodeIntegrations() const; ///< node_integrations setter
		int getNModuleAdditions() const; ///< n_module_additions setter
		int getNModuleDeletions() const; ///< n_module_deletions setter

		void setNNodeRightAdditions(const int& n_node_additions);  ///< n_nodes_additions setter
		void setNNodeDownCopies(const int& n_node_copies); ///< n_node_copies setter
		void setNNodeRightCopies(const int& n_node_copies);  ///< n_node_copies setter
		void setNLinkCopies(const int& n_link_copies); ///< n_link_copies setter
		void setNModuleCopies(const int& n_module_copies); ///< n_module_copies setter
		void setNWeightChanges(const int& n_weight_changes); ///< n_weight_changes setter
		void setWeightChangeStDev(const TensorT& weight_change_stdev); ///< weight_change_stdev setter

		int getNNodeRightAdditions() const;  ///< n_node_additions getter
		int getNNodeDownCopies() const; ///< n_node_copies getter
		int getNNodeRightCopies() const; ///< n_node_copies getter
		int getNLinkCopies() const; ///< n_link_copies setter
		int getNModuleCopies() const; ///< n_module_copies setter
		int getNWeightChanges() const; ///< n_weight_changes getter
    TensorT getWeightChangeStDev() const; ///< weight_change_stdev getter
 
    /**
      @brief Modify (i.e., mutate) an existing model in place

      @param[in, out] model The model to modify
    */ 
    void modifyModel(Model<TensorT>& model, std::string unique_str = "");
 
    /**
      @brief Select nodes given a set of conditions

      @param[in, out] model The model
      @param node_type_exclude Node types to exclude
      @param node_type_include Node types to include

      @returns A node name
    */ 
    std::vector<std::string> selectNodes(
      const Model<TensorT>& model,
      const std::vector<NodeType>& node_type_exclude,
      const std::vector<NodeType>& node_type_include);

		/**
		@brief Select modules given a set of conditions

		@param[in, out] model The model
		@param node_type_exclude Node types to exclude
		@param node_type_include Node types to include

		@returns A node name
		*/
		std::vector<std::string> selectModules(
			const Model<TensorT>& model,
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
      const Model<TensorT>& model,
      const std::vector<NodeType>& node_type_exclude,
      const std::vector<NodeType>& node_type_include,
      const Node<TensorT>& node, 
      const TensorT& distance_weight,
      const std::string& direction);
    std::string selectRandomNode(
      const Model<TensorT>& model,
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
      const Model<TensorT>& model,
      const std::vector<NodeType>& source_node_type_exclude,
      const std::vector<NodeType>& source_node_type_include,
      const std::vector<NodeType>& sink_node_type_exclude,
      const std::vector<NodeType>& sink_node_type_include,
      const std::string& direction);
    std::string selectRandomLink(
      const Model<TensorT>& model,
      const std::vector<NodeType>& source_node_type_exclude,
      const std::vector<NodeType>& source_node_type_include,
      const std::vector<NodeType>& sink_node_type_exclude,
      const std::vector<NodeType>& sink_node_type_include);

		/**
		@brief Select random module given a set of conditions

		@param[in] model The model

		@returns A module name
		*/
		std::string selectRandomModule(
			const Model<TensorT>& model,
			const std::vector<NodeType>& node_type_exclude,
			const std::vector<NodeType>& node_type_include);

		/**
			@brief Copy a node in the model.
				This operation results in a layer addition below the target node
				whereby the weigh between the input node and target node are
				reused for the link between the new node and target node.

			@param[in, out] model The model
		*/
		void copyNodeDown(Model<TensorT>& model, std::string unique_str = "");
		
    /**
      @brief Copy a node in the model.  
				This operation results in a layer expansion to the left or right whereby all target
				node input and output node links are also copied.

      @param[in, out] model The model
    */ 
    void copyNodeRight(Model<TensorT>& model, std::string unique_str = "");

    /**
      @brief Add node to the model (Layer injection down).
        The method utilizes a modified version of the NEAT algorithm whereby a random
        link is chosen and bifurcated with a new node.  Instead, new nodes are added
        using the following procedure:
        1. an existing node is randomly chosen from the model.
        2. a randomly connected input link to the node is chosen.
          Note that an input link is chose because it is easier
          to exclude input nodes than output nodes.
        3. the chosen node is copied and a new link and new weight is added
          between the new node and the existing node.
        4. the new link becomes the input link of the existing node and the output link of the new node, 
          and existing link becomes the input link of the new node.

      References:
        Kenneth O. Stanley & Risto Miikkulainen (2002). "Evolving Neural Networks Through Augmenting Topologies". 
        Evolutionary Computation. 10 (2): 99–127. doi:10.1162/106365602320169811

      @param[in, out] model The model
    */ 
    void addNodeDown(Model<TensorT>& model, std::string unique_str = "", bool as_copy = false);

		/**
			@brief Add node to the model (Layer expansion right).
				New nodes are added	using the following procedure:
				1. an existing node is randomly chosen from the model.
				2. all node input and out put links are replicated and new weights
					for each link are made.
				3. the chosen node is copied.
				4. the new node is then connected to the replicated input and output links.

			@param[in, out] model The model
		*/
		void addNodeRight(Model<TensorT>& model, std::string unique_str = "", bool as_copy = false);

    /**
      @brief add link with a new weight to the model.

      @param[in, out] model The model
    */ 
    void addLink(Model<TensorT>& model, std::string unique_str = "");

		/**
			@brief copy an existing link (no new weight is created), and add the copied link to the model.

			@param[in, out] model The model
		*/
		void copyLink(Model<TensorT>& model, std::string unique_str = "");

		/**
		@brief Add a new module templated off of an existing module with new weights to the model

		@param[in, out] model The model
		*/
		void addModule(Model<TensorT>& model, std::string unique_str = "");

		/**
		@brief Copy an existing module (no new weights are created), and add the copied module to the model

		@param[in, out] model The model
		*/
		void copyModule(Model<TensorT>& model, std::string unique_str = "");

    /**
      @brief delete node to the model

      @param[in, out] model The model
      @param[in] prune_iterations The number of model recursive prune iterations
    */ 
    void deleteNode(Model<TensorT>& model, int prune_iterations = 1e6);

    /**
      @brief delete link to the model

      @param[in, out] model The model
      @param[in] prune_iterations The number of model recursive prune iterations
    */ 
    void deleteLink(Model<TensorT>& model, int prune_iterations = 1e6);

		/**
		@brief delete module in the model

		@param[in, out] model The model
		@param[in] prune_iterations The number of model recursive prune iterations
		*/
		void deleteModule(Model<TensorT>& model, int prune_iterations = 1e6);

		/**
		@brief change node activation

		@param[in, out] model The model
		*/
		void changeNodeActivation(Model<TensorT>& model, std::string unique_str = "");

		/**
		@brief change node integration

		@param[in, out] model The model
		*/
		void changeNodeIntegration(Model<TensorT>& model, std::string unique_str = "");

    /**
      @brief modify weights in the model

      @param[in, out] model The model
    */ 
    void modifyWeight(Model<TensorT>& model);

    /**
      @brief Make a unique time stampped hash of the form
        left_str + right_str + timestamp

      @param[in] left_str
      @param[in] right_str

      @returns A unique string hash
    */ 
    std::string makeUniqueHash(const std::string& left_str, const std::string& right_str);

		/**
		@brief Update the name of a node/link/weight/module

		@param[in] name Original name
		@param[in] new_name_format The format for the new name
		@param[in] unique_str A unique tag

		@returns A new name
		*/
		void updateName(const std::string& name, const std::string& new_name_format, std::string unique_str, 
			std::string& name_prefix, std::string& new_name);

    /**
      @brief randomly order the mutations

      @returns A random list of mutations types
    */ 
    std::vector<std::string> makeRandomModificationOrder();

		/**
		@brief set random model modification parameters

		@param[in] node_down_additions lower/upper bound for the number of potential node additions
		@param[in] link_additions lower/upper bound for the number of potential link additions
		@param[in] node_deletions lower/upper bound for the number of potential node deletions
		@param[in] link_deletions lower/upper bound for the number of potential link deletions
		*/
		void setRandomModifications(
			const std::pair<int, int>& node_down_additions,
			//const std::pair<int, int>& node_right_additions,
			//const std::pair<int, int>& node_down_copies,
			//const std::pair<int, int>& node_right_copies,
			const std::pair<int, int>& link_additions,
			//const std::pair<int, int>& link_copies,
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
			std::vector<Model<TensorT>>& models,
			std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) = 0;

private:
    // modification parameters
    int n_node_down_additions_ = 0; ///< new nodes "down" to add to the model (nodes are created through replication)
		int n_node_right_additions_ = 0; ///< new nodes "right" to add to the model (nodes are created through replication)
		int n_node_down_copies_ = 0; ///< nodes to duplicate "down" in the model
		int n_node_right_copies_ = 0; ///< nodes to duplicate "right" in the model
    int n_link_additions_ = 0; ///< new links to add to the model
		int n_link_copies_ = 0; ///< new links to copy in the model
    int n_node_deletions_ = 0; ///< nodes to remove from the model
    int n_link_deletions_ = 0; ///< links to remove from the model
		int n_node_activation_changes_ = 0; ///< nodes to change the activation
		int n_node_integration_changes_ = 0; ///< nodes to change the activation
		int n_module_additions_ = 0; ///< new modules added to the model (modules are created through replication)
		int n_module_deletions_ = 0; ///< modules to remove from the model

		// random modification parameters
		std::pair<int, int> node_down_additions_ = std::make_pair(0, 0);
		std::pair<int, int> node_right_additions_ = std::make_pair(0, 0);
		std::pair<int, int> node_down_copies_ = std::make_pair(0, 0);
		std::pair<int, int> node_right_copies_ = std::make_pair(0, 0);
		std::pair<int, int> link_additions_ = std::make_pair(0, 0);
		std::pair<int, int> link_copies_ = std::make_pair(0, 0);
		std::pair<int, int> node_deletions_ = std::make_pair(0, 0);
		std::pair<int, int> link_deletions_ = std::make_pair(0, 0);
		std::pair<int, int> node_activation_changes_ = std::make_pair(0, 0);
		std::pair<int, int> node_integration_changes_ = std::make_pair(0, 0);
		std::pair<int, int> module_additions_ = std::make_pair(0, 0);
		std::pair<int, int> module_deletions_ = std::make_pair(0, 0);
		std::vector<std::pair<std::shared_ptr<ActivationOp<TensorT>>, std::shared_ptr<ActivationOp<TensorT>>>> node_activations_;
		std::vector<std::tuple<std::shared_ptr<IntegrationOp<TensorT>>, std::shared_ptr<IntegrationErrorOp<TensorT>>, std::shared_ptr<IntegrationWeightGradOp<TensorT>>>> node_integrations_;
		

		// not yet implemented...
    int n_weight_changes_ = 0; ///< the number of weights to change in the model
    TensorT weight_change_stdev_ = 0; ///< the standard deviation to change the weights in the model
  };
	template<typename TensorT>
	void ModelReplicator<TensorT>::setNNodeDownCopies(const int& n_node_copies)
	{
		n_node_down_copies_ = n_node_copies;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNNodeDownAdditions(const int& n_node_additions)
	{
		n_node_down_additions_ = n_node_additions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNNodeRightCopies(const int& n_node_copies)
	{
		n_node_right_copies_ = n_node_copies;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNNodeRightAdditions(const int& n_node_additions)
	{
		n_node_right_additions_ = n_node_additions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNLinkAdditions(const int& n_link_additions)
	{
		n_link_additions_ = n_link_additions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNLinkCopies(const int& n_link_copies)
	{
		n_link_copies_ = n_link_copies;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNNodeDeletions(const int& n_node_deletions)
	{
		n_node_deletions_ = n_node_deletions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNLinkDeletions(const int& n_link_deletions)
	{
		n_link_deletions_ = n_link_deletions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNNodeActivationChanges(const int & n_node_activation_changes)
	{
		n_node_activation_changes_ = n_node_activation_changes;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNNodeIntegrationChanges(const int & n_node_integration_changes)
	{
		n_node_integration_changes_ = n_node_integration_changes;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNodeActivations(const std::vector<std::pair<std::shared_ptr<ActivationOp<TensorT>>, std::shared_ptr<ActivationOp<TensorT>>>>& node_activations)
	{
		node_activations_ = node_activations;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNodeIntegrations(const std::vector<std::tuple<std::shared_ptr<IntegrationOp<TensorT>>, std::shared_ptr<IntegrationErrorOp<TensorT>>, std::shared_ptr<IntegrationWeightGradOp<TensorT>>>>& node_integrations)
	{
		node_integrations_ = node_integrations;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNModuleAdditions(const int & n_module_additions)
	{
		n_module_additions_ = n_module_additions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNModuleDeletions(const int & n_module_deletions)
	{
		n_module_deletions_ = n_module_deletions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setNWeightChanges(const int& n_weight_changes)
	{
		n_weight_changes_ = n_weight_changes;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setWeightChangeStDev(const TensorT& weight_change_stdev)
	{
		weight_change_stdev_ = weight_change_stdev;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNNodeDownCopies() const
	{
		return n_node_down_copies_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNNodeDownAdditions() const
	{
		return n_node_down_additions_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNNodeRightCopies() const
	{
		return n_node_right_copies_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNNodeRightAdditions() const
	{
		return n_node_right_additions_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNLinkAdditions() const
	{
		return n_link_additions_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNLinkCopies() const
	{
		return n_link_copies_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNNodeDeletions() const
	{
		return n_node_deletions_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNLinkDeletions() const
	{
		return n_link_deletions_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNNodeActivationChanges() const
	{
		return n_node_activation_changes_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNNodeIntegrationChanges() const
	{
		return n_node_integration_changes_;
	}

	template<typename TensorT>
	std::vector<std::pair<std::shared_ptr<ActivationOp<TensorT>>, std::shared_ptr<ActivationOp<TensorT>>>> ModelReplicator<TensorT>::getNodeActivations() const
	{
		return node_activations_;
	}

	template<typename TensorT>
	std::vector<std::tuple<std::shared_ptr<IntegrationOp<TensorT>>, std::shared_ptr<IntegrationErrorOp<TensorT>>, std::shared_ptr<IntegrationWeightGradOp<TensorT>>>> ModelReplicator<TensorT>::getNodeIntegrations() const
	{
		return node_integrations_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNModuleAdditions() const
	{
		return n_module_additions_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNModuleDeletions() const
	{
		return n_module_deletions_;
	}

	template<typename TensorT>
	int ModelReplicator<TensorT>::getNWeightChanges() const
	{
		return n_weight_changes_;
	}

	template<typename TensorT>
	TensorT ModelReplicator<TensorT>::getWeightChangeStDev() const
	{
		return weight_change_stdev_;
	}

	template<typename TensorT>
	std::string ModelReplicator<TensorT>::makeUniqueHash(const std::string& left_str, const std::string& right_str)
	{
		std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
		std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
		std::tm now_tm = *std::localtime(&time_now_t);
		char timestamp[64];
		std::strftime(timestamp, 64, "%Y-%m-%d-%H-%M-%S", &now_tm);

		char hash_char[512];
		sprintf(hash_char, "%s_%s_%s", left_str.data(), right_str.data(), timestamp);
		std::string hash_str(hash_char);

		return hash_str;
	}

	template<typename TensorT>
	std::vector<std::string> ModelReplicator<TensorT>::selectNodes(
		const Model<TensorT>& model,
		const std::vector<NodeType>& node_type_exclude,
		const std::vector<NodeType>& node_type_include)
	{
		// populate our list of nodes to select from
		std::vector<std::string> node_ids;
		for (const Node<TensorT>& node : model.getNodes())
		{
			// check the exclusion list
			bool exclude_node = false;
			for (const NodeType& node_type : node_type_exclude)
			{
				if (node_type == node.getType())
				{
					exclude_node = true;
					break;
				}
			}

			// check the inclusion list
			bool include_node = true;
			if (node_type_include.size() > 0)
			{
				include_node = false;
				for (const NodeType& node_type : node_type_include)
				{
					if (node_type == node.getType())
					{
						include_node = true;
						break;
					}
				}
			}

			// add the node name to the list
			if (include_node && !exclude_node)
				node_ids.push_back(node.getName());
		}
		return node_ids;
	}

	template<typename TensorT>
	std::vector<std::string> ModelReplicator<TensorT>::selectModules(const Model<TensorT>& model, const std::vector<NodeType>& node_type_exclude, const std::vector<NodeType>& node_type_include)
	{
		// populate our list of modules to select from
		std::set<std::string> module_name_set;
		for (const Node<TensorT>& node : model.getNodes())
		{
			// check the exclusion list
			bool exclude_node = false;
			for (const NodeType& node_type : node_type_exclude)
			{
				if (node_type == node.getType())
				{
					exclude_node = true;
					break;
				}
			}

			// check the inclusion list
			bool include_node = true;
			if (node_type_include.size() > 0)
			{
				include_node = false;
				for (const NodeType& node_type : node_type_include)
				{
					if (node_type == node.getType())
					{
						include_node = true;
						break;
					}
				}
			}

			// add the node name to the list
			if (include_node && !exclude_node && !node.getModuleName().empty())
				module_name_set.insert(node.getModuleName());
		}

		std::vector<std::string> module_ids(module_name_set.begin(), module_name_set.end());
		return module_ids;
	}

	//std::string ModelReplicator<TensorT>::selectRandomNode(
	//  const Model<TensorT>& model,
	//  const std::vector<NodeType>& node_type_exclude,
	//  const std::vector<NodeType>& node_type_include,
	//  const Node<TensorT>& node, 
	//  const TensorT& distance_weight,
	//  const std::string& direction)
	//{
	//  // [TODO: add method body]    
	//}

	template<typename TensorT>
	std::string ModelReplicator<TensorT>::selectRandomNode(
		const Model<TensorT>& model,
		const std::vector<NodeType>& node_type_exclude,
		const std::vector<NodeType>& node_type_include)
	{
		std::vector<std::string> node_ids = selectNodes(model, node_type_exclude, node_type_include);

		if (node_ids.size() > 0)
			return selectRandomElement<std::string>(node_ids);
		else
		{
			printf("No nodes were found that matched the inclusion/exclusion criteria.\n");
			return "";
		}
	}

	template<typename TensorT>
	std::string ModelReplicator<TensorT>::selectRandomLink(
		const Model<TensorT>& model,
		const std::vector<NodeType>& source_node_type_exclude,
		const std::vector<NodeType>& source_node_type_include,
		const std::vector<NodeType>& sink_node_type_exclude,
		const std::vector<NodeType>& sink_node_type_include)
	{
		// select all source and sink nodes that meet the inclusion/exclusion criteria
		std::vector<std::string> source_node_ids = selectNodes(model, source_node_type_exclude, source_node_type_include);
		if (source_node_ids.size() == 0)
		{
			printf("No source nodes were found that matched the inclusion/exclusion criteria.\n");
			return "";
		}
		std::vector<std::string> sink_node_ids = selectNodes(model, sink_node_type_exclude, sink_node_type_include);
		if (sink_node_ids.size() == 0)
		{
			printf("No sink nodes were found that matched the inclusion/exclusion criteria.\n");
			return "";
		}

		// find all links that have an existing connection with the source and sink node candidates
		std::vector<std::string> link_ids;
		for (const Link& link : model.getLinks())
		{
			if (std::count(source_node_ids.begin(), source_node_ids.end(), link.getSourceNodeName()) != 0)
				if (std::count(sink_node_ids.begin(), sink_node_ids.end(), link.getSinkNodeName()) != 0)
					link_ids.push_back(link.getName());
		}

		if (link_ids.size() > 0)
			return selectRandomElement<std::string>(link_ids);
		else
		{
			printf("No links were found that matched the node inclusion/exclusion criteria.\n");
			return "";
		}
	}

	template<typename TensorT>
	inline void ModelReplicator<TensorT>::addNodeRight(Model<TensorT>& model, std::string unique_str, bool as_copy)
	{
		// pick a random node from the model
		// that is not an input or bias    
		std::vector<NodeType> node_exclusion_list = { NodeType::bias, NodeType::input, NodeType::output };
		std::vector<NodeType> node_inclusion_list = { NodeType::hidden };
		std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);
		if (random_node_name.empty() || random_node_name == "")
		{
			std::cout << "No nodes were added to the model." << std::endl;
			return;
		}

		// copy the node
		Node<TensorT> new_node = model.getNode(random_node_name);

		std::string new_node_name, add_node_name;
		if (as_copy) updateName(random_node_name, "%s@copyNodeRight#", unique_str, add_node_name, new_node_name);
		else updateName(random_node_name, "%s@addNodeRight#", unique_str, add_node_name, new_node_name);
		new_node.setName(new_node_name);
		new_node.setType(NodeType::hidden); // [TODO: add test to check for the type!
		model.addNodes({ new_node });

		std::vector<std::string> input_link_names, output_link_names;
		std::vector<std::string> bias_link_names;
		for (const Link& link : model.getLinks())
		{
			// find the random_nodes bias
			if (link.getSinkNodeName() == random_node_name &&
				model.getNode(link.getSourceNodeName()).getType() == NodeType::bias){
				bias_link_names.push_back(link.getName());
			}
			if (link.getSinkNodeName() == random_node_name &&
				model.getNode(link.getSourceNodeName()).getType() != NodeType::bias) {
				input_link_names.push_back(link.getName());
			}
			if (link.getSourceNodeName() == random_node_name) {
				output_link_names.push_back(link.getName());
			}
		}
		if (input_link_names.size() == 0)
		{
			std::cout << "No nodes were added to the model." << std::endl;
			return;
		}

		if (bias_link_names.size() != 0) {
			std::string new_bias_name;
			if (!as_copy) {
				// create a new bias
				char new_bias_name_char[512];
				sprintf(new_bias_name_char, "Bias_%s@addNodeRight#", add_node_name.data());
				new_bias_name = makeUniqueHash(new_bias_name_char, unique_str);
				Node<TensorT> new_bias(new_bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
				model.addNodes({ new_bias });
			}
			else
				new_bias_name = model.getLink(bias_link_names[0]).getSourceNodeName();

			// create a link from the new bias to the new node
			std::string weight_bias_name;
			if (!as_copy) {
				char weight_bias_name_char[512];
				sprintf(weight_bias_name_char, "%s_to_%s@addNodeRight#", new_bias_name.data(), new_node_name.data());
				weight_bias_name = makeUniqueHash(weight_bias_name_char, unique_str);
				std::shared_ptr<WeightInitOp<TensorT>> bias_weight_init;
				bias_weight_init.reset(new ConstWeightInitOp<TensorT>(1.0));
				Weight<TensorT> weight_bias = model.getWeight(model.getLink(bias_link_names[0]).getWeightName()); // [OPTIMIZATION: use Link.getWeightName() directly]
				weight_bias.setName(weight_bias_name);
				weight_bias.setWeightInitOp(bias_weight_init);
				model.addWeights({ weight_bias });
			}
			else
				weight_bias_name = model.getLink(bias_link_names[0]).getWeightName();

			char link_bias_name_char[512];
			if (as_copy) sprintf(link_bias_name_char, "%s_to_%s@copyNodeRight#", new_bias_name.data(), new_node_name.data());
			else sprintf(link_bias_name_char, "%s_to_%s@addNodeRight#", new_bias_name.data(), new_node_name.data());
			std::string link_bias_name = makeUniqueHash(link_bias_name_char, unique_str);
			Link link_bias(link_bias_name, new_bias_name, new_node_name, weight_bias_name);
			model.addLinks({ link_bias });
		}

		// replicate all input connections
		for (const std::string& input_link_name : input_link_names) {
			// change the source to new node weight
			std::string weight_name;
			if (!as_copy) {
				Weight<TensorT> weight = model.getWeight(model.getLink(input_link_name).getWeightName()); // copy assignment
				char weight_name_char[512];
				sprintf(weight_name_char, "Weight_%s_to_%s@addNodeRight#", model.getLink(input_link_name).getSourceNodeName().data(), new_node_name.data());
				weight_name = makeUniqueHash(weight_name_char, unique_str);
				weight.setName(weight_name);
				model.addWeights({ weight });
			}
			else
				weight_name = model.getLink(input_link_name).getWeightName();

			// change the source to new node link
			Link modified_link = model.getLink(input_link_name);
			modified_link.setSinkNodeName(new_node_name);
			modified_link.setWeightName(weight_name);
			char modified_link_name_char[512];
			if (as_copy) sprintf(modified_link_name_char, "Link_%s_to_%s@copyNodeRight#", modified_link.getSourceNodeName().data(), new_node_name.data());
			else sprintf(modified_link_name_char, "Link_%s_to_%s@addNodeRight#", modified_link.getSourceNodeName().data(), new_node_name.data());
			std::string modified_link_name = makeUniqueHash(modified_link_name_char, unique_str);
			modified_link.setName(modified_link_name);
			model.addLinks({ modified_link });
		}

		// replicate all output connections
		for (const std::string& output_link_name : output_link_names) {
			// change the source to new node weight
			std::string weight_name;
			if (!as_copy) {
				Weight<TensorT> weight = model.getWeight(model.getLink(output_link_name).getWeightName()); // copy assignment
				char weight_name_char[512];
				sprintf(weight_name_char, "Weight_%s_to_%s@addNodeRight#", new_node_name.data(), model.getLink(output_link_name).getSinkNodeName().data());
				weight_name = makeUniqueHash(weight_name_char, unique_str);
				weight.setName(weight_name);
				model.addWeights({ weight });
			}
			else
				weight_name = model.getLink(output_link_name).getWeightName();

			// change the source to new node link
			Link modified_link = model.getLink(output_link_name);
			modified_link.setSourceNodeName(new_node_name);
			modified_link.setWeightName(weight_name);
			char modified_link_name_char[512];
			if (as_copy) sprintf(modified_link_name_char, "Link_%s_to_%s@copyNodeRight#", new_node_name.data(), modified_link.getSinkNodeName().data());
			else sprintf(modified_link_name_char, "Link_%s_to_%s@addNodeRight#", new_node_name.data(), modified_link.getSinkNodeName().data());
			std::string modified_link_name = makeUniqueHash(modified_link_name_char, unique_str);
			modified_link.setName(modified_link_name);
			model.addLinks({ modified_link });
		}
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::addLink(
		Model<TensorT>& model, std::string unique_str)
	{
		// define the inclusion/exclusion nodes
		const std::vector<NodeType> source_node_type_exclude = { NodeType::bias, NodeType::output }; // no output can be a source
		const std::vector<NodeType> source_node_type_include = {};
		const std::vector<NodeType> sink_node_type_exclude = { NodeType::bias, NodeType::input };  // no input can be a sink
		const std::vector<NodeType> sink_node_type_include = {};

		// select candidate source nodes
		std::vector<std::string> source_node_ids = selectNodes(model, source_node_type_exclude, source_node_type_include);
		if (source_node_ids.size() == 0)
		{
			printf("No source nodes were found that matched the inclusion/exclusion criteria.\n");
			return;
		}

		// select a random source node
		std::string source_node_name = selectRandomElement<std::string>(source_node_ids);

		// select candidate sink nodes
		std::vector<std::string> sink_node_ids = selectNodes(model, sink_node_type_exclude, sink_node_type_include);
		if (sink_node_ids.size() == 0)
		{
			printf("No sink nodes were found that matched the inclusion/exclusion criteria.\n");
			return;
		}

		// remove candidate sink nodes for which a link already exists


		// select a random sink node
		std::string sink_node_name = selectRandomElement<std::string>(sink_node_ids);

		// [TODO: Need a check if the link already exists...]

		// create the new weight based on a random link (this can probably be optmized...)
		std::string random_link = selectRandomLink(model, source_node_type_exclude, source_node_type_include, sink_node_type_exclude, sink_node_type_include);

		Weight<TensorT> weight = model.getWeight(model.getLink(random_link).getWeightName()); // copy assignment
		char weight_name_char[512];
		sprintf(weight_name_char, "Weight_%s_to_%s@addLink#", source_node_name.data(), sink_node_name.data());
		std::string weight_name = makeUniqueHash(weight_name_char, unique_str);
		weight.setName(weight_name);
		model.addWeights({ weight });

		// create the new link
		char link_name_char[512];
		sprintf(link_name_char, "Link_%s_to_%s@addLink#", source_node_name.data(), sink_node_name.data());
		std::string link_name = makeUniqueHash(link_name_char, unique_str);
		Link link(link_name, source_node_name, sink_node_name, weight_name);
		model.addLinks({ link });
	}

	template<typename TensorT>
	inline void ModelReplicator<TensorT>::copyLink(Model<TensorT>& model, std::string unique_str)
	{
		// [TODO: add method body]
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::addModule(Model<TensorT>& model, std::string unique_str)
	{
		// pick a random module from the model
		std::vector<NodeType> node_exclusion_list = {};
		std::vector<NodeType> node_inclusion_list = {};
		std::string random_module_name = selectRandomModule(model, node_exclusion_list, node_inclusion_list);
		if (random_module_name.empty())
		{
			std::cout << "No modules were added to the model." << std::endl;
			return;
		}

		// update the module name [TODO: update the module ID]
		std::string new_name_format = "%s@addModule#";
		std::string new_module_name, module_name_prefix;
		updateName(random_module_name, new_name_format, unique_str, module_name_prefix, new_module_name);
		std::string new_module_suffix = makeUniqueHash("@addModule#", unique_str); // time-stamp should be constant!

		// copy the module and reconnect the links
		std::vector<Node<TensorT>> new_nodes;
		std::vector<Link> new_links;
		std::vector<Weight<TensorT>> new_weights;
		std::vector<Link> connecting_links;
		std::vector<Weight<TensorT>> connecting_weights;
		for (Link& link : model.getLinks())
		{
			if (link.getModuleName() == random_module_name)
			{ // copy the internal nodes, weights, and links, and give them a new name/id/module_name/module_id
				Node<TensorT> source_node = model.getNode(link.getSourceNodeName());
				std::string new_node_name, node_prefix;
				updateName(source_node.getName(), new_name_format, unique_str, node_prefix, new_node_name);
				source_node.setName(node_prefix + new_module_suffix);
				source_node.setModuleName(new_module_name);
				if (std::count(new_nodes.begin(), new_nodes.end(), source_node) == 0)
					new_nodes.push_back(source_node);

				Node<TensorT> sink_node = model.getNode(link.getSinkNodeName());
				updateName(sink_node.getName(), new_name_format, unique_str, node_prefix, new_node_name);
				sink_node.setName(node_prefix + new_module_suffix);
				sink_node.setModuleName(new_module_name);
				if (std::count(new_nodes.begin(), new_nodes.end(), sink_node) == 0)
					new_nodes.push_back(sink_node);

				Weight<TensorT> weight = model.getWeight(link.getWeightName());
				std::string new_weight_name, weight_prefix;
				updateName(weight.getName(), new_name_format, unique_str, weight_prefix, new_weight_name);
				weight.setName(weight_prefix + new_module_suffix);
				weight.setModuleName(new_module_name);
				if (std::count(new_weights.begin(), new_weights.end(), weight) == 0)
					new_weights.push_back(weight);

				std::string new_link_name, link_prefix;
				updateName(link.getName(), new_name_format, unique_str, link_prefix, new_link_name);
				link.setName(link_prefix + new_module_suffix);
				link.setModuleName(new_module_name);
				link.setSourceNodeName(source_node.getName());
				link.setSinkNodeName(sink_node.getName());
				link.setWeightName(weight.getName());
				if (std::count(new_links.begin(), new_links.end(), link) == 0)
					new_links.push_back(link);
			}
			else if (model.getNode(link.getSourceNodeName()).getModuleName() == random_module_name)
			{ // copy the connecting links and weights, and give them a new name/id
				// and update the source node name (i.e., connect to the new module)

				Weight<TensorT> weight = model.getWeight(link.getWeightName());
				std::string new_weight_name, weight_prefix;
				updateName(weight.getName(), new_name_format, unique_str, weight_prefix, new_weight_name);
				weight.setName(weight_prefix + new_module_suffix);
				if (std::count(connecting_weights.begin(), connecting_weights.end(), weight) == 0)
					connecting_weights.push_back(weight);

				std::string new_link_name, link_prefix;
				updateName(link.getName(), new_name_format, unique_str, link_prefix, new_link_name);
				link.setName(link_prefix + new_module_suffix);
				std::string new_node_name, node_prefix;
				updateName(link.getSourceNodeName(), new_name_format, unique_str, node_prefix, new_node_name);
				link.setSourceNodeName(node_prefix + new_module_suffix);
				link.setWeightName(weight.getName());
				if (std::count(connecting_links.begin(), connecting_links.end(), link) == 0)
					connecting_links.push_back(link);
			}
			else if (model.getNode(link.getSinkNodeName()).getModuleName() == random_module_name)
			{ // copy the connecting links and weights, and give them a new name/id
				// and update the sink node name (i.e., connect to the new module)

				Weight<TensorT> weight = model.getWeight(link.getWeightName());
				std::string new_weight_name, weight_prefix;
				updateName(weight.getName(), new_name_format, unique_str, weight_prefix, new_weight_name);
				weight.setName(weight_prefix + new_module_suffix);
				if (std::count(connecting_weights.begin(), connecting_weights.end(), weight) == 0)
					connecting_weights.push_back(weight);

				std::string new_link_name, link_prefix;
				updateName(link.getName(), new_name_format, unique_str, link_prefix, new_link_name);
				link.setName(link_prefix + new_module_suffix);
				std::string new_node_name, node_prefix;
				updateName(link.getSinkNodeName(), new_name_format, unique_str, node_prefix, new_node_name);
				link.setSinkNodeName(node_prefix + new_module_suffix);
				link.setWeightName(weight.getName());
				if (std::count(connecting_links.begin(), connecting_links.end(), link) == 0)
					connecting_links.push_back(link);
			}
		}

		// add the new nodes/links/weights to the model
		model.addNodes(new_nodes);
		model.addWeights(new_weights);
		model.addLinks(new_links);
		model.addWeights(connecting_weights);
		model.addLinks(connecting_links);
	}

	template<typename TensorT>
	inline void ModelReplicator<TensorT>::copyModule(Model<TensorT>& model, std::string unique_str)
	{
		// [TODO: add method body]
	}

	template<typename TensorT>
	std::string ModelReplicator<TensorT>::selectRandomModule(const Model<TensorT>& model, const std::vector<NodeType>& node_type_exclude, const std::vector<NodeType>& node_type_include)
	{
		std::vector<std::string> module_ids = selectModules(model, node_type_exclude, node_type_include);

		if (module_ids.size() > 0)
			return selectRandomElement<std::string>(module_ids);
		else
		{
			printf("No nodes were found that matched the inclusion/exclusion criteria.\n");
			return "";
		}
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::copyNodeDown(Model<TensorT>& model, std::string unique_str)
	{
		addNodeDown(model, unique_str, true);
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::copyNodeRight(Model<TensorT>& model, std::string unique_str)
	{
		addNodeRight(model, unique_str, true);
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::addNodeDown(Model<TensorT>& model, std::string unique_str, bool as_copy)
	{
		// pick a random node from the model
		// that is not an input or bias    
		std::vector<NodeType> node_exclusion_list = { NodeType::bias, NodeType::input };
		std::vector<NodeType> node_inclusion_list = { NodeType::hidden, NodeType::output };
		std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);
		if (random_node_name.empty() || random_node_name == "")
		{
			std::cout << "No nodes were added to the model." << std::endl;
			return;
		}

		// copy the node
		Node<TensorT> new_node = model.getNode(random_node_name);

		// select a random input link
		// [OPTIMIZATION: refactor to pass back the Link and not just the name]
		std::vector<std::string> input_link_names, bias_link_names;
		for (const Link& link : model.getLinks())
		{
			if (link.getSinkNodeName() == random_node_name &&
				model.getNode(link.getSourceNodeName()).getType() != NodeType::bias){
				input_link_names.push_back(link.getName());
			}
			if (link.getSinkNodeName() == random_node_name &&
				model.getNode(link.getSourceNodeName()).getType() == NodeType::bias) {
				bias_link_names.push_back(link.getName());
			}
		}
		if (input_link_names.size() == 0)
		{
			std::cout << "No nodes were added to the model." << std::endl;
			return;
		}
		std::string input_link_name = selectRandomElement<std::string>(input_link_names);

		std::string new_node_name, add_node_name;
		if (as_copy) updateName(random_node_name, "%s@copyNodeDown#", unique_str, add_node_name, new_node_name);
		else updateName(random_node_name, "%s@addNodeDown#", unique_str, add_node_name, new_node_name);
		new_node.setName(new_node_name);
		new_node.setType(NodeType::hidden); // [TODO: add test to check for the type!
		model.addNodes({ new_node });

		if (bias_link_names.size() != 0) {
			std::string new_bias_name;
			if (!as_copy) {
				// create a new bias
				char new_bias_name_char[512];
				sprintf(new_bias_name_char, "Bias_%s@addNodeRight#", add_node_name.data());
				new_bias_name = makeUniqueHash(new_bias_name_char, unique_str);
				Node<TensorT> new_bias(new_bias_name, NodeType::bias, NodeStatus::activated, std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()), std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>()), std::shared_ptr<IntegrationOp<TensorT>>(new SumOp<TensorT>()), std::shared_ptr<IntegrationErrorOp<TensorT>>(new SumErrorOp<TensorT>()), std::shared_ptr<IntegrationWeightGradOp<TensorT>>(new SumWeightGradOp<TensorT>()));
				model.addNodes({ new_bias });
			}
			else
				new_bias_name = model.getLink(bias_link_names[0]).getSourceNodeName();

			// create a link from the new bias to the new node
			std::string weight_bias_name;
			if (!as_copy) {
				char weight_bias_name_char[512];
				sprintf(weight_bias_name_char, "%s_to_%s@addNodeRight#", new_bias_name.data(), new_node_name.data());
				weight_bias_name = makeUniqueHash(weight_bias_name_char, unique_str);
				std::shared_ptr<WeightInitOp<TensorT>> bias_weight_init;
				bias_weight_init.reset(new ConstWeightInitOp<TensorT>(1.0));
				Weight<TensorT> weight_bias = model.getWeight(model.getLink(bias_link_names[0]).getWeightName()); // [OPTIMIZATION: use Link.getWeightName() directly]
				weight_bias.setName(weight_bias_name);
				weight_bias.setWeightInitOp(bias_weight_init);
				model.addWeights({ weight_bias });
			}
			else
				weight_bias_name = model.getLink(bias_link_names[0]).getWeightName();

			char link_bias_name_char[512];
			if (as_copy) sprintf(link_bias_name_char, "%s_to_%s@copyNodeRight#", new_bias_name.data(), new_node_name.data());
			else sprintf(link_bias_name_char, "%s_to_%s@addNodeRight#", new_bias_name.data(), new_node_name.data());
			std::string link_bias_name = makeUniqueHash(link_bias_name_char, unique_str);
			Link link_bias(link_bias_name, new_bias_name, new_node_name, weight_bias_name);
			model.addLinks({ link_bias });
		}

		// change the output node name of the link to the new copied node name
		Link modified_link = model.getLink(input_link_name);
		modified_link.setSinkNodeName(new_node_name);
		char modified_link_name_char[512];
		if (as_copy) sprintf(modified_link_name_char, "Link_%s_to_%s@copyNodeDown#", modified_link.getSourceNodeName().data(), new_node_name.data());
		else sprintf(modified_link_name_char, "Link_%s_to_%s@addNodeDown#", modified_link.getSourceNodeName().data(), new_node_name.data());
		std::string modified_link_name = makeUniqueHash(modified_link_name_char, unique_str);
		modified_link.setName(modified_link_name);
		model.addLinks({ modified_link });

		// add a new weight that connects the new copied node
		// to its original node
		std::string weight_name;
		if (!as_copy) {
			Weight<TensorT> weight = model.getWeight(model.getLink(input_link_name).getWeightName()); // copy assignment
			char weight_name_char[512];
			sprintf(weight_name_char, "Weight_%s_to_%s@addNodeDown#", new_node_name.data(), random_node_name.data());
			weight_name = makeUniqueHash(weight_name_char, unique_str);
			weight.setName(weight_name);
			model.addWeights({ weight });
		}
		else
			weight_name = model.getLink(input_link_name).getWeightName();

		// add a new link that connects the new copied node
		// to its original node
		char link_name_char[512];
		if (as_copy) sprintf(link_name_char, "Link_%s_to_%s@copyNodeDown#", new_node_name.data(), random_node_name.data());
		else sprintf(link_name_char, "Link_%s_to_%s@addNodeDown#", new_node_name.data(), random_node_name.data());
		std::string link_name = makeUniqueHash(link_name_char, unique_str);
		Link link(link_name, new_node_name, random_node_name, weight_name);
		model.addLinks({ link });

		// remove the unmodified link
	  // [CHECK: is this needed?  identified as a high CPU call due to prune weights]
		model.removeLinks({ input_link_name });
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::deleteNode(Model<TensorT>& model, int prune_iterations)
	{
		// pick a random node from the model
		// that is not an input, bias, nor output
		std::vector<NodeType> node_exclusion_list = { NodeType::bias, NodeType::input, NodeType::output, NodeType::unmodifiable };
		std::vector<NodeType> node_inclusion_list = { NodeType::hidden };
		std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);

		// delete the node, its bias, and its bias link
		if (!random_node_name.empty() || random_node_name != "") // isn't this this same thing?
		{
			// std::cout<<"Random node name: "<<random_node_name<<std::endl;
			model.removeNodes({ random_node_name });
			model.pruneModel(prune_iterations);  // this action can remove additional nodes including inputs, biases, and outputs
		}
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::deleteLink(Model<TensorT>& model, int prune_iterations)
	{
		// pick a random link from the model
		// that does not connect from a bias or input
		// [TODO: need to implement a check that the deletion does not also remove an input/output node]
		std::vector<NodeType> source_exclusion_list = { NodeType::bias, NodeType::unmodifiable };
		std::vector<NodeType> source_inclusion_list = {};
		std::vector<NodeType> sink_exclusion_list = { NodeType::bias, NodeType::unmodifiable };
		std::vector<NodeType> sink_inclusion_list = {};
		std::string random_link_name = selectRandomLink(
			model, source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list);

		// delete the link and weight if required
		if (!random_link_name.empty() || random_link_name != "") // isn't this this same thing?
		{
			model.removeLinks({ random_link_name });
			model.pruneModel(prune_iterations);  // this action can remove additional nodes including inputs, biases, and outputs
		}
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::deleteModule(Model<TensorT>& model, int prune_iterations)
	{
		// pick a random module from the model
		std::vector<NodeType> node_exclusion_list = {};
		std::vector<NodeType> node_inclusion_list = {};
		std::string random_module_name = selectRandomModule(model, node_exclusion_list, node_inclusion_list);
		if (random_module_name.empty())
		{
			std::cout << "No modules were deleted from the model." << std::endl;
			return;
		}

		// remove nodes/link/weights from the model
		std::vector<std::string> delete_nodes;
		std::vector<std::string> delete_links;
		std::vector<std::string> delete_weights;
		for (Link& link : model.getLinks())
		{
			if (link.getModuleName() == random_module_name)
			{
				delete_links.push_back(link.getName());
				delete_nodes.push_back(link.getSourceNodeName());
				delete_nodes.push_back(link.getSinkNodeName());
				delete_weights.push_back(link.getWeightName());
			}
		}
		model.removeNodes(delete_nodes);
		model.removeLinks(delete_links);
		model.removeWeights(delete_weights);

		// prune the model
		model.pruneModel(prune_iterations);  // this action can remove additional nodes including inputs, biases, and outputs
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::changeNodeActivation(Model<TensorT>& model, std::string unique_str)
	{
		// pick a random node from the model
		// that is not an input or bias or output
		std::vector<NodeType> node_exclusion_list = { NodeType::bias, NodeType::input, NodeType::output };
		std::vector<NodeType> node_inclusion_list = { NodeType::hidden };
		std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);
		if (random_node_name.empty() || random_node_name == "")
		{
			std::cout << "No node activations were changed in the model." << std::endl;
			return;
		}

		Node<TensorT> new_node = model.getNode(random_node_name); // copy the node		
		std::pair<std::shared_ptr<ActivationOp<TensorT>>, std::shared_ptr<ActivationOp<TensorT>>> new_activation = selectRandomElement(node_activations_); // pick a random activation
		new_node.setActivation(new_activation.first); // change the activation
		new_node.setActivationGrad(new_activation.second); // change the activation
		model.removeNodes({ new_node.getName() }); // delete the original node
		model.addNodes({ new_node }); // add in the new node
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::changeNodeIntegration(Model<TensorT>& model, std::string unique_str)
	{
		// pick a random node from the model
		// that is not an input or bias or output
		std::vector<NodeType> node_exclusion_list = { NodeType::bias, NodeType::input, NodeType::output };
		std::vector<NodeType> node_inclusion_list = { NodeType::hidden };
		std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);
		if (random_node_name.empty() || random_node_name == "")
		{
			std::cout << "No node activations were changed in the model." << std::endl;
			return;
		}

		Node<TensorT> new_node = model.getNode(random_node_name); // copy the node		
		std::tuple<std::shared_ptr<IntegrationOp<TensorT>>, std::shared_ptr<IntegrationErrorOp<TensorT>>, std::shared_ptr<IntegrationWeightGradOp<TensorT>>> new_integration = selectRandomElement(node_integrations_); // pick a random integration
		new_node.setIntegration(std::get<0>(new_integration)); // change the integration
		new_node.setIntegrationError(std::get<1>(new_integration)); // change the integration
		new_node.setIntegrationWeightGrad(std::get<2>(new_integration)); // change the integration
		model.removeNodes({ new_node.getName() }); // delete the original node
		model.addNodes({ new_node }); // add in the new node
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::modifyWeight(Model<TensorT>& model)
	{
		// [TODO: add method body]    

		// select a random link from the model

		// change the weight

		// update the link's weight name

		// add the new weight back into the model
		// delete the previous weight

	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::updateName(const std::string & name, const std::string & new_name_format, std::string unique_str,
		std::string& name_prefix, std::string& new_name)
	{
		std::regex re("@");
		std::vector<std::string> str_tokens;
		name_prefix = name;
		std::copy(
			std::sregex_token_iterator(name.begin(), name.end(), re, -1),
			std::sregex_token_iterator(),
			std::back_inserter(str_tokens));
		if (str_tokens.size() > 1)
			name_prefix = str_tokens[0]; // only retain the last timestamp
																		 // printf("New node name: %s\n", add_name.data());

		char new_name_char[512];
		sprintf(new_name_char, new_name_format.data(), name_prefix.data());
		new_name = makeUniqueHash(new_name_char, unique_str);
	}

	template<typename TensorT>
	std::vector<std::string> ModelReplicator<TensorT>::makeRandomModificationOrder()
	{
		// create the list of modifications
		std::vector<std::string> modifications;
		for (int i = 0; i < n_node_activation_changes_; ++i) modifications.push_back("change_node_activation");
		for (int i = 0; i < n_node_integration_changes_; ++i) modifications.push_back("change_node_integration");
		for (int i = 0; i < n_node_down_additions_; ++i) modifications.push_back("add_node");
		for (int i = 0; i < n_link_additions_; ++i) modifications.push_back("add_link");
		for (int i = 0; i < n_module_additions_; ++i) modifications.push_back("add_module");
		for (int i = 0; i < n_node_deletions_; ++i) modifications.push_back("delete_node");
		for (int i = 0; i < n_link_deletions_; ++i) modifications.push_back("delete_link");
		for (int i = 0; i < n_module_deletions_; ++i) modifications.push_back("delete_module");

		// // randomize
		// std::random_device seed;
		// std::mt19937 engine(seed());
		// std::shuffle(modifications.begin(), modifications.end(), engine);

		return modifications;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::setRandomModifications(
		const std::pair<int, int>& node_down_additions,
		const std::pair<int, int>& link_additions,
		const std::pair<int, int>& node_deletions,
		const std::pair<int, int>& link_deletions,
		const std::pair<int, int>& node_activation_changes,
		const std::pair<int, int>& node_integration_changes,
		const std::pair<int, int>& module_additions,
		const std::pair<int, int>& module_deletions)
	{
		// set 
		node_down_additions_ = node_down_additions;
		link_additions_ = link_additions;
		node_deletions_ = node_deletions;
		link_deletions_ = link_deletions;
		node_activation_changes_ = node_activation_changes;
		node_integration_changes_ = node_integration_changes;
		module_additions_ = module_additions;
		module_deletions_ = module_deletions;
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::makeRandomModifications()
	{
		// random generator for model modifications
		std::random_device rd;
		std::mt19937 gen(rd());

		// set 
		std::uniform_int_distribution<> node_down_addition_gen(node_down_additions_.first, node_down_additions_.second);
		setNNodeDownAdditions(node_down_addition_gen(gen));
		std::uniform_int_distribution<> link_addition_gen(link_additions_.first, link_additions_.second);
		setNLinkAdditions(link_addition_gen(gen));
		std::uniform_int_distribution<> node_deletion_gen(node_deletions_.first, node_deletions_.second);
		setNNodeDeletions(node_deletion_gen(gen));
		std::uniform_int_distribution<> link_deletion_gen(link_deletions_.first, link_deletions_.second);
		setNLinkDeletions(link_deletion_gen(gen));
		std::uniform_int_distribution<> node_activation_changes_gen(node_activation_changes_.first, node_activation_changes_.second);
		setNNodeActivationChanges(node_activation_changes_gen(gen));
		std::uniform_int_distribution<> node_integration_changes_gen(node_integration_changes_.first, node_integration_changes_.second);
		setNNodeIntegrationChanges(node_integration_changes_gen(gen));
		std::uniform_int_distribution<> module_addition_gen(module_additions_.first, module_additions_.second);
		setNModuleAdditions(module_addition_gen(gen));
		std::uniform_int_distribution<> module_deletion_gen(module_deletions_.first, module_deletions_.second);
		setNModuleDeletions(module_deletion_gen(gen));
	}

	template<typename TensorT>
	void ModelReplicator<TensorT>::modifyModel(Model<TensorT>& model, std::string unique_str)
	{
		// randomly order the modifications
		std::vector<std::string> modifications = makeRandomModificationOrder();

		// implement each modification one at a time
		// and track the counts that each modification is called
		std::map<std::string, int> modifications_counts;
		for (const std::string& modification : modifications)
			modifications_counts.emplace(modification, 0);

		const int prune_iterations = 1e6;
		for (const std::string& modification : modifications)
		{
			// [TODO: copyNodeRight]
			if (modification == "add_node")
			{
				addNodeDown(model, unique_str + "-" + std::to_string(modifications_counts.at(modification)));
				modifications_counts[modification] += 1;
			}
			else if (modification == "add_link")
			{
				addLink(model, unique_str + "-" + std::to_string(modifications_counts.at(modification)));
				modifications_counts[modification] += 1;
			}
			else if (modification == "delete_node")
			{
				deleteNode(model, prune_iterations);
				modifications_counts[modification] += 1;
			}
			else if (modification == "delete_link")
			{
				deleteLink(model, prune_iterations);
				modifications_counts[modification] += 1;
			}
			else if (modification == "change_node_activation")
			{
				changeNodeActivation(model);
				modifications_counts[modification] += 1;
			}
			else if (modification == "change_node_integration")
			{
				changeNodeIntegration(model);
				modifications_counts[modification] += 1;
			}
			if (modification == "add_module")
			{
				addModule(model, unique_str + "-" + std::to_string(modifications_counts.at(modification)));
				modifications_counts[modification] += 1;
			}
			else if (modification == "delete_module")
			{
				deleteModule(model, prune_iterations);
				modifications_counts[modification] += 1;
			}
			// [TODO: modifyWeight]
		}
	}
}

#endif //SMARTPEAK_MODELREPLICATOR_H