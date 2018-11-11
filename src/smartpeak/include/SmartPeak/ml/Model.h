/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

// .h
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Weight.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <tuple>
#include <list>
#include <set>

// .cpp
#include <SmartPeak/graph/CircuitFinder.h>

#include <iostream>

namespace SmartPeak
{
  /**
    @brief Directed Network Model

    Assumptions about the model structure:
    1. Inputs can only be sources
    2. Outputs can only be sinks (will break back propogation algorithm)
  */
	template<typename TensorT>
  class Model
  {
public:
    Model() = default; ///< Default constructor
    Model(const Model& other); ///< Copy constructor that does not create a shared memory address between model nodes/links/weights
    Model(const int& id); ///< Explicit constructor  
    ~Model() = default; ///< Default destructor

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


		/**
		@brief Copy assignment operator that creates a new model with different memory addresses
		*/
    inline Model& operator=(const Model& other)
    {
      id_ = other.id_;
      name_ = other.name_;
      links_ = other.links_;
      nodes_ = other.nodes_;
      weights_ = other.weights_;
			cyclic_pairs_ = other.cyclic_pairs_;
      return *this;
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< name setter
    std::string getName() const; ///< name getter

		std::vector<std::shared_ptr<Node<TensorT>>> getInputNodes(); ///< input_node getter
		std::vector<std::shared_ptr<Node<TensorT>>> getOutputNodes(); ///< output_node getter
		std::vector<std::string> getOutputNodeNames() const;
 
    /**
      @brief Add new links to the model.

      @param[in] links Links to add to the model
    */ 
    void addLinks(const std::vector<Link>& links);
    Link getLink(const std::string& link_name) const; ///< link getter
    std::vector<Link> getLinks() const;  ///< links getter
		std::map<std::string, std::shared_ptr<Link>> getLinksMap();  ///< return a modifiable version of weights
 
    /**
      @brief Remove existing links from the model.

      @param[in] Link_names Links to remove from the model
    */ 
    void removeLinks(const std::vector<std::string>& link_names);
 
    /**
      @brief Add new nodes to the model.

      @param[in] nodes Nodes to add to the model
    */ 
    void addNodes(const std::vector<Node<TensorT>>& nodes);
    Node<TensorT> getNode(const std::string& node_name) const; ///< node getter
    std::vector<Node<TensorT>> getNodes() const; ///< nodes getter
		std::map<std::string, std::shared_ptr<Node<TensorT>>> getNodesMap();  ///< return a modifiable version of weights
		std::map<std::string, std::vector<std::string>> getModuleNodeNameMap() const; ///< return a map of modules to a vector of node names [TODO: test!]

    /**
      @brief Remove existing nodes from the model.

      @param[in] node_names Nodes to remove from the model
    */ 
    void removeNodes(const std::vector<std::string>& node_names);
 
    /**
      @brief Add new weights to the model.

      @param[in] weights Weights to add to the model
    */ 
    void addWeights(const std::vector<Weight<TensorT>>& weights);
    Weight<TensorT> getWeight(const std::string& weight_name) const; ///< weight getter
    std::vector<Weight<TensorT>> getWeights() const;  ///< weights getter
		std::map<std::string, std::shared_ptr<Weight<TensorT>>> getWeightsMap();  ///< return a modifiable version of weights_		
 
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

		/**
		@brief Check that the path from input to output is not broken

		Note: The method will modify the model weights, nodes, and errors
			It is recommended to first create a copy of the model that will be later discarded
			Or re-initialize the model after.

		[DEPRECATED: params no longer needed]
		@param[in] input_nodes
		@param[out] output_nodes
		*/
		bool checkCompleteInputToOutput();

		/**
		@brief Check model link node and weight names

		[TODO: add tests...]

		@param[out] nodes_not_found
		@param[out] weights_not_found
		*/
		bool checkLinksNodeAndWeightNames(
			std::vector<std::string>& nodes_not_found,
			std::vector<std::string>& weights_not_found);

		/**
		@brief Remove hidden nodes that have either only 1 source and no sink connection
			or 1 sink and no source connection
		*/
		bool removeIsolatedNodes();

		/**
		@brief Convert model to adjacency list
		TODO: Implement tests

		@param[out] node_id_map Map of node id to node name
		@param[out] node_cnt the number of vertices in the adjacency list

		@returns An adjacency list representation of a graph
		*/
		std::list<int>* convertToAdjacencyList(std::map<int, std::string>& node_id_map, int& node_cnt);
		void findCycles();

		std::vector<std::pair<std::string, std::string>> getCyclicPairs();

private:
    int id_; ///< Model ID
    std::string name_; ///< Model Name
    std::map<std::string, std::shared_ptr<Link>> links_; ///< Model links
    std::map<std::string, std::shared_ptr<Node<TensorT>>> nodes_; ///< Model nodes
    std::map<std::string, std::shared_ptr<Weight<TensorT>>> weights_; ///< Model nodes
		std::vector<std::pair<std::string, std::string>> cyclic_pairs_;
		std::vector<std::shared_ptr<Node<TensorT>>> input_nodes_;
		std::vector<std::shared_ptr<Node<TensorT>>> output_nodes_;
  };
	template<typename TensorT>
	inline Model<TensorT>::Model(const Model<TensorT>& other)
	{
		id_ = other.id_;
		name_ = other.name_;
		addLinks(other.getLinks());
		addNodes(other.getNodes());
		addWeights(other.getWeights());
		cyclic_pairs_ = other.cyclic_pairs_;
	}

	template<typename TensorT>
	inline Model<TensorT>::Model(const int& id) :
		id_(id)
	{
	}

	template<typename TensorT>
	inline void Model<TensorT>::setId(const int& id)
	{
		id_ = id;
	}
	template<typename TensorT>
	inline int Model<TensorT>::getId() const
	{
		return id_;
	}

	template<typename TensorT>
	inline void Model<TensorT>::setName(const std::string& name)
	{
		name_ = name;
	}
	template<typename TensorT>
	inline std::string Model<TensorT>::getName() const
	{
		return name_;
	}

	template<typename TensorT>
	inline std::vector<std::shared_ptr<Node<TensorT>>> Model<TensorT>::getInputNodes()
	{
		return input_nodes_;
	}

	template<typename TensorT>
	inline std::vector<std::shared_ptr<Node<TensorT>>> Model<TensorT>::getOutputNodes()
	{
		return output_nodes_;
	}
	template<typename TensorT>
	inline std::vector<std::string> Model<TensorT>::getOutputNodeNames() const
	{
		std::vector<std::string> nodes;
		for (const auto& node : output_nodes_)
		{
			nodes.push_back(node->getName());
		}
		return nodes;
	}

	template<typename TensorT>
	inline void Model<TensorT>::addNodes(const std::vector<Node<TensorT>>& nodes)
	{
		for (const Node<TensorT>& node : nodes)
		{
			std::shared_ptr<Node<TensorT>> node_ptr;
			node_ptr.reset(new Node<TensorT>(node));
			auto found = nodes_.emplace(node.getName(), node_ptr);
			if (!found.second)
			{
				// TODO: move to debug log
				std::cout << "Node name " << node.getName() << " already exists!" << std::endl;
			}
			else {
				if (node.getType() == NodeType::input) {
					std::shared_ptr<Node<TensorT>> node_ptr_cpy = node_ptr;
					input_nodes_.push_back(node_ptr_cpy);
				}
				else if (node.getType() == NodeType::output) {
					std::shared_ptr<Node<TensorT>> node_ptr_cpy = node_ptr;
					output_nodes_.push_back(node_ptr);
				}
			}
		}
	}

	template<typename TensorT>
	inline Node<TensorT> Model<TensorT>::getNode(const std::string& node_name) const
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

	template<typename TensorT>
	inline std::vector<Node<TensorT>> Model<TensorT>::getNodes() const
	{
		std::vector<Node<TensorT>> nodes;
		for (const auto& node : nodes_)
		{
			nodes.push_back(*node.second);
		}
		return nodes;
	}

	template<typename TensorT>
	inline std::map<std::string, std::shared_ptr<Node<TensorT>>> Model<TensorT>::getNodesMap()
	{
		return nodes_;
	}

	template<typename TensorT>
	inline std::map<std::string, std::vector<std::string>> Model<TensorT>::getModuleNodeNameMap() const
	{
		std::map<std::string, std::vector<std::string>> module_to_node_names;
		for (const auto& node_map : nodes_) {
			std::vector<std::string> node_names = { node_map.first };
			auto found = module_to_node_names.emplace(node_map.second->getModuleName(), node_names);
			if (!found.second) {
				module_to_node_names.at(node_map.second->getModuleName()).push_back(node_map.first);
			}
		}
		return module_to_node_names;
	}

	template<typename TensorT>
	inline void Model<TensorT>::removeNodes(const std::vector<std::string>& node_names)
	{
		for (const std::string& node_name : node_names)
		{
			// check for duplicate nodes (by id)
			if (nodes_.count(node_name) != 0)
			{
				nodes_.erase(node_name);
			}
		}
		// pruneLinks(); // Allow for dangling links
	}

	template<typename TensorT>
	inline void Model<TensorT>::addWeights(const std::vector<Weight<TensorT>>& weights)
	{
		for (const Weight<TensorT>& weight : weights)
		{
			std::shared_ptr<Weight<TensorT>> weight_ptr;
			weight_ptr.reset(new Weight<TensorT>(weight));
			auto found = weights_.emplace(weight.getName(), weight_ptr);
			if (!found.second)
			{
				// TODO: move to debug log
				std::cout << "Weight name " << weight.getName() << " already exists!" << std::endl;
			}
		}
	}

	template<typename TensorT>
	inline Weight<TensorT> Model<TensorT>::getWeight(const std::string& weight_name) const
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

	template<typename TensorT>
	inline std::vector<Weight<TensorT>> Model<TensorT>::getWeights() const
	{
		std::vector<Weight<TensorT>> weights;
		for (const auto& weight : weights_)
		{
			weights.push_back(*weight.second);
		}
		return weights;
	}

	template<typename TensorT>
	inline std::map<std::string, std::shared_ptr<Weight<TensorT>>> Model<TensorT>::getWeightsMap()
	{
		return weights_;
	}

	template<typename TensorT>
	inline void Model<TensorT>::removeWeights(const std::vector<std::string>& weight_names)
	{
		for (std::string const& weight_name : weight_names)
		{
			// check for duplicate weights (by id)
			if (weights_.count(weight_name) != 0)
			{
				weights_.erase(weight_name);
			}
		}
		pruneLinks();
	}

	template<typename TensorT>
	inline void Model<TensorT>::addLinks(const std::vector<Link>& links)
	{
		for (const Link& link : links)
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

	template<typename TensorT>
	inline void Model<TensorT>::removeLinks(const std::vector<std::string>& link_names)
	{
		for (const std::string& link_name : link_names)
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

	template<typename TensorT>
	inline Link Model<TensorT>::getLink(const std::string& link_name) const
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

	template<typename TensorT>
	inline std::vector<Link> Model<TensorT>::getLinks() const
	{
		std::vector<Link> links;
		for (const auto& link : links_)
		{
			links.push_back(*link.second);
		}
		return links;
	}
	
	template<typename TensorT>
	inline std::map<std::string, std::shared_ptr<Link>> Model<TensorT>::getLinksMap()
	{
		return links_;
	}

	template<typename TensorT>
	inline bool Model<TensorT>::pruneNodes()
	{
		std::vector<std::string> node_names;
		if (nodes_.empty()) { return false; }
		for (const auto& node : nodes_)
		{
			bool found = false;
			// if (links_.empty()) { found = true; }
			for (const auto& link : links_)
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

	template<typename TensorT>
	inline bool Model<TensorT>::pruneWeights()
	{
		std::vector<std::string> weight_names;
		if (weights_.empty()) { return false; }
		for (const auto& weight : weights_)
		{
			bool found = false;
			// if (links_.empty()) { found = true; }
			for (const auto& link : links_)
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

	template<typename TensorT>
	inline bool Model<TensorT>::pruneLinks()
	{
		std::vector<std::string> link_names;
		if (links_.empty()) { return false; }
		for (const auto& link : links_)
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

	template<typename TensorT>
	inline void Model<TensorT>::pruneModel(int iterations)
	{
		try
		{
			int cnt = 0;
			while (pruneLinks() || pruneWeights() || pruneNodes())
			{
				if (cnt >= iterations) { break; }
				// std::cout<<"Pruning model iteration: "<<cnt<<std::endl;
				cnt += 1;
			}
		}
		catch (std::exception& e)
		{
			printf("Exception: %s", e.what());
		}
	}
	
	template<typename TensorT>
	inline bool Model<TensorT>::checkNodeNames(const std::vector<std::string> node_names)
	{
		bool nodes_found = true;
		for (const std::string& node_name : node_names)
		{
			if (nodes_.empty() || nodes_.count(node_name) == 0)
			{
				nodes_found = false;
				std::cout << "Node name " << node_name << " not found!" << std::endl;
			}
		}
		return nodes_found;
	}

	template<typename TensorT>
	inline bool Model<TensorT>::checkLinkNames(const std::vector<std::string> link_names)
	{
		bool links_found = true;
		for (const std::string& link_name : link_names)
		{
			if (links_.empty() || links_.count(link_name) == 0)
			{
				links_found = false;
				std::cout << "Link name " << link_name << " not found!" << std::endl;
			}
		}
		return links_found;
	}

	template<typename TensorT>
	inline bool Model<TensorT>::checkWeightNames(const std::vector<std::string> weight_names)
	{
		bool weights_found = true;
		for (const std::string& weight_name : weight_names)
		{
			if (weights_.empty() || weights_.count(weight_name) == 0)
			{
				weights_found = false;
				std::cout << "Weight name " << weight_name << " not found!" << std::endl;
			}
		}
		return weights_found;
	}

	template<typename TensorT>
	inline bool Model<TensorT>::checkCompleteInputToOutput()
	{
		return true;
	}

	template<typename TensorT>
	inline bool Model<TensorT>::checkLinksNodeAndWeightNames(std::vector<std::string>& nodes_not_found, std::vector<std::string>& weights_not_found)
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

	template<typename TensorT>
	inline bool Model<TensorT>::removeIsolatedNodes()
	{
		// key/value pair of node name and source/sink count pair
		std::map<std::string, std::pair<int, int>> node_counts;

		// count all sink/source connections for each node
		for (const auto& link_map : links_)
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
		for (const auto& node_count : node_counts)
		{
			if (node_count.second.first == 0 || node_count.second.second == 0)
			{
				removeNodes({ node_count.first });
				dead_end_node_found = true;
			}
		}
		return dead_end_node_found;
	}

	template<typename TensorT>
	inline std::list<int>* Model<TensorT>::convertToAdjacencyList(std::map<int, std::string>& node_id_map, int& node_cnt)
	{
		// create a map of node id to node name (excluding bias nodes)
		node_id_map.clear();
		node_cnt = 0;
		for (auto& node_map : nodes_) {
			if (node_map.second->getType() != NodeType::bias) {
				++node_cnt;
				node_map.second->setId(node_cnt);
				node_id_map.emplace(node_cnt, node_map.first);
			}
			else {
				node_map.second->setId(-1);
			}
		}

		// create the DFS trees (excluding bias nodes)
		std::list<int> *adj;
		adj = new std::list<int>[node_cnt];

		// add the actual nodes
		for (auto& link_map : links_)
			if (nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias)
				adj[nodes_.at(link_map.second->getSourceNodeName())->getId() - 1].push_back(nodes_.at(link_map.second->getSinkNodeName())->getId());

		return adj;
	}

	template<typename TensorT>
	inline void Model<TensorT>::findCycles()
	{
		std::map<int, std::string> node_id_map;
		int node_cnt;
		std::list<int> *adj = convertToAdjacencyList(node_id_map, node_cnt);

		CircuitFinder CF(adj, node_cnt);
		CF.run();

		cyclic_pairs_.clear();
		for (const auto& source_sink : CF.getCycles()) {
			if (nodes_.at(node_id_map.at(source_sink.second))->getType() == NodeType::recursive) // enforce order of recursive nodes
				cyclic_pairs_.push_back(std::make_pair(node_id_map.at(source_sink.second), node_id_map.at(source_sink.first)));
			else
				cyclic_pairs_.push_back(std::make_pair(node_id_map.at(source_sink.first), node_id_map.at(source_sink.second)));
		}
	}

	template<typename TensorT>
	inline std::vector<std::pair<std::string, std::string>> Model<TensorT>::getCyclicPairs()
	{
		return cyclic_pairs_;
	}
}

#endif //SMARTPEAK_MODEL_H