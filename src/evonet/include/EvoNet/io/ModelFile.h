/**TODO:  Add copyright*/

#ifndef EVONET_MODELFILE_H
#define EVONET_MODELFILE_H

// .h
#include <EvoNet/ml/Model.h>
#include <iostream>
#include <fstream>
#include <vector>

// .cpp
#include <EvoNet/io/NodeFile.h>
#include <EvoNet/io/WeightFile.h>
#include <EvoNet/io/LinkFile.h>

//#include <filesystem> // C++ 17

#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp> // std::pair
#include <cereal/types/vector.hpp>
#include <cereal/types/set.hpp>
#include <cereal/archives/binary.hpp>

namespace EvoNet
{
  /**
    @brief ModelFile
  */
	template<typename TensorT>
  class ModelFile
  {
public:
    ModelFile() = default; ///< Default constructor
    ~ModelFile() = default; ///< Default destructor
 
		/**
			@brief store Model from file

			@param filename The name of the model file
			@param model The model to store

			@returns Status True on success, False if not
		*/
		bool storeModelBinary(const std::string& filename, const Model<TensorT>& model);
 
		/**
			@brief load Model from file

			@param filename The name of the model file
			@param model The model to load data into

			@returns Status True on success, False if not
		*/
		bool loadModelBinary(const std::string& filename, Model<TensorT>& model);

    /**
      @brief load Model weights from a binarized model file file

      @param filename The name of the model file
      @param weights The weights to load data into

      @returns Status True on success, False if not
    */
    bool loadWeightValuesBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights);

		/**
			@brief store nodes, links, and weights as a .csv file from a Model

			@param filename_nodes The name of the node file
			@param filename_links The name of the link file
			@param filename_weights The name of the weight file
			@param model The model to load data into

			@returns Status True on success, False if not
		*/
		bool storeModelCsv(const std::string& filename_nodes, const std::string& filename_links, const std::string& filename_weights, Model<TensorT>& model,
			bool store_nodes = true, bool store_links = true, bool store_weights = true);

		/**
			@brief Load nodes, links, and weights from file and create a Model

			@param filename_nodes The name of the node file
			@param filename_links The name of the link file
			@param filename_weights The name of the weight file
			@param model The model to load data into

			@returns Status True on success, False if not
		*/
		bool loadModelCsv(const std::string& filename_nodes, const std::string& filename_links, const std::string& filename_weights, Model<TensorT>& model,
			bool load_nodes = true, bool load_links = true, bool load_weights = true);

		/**
		@brief save network model to file in dot format for visualization
			using e.g., GraphVIZ

		[TODO: move to GraphFile and take in the model as input
			to allow for the following
			1. coloring of nodes based on node type (i.e., input, hidden, bias, or output)
				e.g. node1 [shape=circle,style=filled,color=".7 .3 1.0"];
			2. annotation of links with the value of the weight
				e.g. node1 -> node2 [style=italic,label="weight = 10"];
		]

		@param filename The name of the links file (.gv extension)
		@param links The links to save to disk

		@returns Status True on success, False if not
		*/
		bool storeModelDot(const std::string& filename, const Model<TensorT>& model);
  };

	template<typename TensorT>
	bool ModelFile<TensorT>::storeModelBinary(const std::string & filename, const Model<TensorT>& model)
	{
		std::ofstream ofs(filename, std::ios::binary);  
		//if (ofs.is_open() == false) {// Lines check to make sure the file is not already created
		cereal::BinaryOutputArchive oarchive(ofs); 
		oarchive(model); 
		ofs.close();
		//}// Lines check to make sure the file is not already created
		return true;
	}

	template<typename TensorT>
	bool ModelFile<TensorT>::loadModelBinary(const std::string & filename, Model<TensorT>& model)
	{		
		std::ifstream ifs(filename, std::ios::binary); 
		if (ifs.is_open()) {
			cereal::BinaryInputArchive iarchive(ifs);
			iarchive(model);
			ifs.close();
		}
    else {
      std::cout << "The model with filename " + filename + " was not found." << std::endl;
    }
		return true;
	}

  template<typename TensorT>
  inline bool ModelFile<TensorT>::loadWeightValuesBinary(const std::string & filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights)
  {
    // Load in the binarized model
    Model<TensorT> model;
    loadModelBinary(filename, model);

    // Transer over the weights
    for (auto& weight_map: weights){
      // parse the weight value
      try
      {
        weight_map.second->setWeight(model.weights_.at(weight_map.first)->getWeight());
        weight_map.second->setInitWeight(false);
      }
      catch (std::exception& e)
      {
        printf("Exception: %s", e.what());
      }
    }
    return true;
  }

	template<typename TensorT>
	bool ModelFile<TensorT>::storeModelCsv(const std::string & filename_nodes, const std::string & filename_links, const std::string & filename_weights, Model<TensorT>& model,
		bool store_nodes, bool store_links, bool store_weights)
	{
		// [PERFORMANCE: this can be parallelized using threads]
		if (store_nodes) {
			NodeFile<TensorT> node_file;
			node_file.storeNodesCsv(filename_nodes, model.nodes_);
		}
		if (store_links) {
			LinkFile link_file;
			link_file.storeLinksCsv(filename_links, model.links_);
		}
		if (store_weights) {
			WeightFile<TensorT> weight_file;
			weight_file.storeWeightsCsv(filename_weights, model.weights_);
		}
		return true;
	}

	template<typename TensorT>
	bool ModelFile<TensorT>::loadModelCsv(const std::string & filename_nodes, const std::string & filename_links, const std::string & filename_weights, Model<TensorT>& model,
		bool load_nodes, bool load_links, bool load_weights)
	{
		// [PERFORMANCE: this can be parallelized using threads]
		// load the nodes
		if (load_nodes) {
			NodeFile<TensorT> node_file;
			std::map<std::string, std::shared_ptr<Node<TensorT>>> nodes;
			node_file.loadNodesCsv(filename_nodes, nodes);
			model.nodes_ = nodes;
      model.setInputAndOutputNodes(); // Need to initialize the input/output node cache
		}

		// load the links
		if (load_links) {
			LinkFile link_file;
			std::map<std::string, std::shared_ptr<Link>> links;
			link_file.loadLinksCsv(filename_links, links);
			model.links_ = links;
		}

		// load the weights
		if (load_weights) {
			WeightFile<TensorT> weight_file;
			std::map<std::string, std::shared_ptr<Weight<TensorT>>> weights;
			weight_file.loadWeightsCsv(filename_weights, weights);
			model.weights_ = weights;
		}

		return true;
	}

	template<typename TensorT>
	bool ModelFile<TensorT>::storeModelDot(const std::string& filename, const Model<TensorT>& model)
	{
		std::fstream file;
		// Open the file in truncate mode
		file.open(filename, std::ios::out | std::ios::trunc);

		file << "digraph G {\n"; // first line

		// write node formating to file
		for (const Node<TensorT>& node : model.getNodes())
		{
			if (node.getType() == NodeType::input)
			{
				char line_char[512];
				sprintf(line_char, "\t\"%s\" [shape=circle,style=filled,color=\"#D3D3D3\"];\n", node.getName().data());
				std::string line(line_char);
				file << line;
			}
			else if (node.getType() == NodeType::output)
			{
				char line_char[512];
				sprintf(line_char, "\t\"%s\" [shape=circle,style=filled,color=\"#00FFFF\"];\n", node.getName().data());
				std::string line(line_char);
				file << line;
			}
		}

		// write each source/sink to file
		for (const Link& link : model.getLinks())
		{
			if (model.getNode(link.getSourceNodeName()).getType() != NodeType::bias)
			{
				char line_char[512];
				sprintf(line_char, "\t\"%s\" -> \"%s\";\n", link.getSourceNodeName().data(), link.getSinkNodeName().data());
				std::string line(line_char);
				file << line;
			}
		}

		file << "}";  // last line
		file.close();

		return true;
	}
}

#endif //EVONET_MODELFILE_H