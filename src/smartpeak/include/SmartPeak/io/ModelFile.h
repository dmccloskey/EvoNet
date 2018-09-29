/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELFILE_H
#define SMARTPEAK_MODELFILE_H

#include <SmartPeak/ml/Model.h>

#include <iostream>
#include <fstream>
#include <vector>

namespace SmartPeak
{

  /**
    @brief ModelFile
  */
  class ModelFile
  {
public:
    ModelFile(); ///< Default constructor
    ~ModelFile(); ///< Default destructor
 
		/**
			@brief store Model from file

			@param filename The name of the model file
			@param model The model to store

			@returns Status True on success, False if not
		*/
		bool storeModelBinary(const std::string& filename, const Model& model);
 
		/**
			@brief load Model from file

			@param filename The name of the model file
			@param model The model to load data into

			@returns Status True on success, False if not
		*/
		bool loadModelBinary(const std::string& filename, Model& model);

		/**
			@brief Load nodes, links, and weights from file and create a Model

			@param filename_nodes The name of the node file
			@param filename_links The name of the link file
			@param filename_weights The name of the weight file
			@param model The model to load data into

			@returns Status True on success, False if not
		*/
		bool loadModelCsv(const std::string& filename_nodes, const std::string& filename_links, const std::string& filename_weights, Model& model);

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
		bool storeModelDot(const std::string& filename, const Model& model);
  };
}

#endif //SMARTPEAK_MODELFILE_H