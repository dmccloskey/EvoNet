/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELBUILDER_H
#define SMARTPEAK_MODELBUILDER_H

#include <SmartPeak/ml/Model.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <tuple>

namespace SmartPeak
{

  /**
    @brief Class to help create complex network models
  */
  class ModelBuilder
  {
public:
    ModelBuilder() = default; ///< Default constructor
    ~ModelBuilder() = default; ///< Default destructor

		/**
		@brief Add a fully connected layer to a model

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the fully connected layer to
		@param[in] n_nodes The number of output nodes

		@returns vector of output node names
		*/
		std::vector<std::string> addFullyConnected(Model& model, const std::string& name, const std::vector<std::string>& source_node_names, const int& n_nodes);

		/**
		@brief Add a Soft Max

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to
		@param[in] n_nodes The number of output nodes

		@returns vector of output node names
		*/
		std::vector<std::string> addSoftMax(Model& model, const std::string& name, const std::vector<std::string>& source_node_names, const int& n_nodes);

		/**
		@brief Add a Convolution layer

		The input is considered a linearized matrix in column order
		The output is considered a linearized matrix in column order

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to
		@param[in] input_width The width of the input
		@param[in] input_height The height of the input
		@param[in] depth The number of convolution filters
		@param[in] extent_width The width of the filter
		@param[in] extent_height The height of the filter
		@param[in] stride The spacing between filters
		@param[in] zero_padding

		@returns vector of output node names
		*/
		std::vector<std::string> addConvolution(Model& model, const std::string& name, const std::vector<std::string>& source_node_names, 
			const int& input_width, const int& input_height,
			const int& extent_width, const int& extent_height,
			const int& depth, const int& stride, const int& zero_padding);

		/**
		@brief Add a Convolution layer

		The input is considered a linearized matrix in column order
		The output is considered a linearized matrix in column order

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to
		@param[in] input_width The width of the input
		@param[in] input_height The height of the input
		@param[in] depth The number of convolution filters
		@param[in] extent_width The width of the filter
		@param[in] extent_height The height of the filter
		@param[in] stride The spacing between filters

		@returns vector of output node names
		*/
		std::vector<std::string> addPooling(Model& model, const std::string& name, const std::vector<std::string>& source_node_names,
			const int& input_width, const int& input_height,
			const int& extent_width, const int& extent_height,
			const int& stride);

		/**
		@brief Add a LSTM layer

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to
		@param[in] n_hidden The number of LSTM hidden states

		@returns vector of output node names
		*/
		std::vector<std::string> addPooling(Model& model, const std::string& name, const std::vector<std::string>& source_node_names,
			const int& n_hidden);

		/**
		@brief Add one model to another

		@param[in, out] Model
		@param[in] source_node_names Node_names in the LH model to add to
		@param[in] sink_node_names Node names in the RH model to join
		@param[in] model_rh The RH model to add to the LH model

		@returns vector of output node names
		*/
		std::vector<std::string> addModel(Model& model, const std::vector<std::string>& source_node_names,
			const std::vector<std::string>& sink_node_names, const Model& model_rh);
  };
}

#endif //SMARTPEAK_MODELBUILDER_H