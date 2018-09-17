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
		@brief Add inputs nodes to an empty model

		@param[in, out] model An empty model
		@param[in] names Prefix name to use for the nodes
		@param[in] n_nodes The number of output nodes

		@returns vector of output node names
		*/
		std::vector<std::string> addInputNodes(Model& model, const std::string& name, const int& n_nodes);

		/**
		@brief Add a fully connected layer to a model

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the fully connected layer to
		@param[in] n_nodes The number of output nodes
    @param[in] node_activation The activation function of the hidden node to create
    @param[in] node_activation_grad The activation function gradient of the hidden node to create
    @param[in] node_integration The integration function of the hidden node to create
		@param[in] drop_out_prob Node drop out probability
		@param[in] drop_connection_prob Weight drop out probability
		@param[in] biases Whether to include bias nodes or not

		@returns vector of output node names
		*/
		std::vector<std::string> addFullyConnected(Model& model, const std::string& name, const std::string& module_name, 
			const std::vector<std::string>& source_node_names, const int& n_nodes,
			const std::shared_ptr<ActivationOp<float>>& node_activation,
			const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<float>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			float drop_out_prob = 0.0f, float drop_connection_prob = 0.0f, bool biases = true);

		/**
		@brief Add a Soft Max

		def stable_softmax(X):
		exps = np.exp(X)
		return exps / np.sum(exps)

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to

		@returns vector of output node names
		*/
		std::vector<std::string> addSoftMax(Model& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& source_node_names);

		/**
		@brief Add a Stable Soft Max

		def stable_softmax(X):
			exps = np.exp(X - np.max(X))
			return exps / np.sum(exps)

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to

		@returns vector of output node names
		*/
		std::vector<std::string> addStableSoftMax(Model& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& source_node_names);

		/**
		@brief Add a Convolution layer or Pooling layer

		The input is considered a linearized matrix in column order
		The output is considered a linearized matrix in column order

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to
		@param[in] input_width The width of the input
		@param[in] input_height The height of the input
		@param[in] input_width_zero_padding Add 0s to the left and right of the input
		@param[in] input_height_zero_padding Add 0s to the top and bottom of the input
		@param[in] depth The number of convolution filters
		@param[in] extent_width The width of the filter
		@param[in] extent_height The height of the filter
		@param[in] stride The spacing between filters
		@param[in] output_width_zero_padding Add 0s to the left and right of the output
		@param[in] output_height_zero_padding Add 0s to the top and bottom of the output

		@returns vector of output node names
		*/
		std::vector<std::string> addConvolution(Model & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names,
			const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
			const int & extent_width, const int & extent_height, const int & stride,
			const int & output_width_zero_padding, const int& output_height_zero_padding,
			const std::shared_ptr<ActivationOp<float>>& node_activation,
			const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<float>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			float drop_out_prob = 0.0f, float drop_connection_prob = 0.0f, bool biases = true);

		/**
		@brief Add a normalization layer with activation

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the fully connected layer to
		@param[in] n_nodes The number of output nodes
		@param[in] node_activation The activation function of the hidden node to create
		@param[in] node_activation_grad The activation function gradient of the hidden node to create
		@param[in] node_integration The integration function of the hidden node to create
		@param[in] drop_out_prob Node drop out probability
		@param[in] drop_connection_prob Weight drop out probability
		@param[in] biases Whether to include bias nodes or not

		@returns vector of output node names
		*/
		std::vector<std::string> addNormalization(Model& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const std::shared_ptr<ActivationOp<float>>& node_activation,
			const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			float drop_out_prob = 0.0f, float drop_connection_prob = 0.0f, bool biases = true);

		/**
		@brief Add a VAE Encoding layer with input node

		@param[in, out] Model
		@param[in] mu_node_names Node_names from the average layer
		@param[in] logvar_node_names Nodes names from the logvar layer

		@returns vector of output node names
		*/
		std::vector<std::string> addVAEEncoding(Model& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names);

		/**
		@brief Add a LSTM layer

		Reference:
		Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to
		@param[in] n_blocks The number of independent LSTM cell blocks
		@param[in] n_cells The number of shared memory cells per LSTM block
    @param[in] node_activation The activation function of the input node to create
    @param[in] node_activation_grad The activation function gradient of the input node to create
    @param[in] node_integration The integration function of the input node to create
    @param[in] node_integration_error The integration function of the input node to create
    @param[in] node_integration_weight_grad The integration function of the input node to create
		@param[in] drop_out_prob input or output Node drop out probability
		@param[in] drop_connection_prob input or output Weight drop out probability
		@param[in] biases Whether to include bias nodes or not

		@returns vector of output node names
		*/
		std::vector<std::string> addLSTM(Model& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_blocks, const int& n_cells,
			const std::shared_ptr<ActivationOp<float>>& node_activation,
			const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<float>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			float drop_out_prob = 0.0f, float drop_connection_prob = 0.0f, bool biases = true);
		std::vector<std::string> addLSTMBlock(Model& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_cells,
			const std::shared_ptr<ActivationOp<float>>& node_activation,
			const std::shared_ptr<ActivationOp<float>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<float>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<float>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<float>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			float drop_out_prob = 0.0f, float drop_connection_prob = 0.0f, bool biases = true);

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