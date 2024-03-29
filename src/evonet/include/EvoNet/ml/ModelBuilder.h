﻿/**TODO:  Add copyright*/

#ifndef EVONET_MODELBUILDER_H
#define EVONET_MODELBUILDER_H

// .h
#include <EvoNet/ml/Model.h>

#include <unsupported/Eigen/CXX11/Tensor>

// .cpp
#include <EvoNet/core/Preprocessing.h>

namespace EvoNet
{

  /**
    @brief Class to help create complex network models

		NOTE: the ModelInterpreter class arranges the Tensor layers according to node name ascending order.
			Therefore, the node name indices are buffered with 0's of length 12 to ensure proper sorting of
			nodes within a tensor layer.
  */
	template<typename TensorT>
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
		std::vector<std::string> addInputNodes(Model<TensorT>& model, const std::string& name, const std::string & module_name, const int& n_nodes, const bool& specify_layer = false);

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
		std::vector<std::string> addFullyConnected(Model<TensorT>& model, const std::string& name, const std::string& module_name, 
			const std::vector<std::string>& source_node_names, const int& n_nodes,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, const bool& specify_layer = false);
		void addFullyConnected(Model<TensorT>& model, const std::string& module_name,
			const std::vector<std::string>& source_node_names, const std::vector<std::string>& sink_node_names,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver, 
			const TensorT& drop_connection_prob = 0.0f, const bool& specify_layer = false);

		/**
		@brief Add a singly connected layer to a model

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the singly connected layer to
		@param[in] n_nodes The number of output nodes
		@param[in] node_activation The activation function of the hidden node to create
		@param[in] node_activation_grad The activation function gradient of the hidden node to create
		@param[in] node_integration The integration function of the hidden node to create
		@param[in] node_integration_error The integration error function of the hidden node to create
		@param[in] node_integration_weight_grad The integration weight gradient function of the hidden node to create
		@param[in] weight_init The weight initialization function
		@param[in] solver The weight solver
		@param[in] drop_out_prob Node drop out probability
		@param[in] drop_connection_prob Weight drop out probability
		@param[in] biases Whether to include bias nodes or not

		@returns vector of output node names
		*/
		std::vector<std::string> addSinglyConnected(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names, const int& n_nodes,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, const bool& specify_layer = false);
		void addSinglyConnected(Model<TensorT>& model, const std::string& module_name,
			const std::vector<std::string>& source_node_names, const std::vector<std::string>& sink_node_names,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_connection_prob = 0.0f, const bool& specify_layer = false);

    /*
    @brief Add biases

    A special case of `addSinglyConnected` where the source nodes are biases

		@param[in, out] Model
		@param[in] module_name The module name used when specifying layers
		@param[in] sink_node_names Node_names to add the singly connected layer to
		@param[in] weight_init The weight initialization function
		@param[in] solver The weight solver
		@param[in] drop_connection_prob Weight drop out probability
		@param[in] specify_layer Manually specify the layer that the node should be placed on

		@returns vector of output node names
    */
    std::vector<std::string> addBiases(Model<TensorT>& model, const std::string& module_name, const std::vector<std::string>& sink_node_names,
      const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
      const TensorT& drop_connection_prob = 0.0f, const bool& specify_layer = false);

    /**
    @brief Add hidden nodes

    @param[in, out] Model
    @param[in] n_nodes The number of output nodes
    @param[in] node_activation The activation function of the hidden node to create
    @param[in] node_activation_grad The activation function gradient of the hidden node to create
    @param[in] node_integration The integration function of the hidden node to create
    @param[in] node_integration_error The integration error function of the hidden node to create
    @param[in] node_integration_weight_grad The integration weight gradient function of the hidden node to create
    @param[in] drop_out_prob Node drop out probability
    @param[in] biases Whether to include bias nodes or not

    @returns vector of output node names
    */
    std::vector<std::string> addHiddenNodes(Model<TensorT>& model, const std::string& name, const std::string& module_name, const int& n_nodes,
      const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
      const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
      const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
      const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
      const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad, 
      const std::shared_ptr<SolverOp<TensorT>>& solver,
      const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, const bool& specify_layer = false);

		/**
		@brief Add a Soft Max

		def stable_softmax(X):
		exps = np.exp(X)
		return exps / np.sum(exps)

		@param[in, out] Model<TensorT>
		@param[in] source_node_names Node_names to add the layer to

		@returns vector of output node names
		*/
		std::vector<std::string> addSoftMax(Model<TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& source_node_names, const bool& specify_layer = false);

		/**
		@brief Add a Stable Soft Max

		def stable_softmax(X):
			exps = np.exp(X - np.max(X))
			return exps / np.sum(exps)

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to

		@returns vector of output node names
		*/
		std::vector<std::string> addStableSoftMax(Model<TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& source_node_names,
			const bool& specify_layer = false);

		/**
		@brief Add a Convolution layer or Pooling layer

		The input is considered a linearized matrix in column order
		The output is considered a linearized matrix in column order

		BUG: addition of bias causes an odd bug in model interpreter

		Overload is provided to add additional filters that operate over the same
		input and output nodes

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
		std::vector<std::string> addConvolution(Model<TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names,
			const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
			const int & extent_width, const int & extent_height, const int & stride,
			const int & output_width_zero_padding, const int& output_height_zero_padding,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool split_filter_layers = true, bool share_weights = true);
		void addConvolution(Model<TensorT> & model, const std::string & name, const std::string& module_name, 
			const std::vector<std::string>& source_node_names,
			const std::vector<std::string>& output_node_names,
			const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
			const int & extent_width, const int & extent_height, const int & stride,
			const int & output_width_zero_padding, const int& output_height_zero_padding,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, bool split_filter_layers = true);

		/**
		@brief Add a Projection layer (i.e., inverse convolution)

		The input is considered a linearized matrix in column order
		The output is considered a linearized matrix in column order

		BUG: addition of bias causes an odd bug in model interpreter

		Overload is provided to add additional filters that operate over the same
		input and output nodes

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
		std::vector<std::string> addProjection(Model<TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names,
			const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
			const int & extent_width, const int & extent_height, const int & stride,
			const int & output_width_zero_padding, const int& output_height_zero_padding,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool split_filter_layers = true, bool share_weights = true);
		void addProjection(Model<TensorT> & model, const std::string & name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const std::vector<std::string>& output_node_names,
			const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
			const int & extent_width, const int & extent_height, const int & stride,
			const int & output_width_zero_padding, const int& output_height_zero_padding,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, bool split_filter_layers = true);

		/**
		@brief Add a normalization layer with activation.
       If a learnable gain/offset or application of an activation is desired,
       the user can add a singly connected layer after the unit scale layer.

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the fully connected layer to
		@param[in] node_activation The activation function of the hidden node to create
		@param[in] node_activation_grad The activation function gradient of the hidden node to create
		@param[in] node_integration The integration function of the hidden node to create
		@param[in] drop_out_prob Node drop out probability
		@param[in] drop_connection_prob Weight drop out probability
		@param[in] biases Whether to include bias nodes or not

		@returns vector of output node names
		*/
		std::vector<std::string> addNormalization(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names, const bool& specify_layers = false);

    /**
    @brief Add a unit scale layer.
       If a learnable gain/offset or application of an activation is desired,
       the user can add a singly connected layer after the unit scale layer.

    @param[in, out] Model
    @param[in] source_node_names Node_names to add the fully connected layer to

    @returns vector of output node names
    */
    std::vector<std::string> addUnitScale(Model<TensorT>& model, const std::string& name, const std::string& module_name,
      const std::vector<std::string>& source_node_names, const bool& specify_layers = false);

    /**
    @brief Add a unit scale layer.
       If a learnable gain/offset or application of an activation is desired,
       the user can add a singly connected layer after the unit scale layer.

    @param[in, out] Model
    @param[in] source_node_names Node_names to add the fully connected layer to
    @param[in] range_min The minimum value to project to
    @param[in] range_max The maximum value to project to

    @returns vector of output node names
    */
    std::vector<std::string> addLinearScale(Model<TensorT>& model, const std::string& name, const std::string& module_name,
      const std::vector<std::string>& source_node_names, const TensorT& range_min, const TensorT& range_max, const bool& specify_layers = false);

		/**
		@brief Add a VAE Encoding layer for a gaussian distribution with input node

		Note:
			Input node names generated by the method are the following
			"%s_%012d-Sampler" where "%s" is filled by the `name` argument

		@param[in, out] Model
		@param[in] mu_node_names Node_names from the average layer
		@param[in] logvar_node_names Nodes names from the logvar layer

		@returns vector of output node names
		*/
		std::vector<std::string> addGaussianEncoding(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names, const bool& specify_layer = false);

		/**
		@brief Add a VAE Encoding layer for a Gumble/concrete categorical distribution with input node

		The categorical distribution is calculated as the following:
			yi = [exp((log(alphai) + gi)/tau)] / [SUM j=1 to N exp((log(alphaj) + gj)/tau)]; for i = 1; ...; k;
			with parameters alpha, sampled Gumbel values g, and temperature tau
			where the Gumbel(0; 1) distribution can be sampled using inverse transform sampling by drawing u 
				Uniform(0; 1) and computing g = -log(-log(u)).

		References:
			Maddison 2017 The concrete distribution
			Jang 2017 Categorical reparameterization with Gumbel-softmax

		Generated input node generated by the method are the following:
			"%s_%012d-InverseTau" (input values specified from 0 to inf for 1/tau) where "%s" is filled by the `name` argument
			"%s_%012d-GumbelSampler" (Gumbel sampled values) where "%s" is filled by the `name` argument

		@param[in, out] Model
		@param[in] alpha_node_names Nodes names from the catergorical logit layer

		@returns vector of output node names
		*/
		std::vector<std::string> addCategoricalEncoding(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& alpha_node_names, const bool& specify_layer = false);

		/**
		@brief Add a VAE Encoding layer with input node

		@param[in, out] Model
		@param[in] encoding_node_names Node_names for the latent distribution

		@returns vector of output node names
		*/
		std::vector<std::string> addDiscriminator(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& encoding_node_names);

		/**
		@brief Add a LSTM layer

		Reference:
		1. Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.
		2. Gers, F. A.; Schmidhuber, J. (2001). "LSTM Recurrent Networks Learn Simple Context Free and Context Sensitive Languages" (PDF). IEEE Transactions on Neural Networks. 12 (6): 1333–1340. doi:10.1109/72.963769.


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
		@param[in] forget_gat Whether to include forget gates or not
		@param[in] block_version
			1 Traditional: output multiplier is connected to block input and block gates
			2 Peep holes: memory cell is connected to block gates

		@returns vector of output node names
		*/
		std::vector<std::string> addLSTM(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_blocks, const int& n_cells,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true,
			bool forget_gate = true, int block_version = 1, const bool& specify_layer = false, bool specify_cyclic_pairs = false);
		std::vector<std::string> addLSTMBlock1(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_cells,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true,
			bool forget_gate = true, const bool& specify_layer = false, bool specify_cyclic_pairs = false);
		std::vector<std::string> addLSTMBlock2(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_cells,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true,
			bool forget_gate = true, const bool& specify_layer = false, bool specify_cyclic_pairs = false);

		/**
		@brief Add a GRU layer

		Reference:
		1. Cho, Kyunghyun; van Merrienboer, Bart; Gulcehre, Caglar; Bahdanau, Dzmitry; Bougares, Fethi; Schwenk, Holger; Bengio, Yoshua (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation". arXiv:1406.1078
		2. Zhou, G., Wu, J., Zhang, C., Zhou, Z. Minimal Gated Unit for Recurrent	Neural Networks.arXiv preprint arXiv : 1603.09420v1, 2016.

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to
		@param[in] n_blocks The number of independent GRU cell blocks
		@param[in] node_activation The activation function of the input node to create
		@param[in] node_activation_grad The activation function gradient of the input node to create
		@param[in] node_integration The integration function of the input node to create
		@param[in] node_integration_error The integration function of the input node to create
		@param[in] node_integration_weight_grad The integration function of the input node to create
		@param[in] drop_out_prob input or output Node drop out probability
		@param[in] drop_connection_prob input or output Weight drop out probability
		@param[in] biases Whether to include bias nodes or not
		@param[in] input_gate_connection Whether to include an input connection to the gates
		@param[in] block_version
			1 GRU: input and reset gates
			2 MGRU: forget gate

		@returns vector of output node names
		*/
		std::vector<std::string> addGRU(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_blocks, 
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true,
			bool forget_gate = true, int block_version = 1, const bool& specify_layer = false);
		std::vector<std::string> addGRU1(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool input_gate_connection = true, const bool& specify_layer = false);
		std::vector<std::string> addGRU2(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool input_gate_connection = true, const bool& specify_layer = false);

		/**
		@brief Add a dot product self attention layer with activation

		References:
		Vaswani, et al. 2017 Attention is all you need

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the fully connected layer to
		...
		@param[in] n_nodes The number of output nodes
		@param[in] node_activation The activation function of the hidden node to create
		@param[in] node_activation_grad The activation function gradient of the hidden node to create
		@param[in] node_integration The integration function of the hidden node to create
		@param[in] drop_out_prob Node drop out probability
		@param[in] drop_connection_prob Weight drop out probability
		@param[in] biases Whether to include bias nodes or not

		@returns vector of output node names
		*/
		std::vector<std::string> addMultiHeadAttention(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& query_node_names, const std::vector<std::string>& key_node_names, const std::vector<std::string>& values_node_names,
			const int& n_heads, const std::string& attention_type, const int & model_length, const int& key_length, const int& values_length,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool split_attention_layers = true);

		/**
		@brief Add a scaled dot product self attention layer with activation

		References:
		Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
			Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv preprint arXiv:1706.03762,
			2017.
		*/
		std::vector<std::string> addDotProdAttention(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& query_node_names, const std::vector<std::string>& key_node_names, const std::vector<std::string>& values_node_names,
			const int& key_length, const int& values_length,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool split_attention_layers = true);

		/**
		@brief Add an additive attention layer with activation

		References:
		Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly
			learning to align and translate. CoRR, abs/1409.0473, 2014.
		*/
		std::vector<std::string> addAdditiveAttention(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& query_node_names, const std::vector<std::string>& key_node_names, const std::vector<std::string>& values_node_names,
			const int& key_length, const int& values_length,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool split_attention_layers = true);

		/**
		@brief Add a concatenation attention layer with activation

		References:
		Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey,
			Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson,
			Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa,
			Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa,
			Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google's neural
			machine translation system: Bridging the gap between human and machine translation. arXiv
			preprint arXiv:1609.08144, 2016.

		Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,
			and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
			arXiv preprint arXiv:1701.06538, 2017.
		*/
		std::vector<std::string> addConcatAttention(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& query_node_names, const std::vector<std::string>& key_node_names, const std::vector<std::string>& values_node_names,
			const int& key_length, const int& values_length,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
			const TensorT& drop_out_prob = 0.0f, const TensorT& drop_connection_prob = 0.0f, const bool& biases = true, bool split_attention_layers = true);

		/**
		@brief Add a Scalar layer to the model

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the fully connected layer to
		@param[in] n_nodes The number of output nodes
		@param[in] scalar_value The value of the scalar
		@param[in] node_activation The activation function of the hidden node to create
		@param[in] node_activation_grad The activation function gradient of the hidden node to create
		@param[in] specify_layer Whether to specify the layer or not

		@returns vector of output node names
		*/
		std::vector<std::string> addScalar(Model<TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names, const TensorT& scalar_value,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const bool& specify_layer = false);

    /**
    @brief Add a Gaussian posterior to the model.  The loss is then calculated on the output
      nodes using `NegativeLogLikelihoodLoss` with an expectation of 1 and scaled by the inverse of the batch size as specified
      in the original Bayes by Backprop formulation

    Reference:
    Blundell 2015 Weight uncertainty in neural networks arXiv:1505.05424
    and the tutorial @ https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html

    @param[in, out] Model
    @param[in] mu_node_names Node_names of the mean output layer
    @param[in] logvar_node_names Node_names of the logvar output layer
    @param[in] gaussian_node_names Node_names of the guassian output layer
    @param[in] specify_layer Whether to specify the layer or not

    @returns vector of output node names
    */
    std::vector<std::string> addGaussianPosterior(Model<TensorT>& model, const std::string& name, const std::string& module_name,
      const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names, const std::vector<std::string>& gaussian_node_names,
      const bool& specify_layer = false);

    /*
    @brief Add a Gaussian difference layer according to the calculations

    scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return scaling * bell
    */
    std::vector<std::string> addGaussian_(Model<TensorT>& model, const std::string& name, const std::string& module_name,
      const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names, const std::vector<std::string>& gaussian_node_names,
      const bool& specify_layer = false);

    /**
    @brief Add a mixed Gaussian prior to the model.  The loss is then calculated on the output
      nodes using `NegativeLogLikelihoodLoss` with an expectation of 1 and scaled by the inverse of the batch size as specified
      in the original Bayes by Backprop formulation

    Calculations:
      first_gaussian = pi * gaussian(x, 0., sigma_p1)
      second_gaussian = (1 - pi) * gaussian(x, 0., sigma_p2)
      return first_gaussian + second_gaussian

    Reference:
    Blundell 2015 Weight uncertainty in neural networks arXiv:1505.05424
    and the tutorial @ https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html

    @param[in, out] Model
    @param[in] gaussian_node_names Node_names of the guassian output layer
    @param[in] logvar_1 Variance 1: -log sigma_1 {0, 1, 2}
    @param[in] logvar_2 Variance 2: -log sigma_2 {3, 4, 5}
    @param[in] pi Mixture percent: pi {0.25, 0.5, 0.75}
    @param[in] specify_layer Whether to specify the layer or not

    @returns vector of output node names
    */
    std::vector<std::string> addMixedGaussianPior(Model<TensorT>& model, const std::string& name, const std::string& module_name,
      const std::vector<std::string>& gaussian_node_names,
      const TensorT& sigma_1, const TensorT& sigma_2, const TensorT& pi,
      const bool& specify_layer = false);

    /**
    @brief Add a fully connected Bayesian layer to a model

    @param[in, out] Model
    @param[in] source_node_names Node_names to add the fully connected layer to
    @param[in] n_nodes The number of output nodes
    @param[in] node_activation The activation function of the hidden node to create
    @param[in] node_activation_grad The activation function gradient of the hidden node to create
    @param[in] node_integration The integration function of the hidden node to create
    ...
    @param[in] logvar_1 Variance 1: -log sigma_1 {0, 1, 2}
    @param[in] logvar_2 Variance 2: -log sigma_2 {3, 4, 5}
    @param[in] pi Mixture percent: pi {0.25, 0.5, 0.75}
    @param[in] node_names_logvar Node_names of the logvar layer
    @param[in] node_names_posterior_output Node_names of the posterior layer
    @param[in] node_names_prior_output Node_names of the prior layer
    @param[out] 

    @returns vector of output node names
    */
    std::vector<std::string> addFullyConnectedBayesian(Model<TensorT>& model, const std::string& name, const std::string& module_name,
      const std::vector<std::string>& source_node_names, const int& n_nodes,
      const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
      const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
      const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
      const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
      const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
      const std::shared_ptr<WeightInitOp<TensorT>>& weight_init_mu, const std::shared_ptr<SolverOp<TensorT>>& solver_mu,
      const std::shared_ptr<WeightInitOp<TensorT>>& weight_init_logvar, const std::shared_ptr<SolverOp<TensorT>>& solver_logvar,
      const TensorT& sigma_1, const TensorT& sigma_2, const TensorT& pi, 
      std::vector<std::string>& node_names_logvar_output, std::vector<std::string>& node_names_posterior_output, std::vector<std::string>& node_names_prior_output,
      const bool& specify_layer = false);

		/**
		@brief Add one model to another

		@param[in, out] Model
		@param[in] source_node_names Node_names in the LH model to add to
		@param[in] sink_node_names Node names in the RH model to join
		@param[in] model_rh The RH model to add to the LH model

		@returns vector of output node names
		*/
		std::vector<std::string> addModel(Model<TensorT>& model, const std::vector<std::string>& source_node_names,
			const std::vector<std::string>& sink_node_names, const Model<TensorT>& model_rh);

		/*
		@brief Make a unity weight
		*/
		std::string makeUnityWeight(Model<TensorT>& model, const TensorT& scale, const std::string& module_name, const std::string& name_format, const std::string& lhs, const std::string& rhs, const bool& specify_layer = false);
  };
	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addInputNodes(Model<TensorT> & model, const std::string & name, const std::string & module_name, const int & n_nodes, const bool& specify_layer)
	{
		std::vector<std::string> node_names;

		// Create the input nodes
		for (int i = 0; i < n_nodes; ++i)
		{
			char* node_name_char = new char[512];
			sprintf(node_name_char, "%s_%012d", name.data(), i);
			std::string node_name(node_name_char);
			node_names.push_back(node_name);
			Node<TensorT> node(node_name, NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			node.setModuleName(module_name);
			if (specify_layer) node.setLayerName(module_name);
			model.addNodes({ node });
      delete[] node_name_char;
		}
		return node_names;
	}
	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addFullyConnected(Model<TensorT>& model, const std::string& name, const std::string& module_name,
		const std::vector<std::string>& source_node_names, const int& n_nodes,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
		const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
		const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, const bool& specify_layer)
	{
		std::vector<std::string> node_names;

		// Create the hidden nodes + biases and hidden to bias links
		for (int i = 0; i < n_nodes; ++i)
		{
			char* node_name_char = new char[512];
			sprintf(node_name_char, "%s_%012d", name.data(), i);
			std::string node_name(node_name_char);
			node_names.push_back(node_name);
			Node<TensorT> node(node_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
			node.setModuleName(module_name);
			node.setDropProbability(drop_out_prob);
			if (specify_layer) node.setLayerName(module_name);
			model.addNodes({ node });
      delete[] node_name_char;

			if (biases) {
				char* bias_name_char = new char[512];
				sprintf(bias_name_char, "%s-bias_%012d", name.data(), i);
				std::string bias_name(bias_name_char);
				Node<TensorT> bias(bias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
				bias.setModuleName(module_name);
				model.addNodes({ bias });
        delete[] bias_name_char;

				char* weight_bias_name_char = new char[512];
				sprintf(weight_bias_name_char, "%s-bias_%012d_to_%s_%012d", name.data(), i, name.data(), i);
				std::string weight_bias_name(weight_bias_name_char);
        delete[] weight_bias_name_char;

				char* link_bias_name_char = new char[512];
				sprintf(link_bias_name_char, "%s-bias_%012d_to_%s_%012d", name.data(), i, name.data(), i);
				std::string link_bias_name(link_bias_name_char);
        delete[] link_bias_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  bias_weight_init;
				bias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>((TensorT)0));
				std::shared_ptr<SolverOp<TensorT>>  bias_solver = solver;
				Weight<TensorT> weight_bias(weight_bias_name, bias_weight_init, bias_solver);
				weight_bias.setModuleName(module_name);
				weight_bias.setDropProbability(drop_connection_prob);
				Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);
				link_bias.setModuleName(module_name);

				model.addWeights({ weight_bias });
				model.addLinks({ link_bias });
			}
		}

		// Create the weights and links for input to hidden
		for (int i = 0; i < source_node_names.size(); ++i)
		{
			for (int j = 0; j < n_nodes; ++j)
			{
				char* hidden_name_char = new char[512];
				sprintf(hidden_name_char, "%s_%012d", name.data(), j);
				std::string hidden_name(hidden_name_char);
        delete[] hidden_name_char;

				char* link_name_char = new char[512];
				sprintf(link_name_char, "%s_to_%s_%012d", source_node_names[i].data(), name.data(), j);
				std::string link_name(link_name_char);
        delete[] link_name_char;

				char* weight_name_char = new char[512];
				sprintf(weight_name_char, "%s_to_%s_%012d", source_node_names[i].data(), name.data(), j);
				std::string weight_name(weight_name_char);
        delete[] weight_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  hidden_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  hidden_solver = solver;
				Weight<TensorT> weight(weight_name, hidden_weight_init, hidden_solver);
				weight.setModuleName(module_name);
				weight.setDropProbability(drop_connection_prob);
				if (specify_layer) weight.setLayerName(module_name);
				Link link(link_name, source_node_names[i], hidden_name, weight_name);
				link.setModuleName(module_name);

				model.addWeights({ weight });
				model.addLinks({ link });
			}
		}
		return node_names;
	}
	template<typename TensorT>
	void ModelBuilder<TensorT>::addFullyConnected(Model<TensorT> & model, const std::string & module_name, const std::vector<std::string>& source_node_names, const std::vector<std::string>& sink_node_names,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver, const TensorT& drop_connection_prob, const bool& specify_layer)
	{

		// Create the weights and links for input to hidden
		for (const std::string& source_node_name : source_node_names)
		{
			for (const std::string& sink_node_name : sink_node_names)
			{
				char* link_name_char = new char[512];
				sprintf(link_name_char, "%s_to_%s", source_node_name.data(), sink_node_name.data());
				std::string link_name(link_name_char);
        delete[] link_name_char;

				char* weight_name_char = new char[512];
				sprintf(weight_name_char, "%s_to_%s", source_node_name.data(), sink_node_name.data());
				std::string weight_name(weight_name_char);
        delete[] weight_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  hidden_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  hidden_solver = solver;
				Weight<TensorT> weight(weight_name, hidden_weight_init, hidden_solver);
				weight.setModuleName(module_name);
				weight.setDropProbability(drop_connection_prob);
				if (specify_layer) weight.setLayerName(module_name);
				Link link(link_name, source_node_name, sink_node_name, weight_name);
				link.setModuleName(module_name);

				model.addWeights({ weight });
				model.addLinks({ link });
			}
		}
	}
	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addSinglyConnected(Model<TensorT>& model, const std::string& name, const std::string& module_name,
		const std::vector<std::string>& source_node_names, const int& n_nodes,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
		const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
		const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, const bool& specify_layer)
	{
		std::vector<std::string> node_names;

		assert(source_node_names.size() == n_nodes);

		// Create the hidden nodes + biases and hidden to bias links
		for (int i = 0; i < n_nodes; ++i)
		{
			char* node_name_char = new char[512];
			sprintf(node_name_char, "%s_%012d", name.data(), i);
			std::string node_name(node_name_char);
			node_names.push_back(node_name);
			Node<TensorT> node(node_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
			node.setModuleName(module_name);
			node.setDropProbability(drop_out_prob);
			if (specify_layer) node.setLayerName(module_name);
			model.addNodes({ node });
      delete[] node_name_char;

			if (biases) {
				char* bias_name_char = new char[512];
				sprintf(bias_name_char, "%s-bias_%012d", name.data(), i);
				std::string bias_name(bias_name_char);
				Node<TensorT> bias(bias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
				bias.setModuleName(module_name);
				model.addNodes({ bias });
        delete[] bias_name_char;

				char* weight_bias_name_char = new char[512];
				sprintf(weight_bias_name_char, "%s-bias_%012d_to_%s_%012d", name.data(), i, name.data(), i);
				std::string weight_bias_name(weight_bias_name_char);
        delete[] weight_bias_name_char;

				char* link_bias_name_char = new char[512];
				sprintf(link_bias_name_char, "%s-bias_%012d_to_%s_%012d", name.data(), i, name.data(), i);
				std::string link_bias_name(link_bias_name_char);
        delete[] link_bias_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  bias_weight_init;
				bias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
				std::shared_ptr<SolverOp<TensorT>>  bias_solver = solver;
				Weight<TensorT> weight_bias(weight_bias_name, bias_weight_init, bias_solver);
				weight_bias.setModuleName(module_name);
				weight_bias.setDropProbability(drop_connection_prob);
				Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);
				link_bias.setModuleName(module_name);

				model.addWeights({ weight_bias });
				model.addLinks({ link_bias });
			}

			// Create the weights and links for input to hidden
			char* hidden_name_char = new char[512];
			sprintf(hidden_name_char, "%s_%012d", name.data(), i);
			std::string hidden_name(hidden_name_char);
      delete[] hidden_name_char;

			char* link_name_char = new char[512];
			sprintf(link_name_char, "%s_to_%s_%012d", source_node_names[i].data(), name.data(), i);
			std::string link_name(link_name_char);
      delete[] link_name_char;

			char* weight_name_char = new char[512];
			sprintf(weight_name_char, "%s_to_%s_%012d", source_node_names[i].data(), name.data(), i);
			std::string weight_name(weight_name_char);
      delete[] weight_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  hidden_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  hidden_solver = solver;
			Weight<TensorT> weight(weight_name, hidden_weight_init, hidden_solver);
			weight.setModuleName(module_name);
			weight.setDropProbability(drop_connection_prob);
			if (specify_layer) weight.setLayerName(module_name);
			Link link(link_name, source_node_names[i], hidden_name, weight_name);
			link.setModuleName(module_name);

			model.addWeights({ weight });
			model.addLinks({ link });
		}
		return node_names;
	}
	template<typename TensorT>
	void ModelBuilder<TensorT>::addSinglyConnected(Model<TensorT> & model, const std::string & module_name, const std::vector<std::string>& source_node_names, const std::vector<std::string>& sink_node_names,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver, const TensorT& drop_connection_prob, const bool& specify_layer)
	{

		assert(source_node_names.size() == sink_node_names.size());

		// Create the weights and links for input to hidden
		for (int i=0; i<source_node_names.size(); ++i)
		{
			char* link_name_char = new char[512];
			sprintf(link_name_char, "%s_to_%s", source_node_names[i].data(), sink_node_names[i].data());
			std::string link_name(link_name_char);
      delete[] link_name_char;

			char* weight_name_char = new char[512];
			sprintf(weight_name_char, "%s_to_%s", source_node_names[i].data(), sink_node_names[i].data());
			std::string weight_name(weight_name_char);
      delete[] weight_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  hidden_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  hidden_solver = solver;
			Weight<TensorT> weight(weight_name, hidden_weight_init, hidden_solver);
			weight.setModuleName(module_name);
			weight.setDropProbability(drop_connection_prob);
			if (specify_layer) weight.setLayerName(module_name);
			Link link(link_name, source_node_names[i], sink_node_names[i], weight_name);
			link.setModuleName(module_name);

			model.addWeights({ weight });
			model.addLinks({ link });
		}
	}
  template<typename TensorT>
  std::vector<std::string> ModelBuilder<TensorT>::addBiases(Model<TensorT>& model, const std::string& module_name, const std::vector<std::string>& sink_node_names,
    const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
    const TensorT& drop_connection_prob, const bool& specify_layer)
  {
    std::vector<std::string> biases_names;

    for (const std::string& node : sink_node_names) {
      // make the bias
      char* bias_name_char = new char[512];
      sprintf(bias_name_char, "%s-bias", node.data());
      std::string bias_name = std::string(bias_name_char);
      Node<TensorT> bias(bias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
      bias.setModuleName(module_name);
      if (specify_layer) bias.setLayerName(module_name + "-bias");
      model.addNodes({ bias });
      biases_names.push_back(bias_name);
      delete[] bias_name_char;

      // make the bias weight
      char* weight_bias_name_char = new char[512];
      sprintf(weight_bias_name_char, "%s_to_%s", bias_name.data(), node.data());
      std::string weight_bias_name = std::string(weight_bias_name_char);
      Weight<TensorT> weight_bias(weight_bias_name, weight_init, solver);
      weight_bias.setModuleName(module_name);
      weight_bias.setDropProbability(drop_connection_prob);
      model.addWeights({ weight_bias });
      delete[] weight_bias_name_char;

      // make the bias link
      char* link_bias_name_char = new char[512];
      sprintf(link_bias_name_char, "%s_to_%s", bias_name.data(), node.data());
      std::string link_bias_name(link_bias_name_char);
      Link link_bias(link_bias_name, bias_name, node, weight_bias_name);
      link_bias.setModuleName(module_name);
      model.addLinks({ link_bias });
      delete[] link_bias_name_char;
    }
    return biases_names;
  }
  template<typename TensorT>
  inline std::vector<std::string> ModelBuilder<TensorT>::addHiddenNodes(Model<TensorT>& model, const std::string& name, const std::string& module_name, const int& n_nodes, const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad, const std::shared_ptr<IntegrationOp<TensorT>>& node_integration, const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error, const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad, const std::shared_ptr<SolverOp<TensorT>>& solver, const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, const bool& specify_layer)
  {
    std::vector<std::string> node_names;

    // Create the hidden nodes + biases and hidden to bias links
    for (int i = 0; i < n_nodes; ++i)
    {
      char* node_name_char = new char[512];
      sprintf(node_name_char, "%s_%012d", name.data(), i);
      std::string node_name(node_name_char);
      node_names.push_back(node_name);
      Node<TensorT> node(node_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
      node.setModuleName(module_name);
      node.setDropProbability(drop_out_prob);
      if (specify_layer) node.setLayerName(module_name);
      model.addNodes({ node });
      delete[] node_name_char;

      if (biases) {
        char* bias_name_char = new char[512];
        sprintf(bias_name_char, "%s-bias_%012d", name.data(), i);
        std::string bias_name(bias_name_char);
        Node<TensorT> bias(bias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
        bias.setModuleName(module_name);
        model.addNodes({ bias });
        delete[] bias_name_char;

        char* weight_bias_name_char = new char[512];
        sprintf(weight_bias_name_char, "%s-bias_%012d_to_%s_%012d", name.data(), i, name.data(), i);
        std::string weight_bias_name(weight_bias_name_char);
        delete[] weight_bias_name_char;

        char* link_bias_name_char = new char[512];
        sprintf(link_bias_name_char, "%s-bias_%012d_to_%s_%012d", name.data(), i, name.data(), i);
        std::string link_bias_name(link_bias_name_char);
        delete[] link_bias_name_char;

        std::shared_ptr<WeightInitOp<TensorT>>  bias_weight_init;
        bias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
        std::shared_ptr<SolverOp<TensorT>> bias_solver = solver;
        Weight<TensorT> weight_bias(weight_bias_name, bias_weight_init, bias_solver);
        weight_bias.setModuleName(module_name);
        weight_bias.setDropProbability(drop_connection_prob);
        Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);
        link_bias.setModuleName(module_name);

        model.addWeights({ weight_bias });
        model.addLinks({ link_bias });
      }
    }
    return node_names;
  }
  template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addSoftMax(Model<TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names, const bool& specify_layer)
	{
		std::vector<std::string> node_names;
		std::string unity_weight_name;

		// Create the Softmax Inverse/Sum node
		char* sms_node_name_char = new char[512];
		sprintf(sms_node_name_char, "%s-Sum", name.data());
		std::string sms_node_name(sms_node_name_char);
		Node<TensorT> sms_node(sms_node_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<InverseOp<TensorT>>(InverseOp<TensorT>()), std::make_shared<InverseGradOp<TensorT>>(InverseGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		sms_node.setModuleName(module_name);
    if (specify_layer) sms_node.setLayerName(module_name + "-SoftMaxSum");
		model.addNodes({ sms_node });
    delete[] sms_node_name_char;

		// Create the Softmax input/output layer
		for (int i = 0; i < source_node_names.size(); ++i)
		{
			// Create the input layer
			char* smi_node_name_char = new char[512];
			sprintf(smi_node_name_char, "%s-In_%012d", name.data(), i);
			std::string smi_node_name(smi_node_name_char);
			Node<TensorT> smi_node(smi_node_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<ExponentialOp<TensorT>>(ExponentialOp<TensorT>()), std::make_shared<ExponentialGradOp<TensorT>>(ExponentialGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
      smi_node.setModuleName(module_name);
      if (specify_layer) smi_node.setLayerName(module_name + "-SoftMaxIn");
      smi_node.setModuleName(module_name);
      delete[] smi_node_name_char;

			// Create the output layer
			char* smo_node_name_char = new char[512];
			sprintf(smo_node_name_char, "%s-Out_%012d", name.data(), i);
			std::string smo_node_name(smo_node_name_char);
			node_names.push_back(smo_node_name);
			Node<TensorT> smo_node(smo_node_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), 
        std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
      smo_node.setModuleName(module_name);
      if (specify_layer) smo_node.setLayerName(module_name + "-SoftMaxOut");
      smo_node.setModuleName(module_name);
      delete[] smo_node_name_char;

			model.addNodes({ smi_node, smo_node });

			// Create the weights and links for the input to softmax input layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", source_node_names[i], smi_node_name);
			char* ismi_link_name_char = new char[512];
			sprintf(ismi_link_name_char, "%s_to_%s", source_node_names[i].data(), smi_node_name.data());
			std::string ismi_link_name(ismi_link_name_char);
			Link ismi_link(ismi_link_name, source_node_names[i], smi_node_name, unity_weight_name);
			ismi_link.setModuleName(module_name);
			model.addLinks({ ismi_link });
      delete[] ismi_link_name_char;

			// Create the weights and links for the softmax input layer to softmax sum layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", smi_node_name, sms_node_name);
			char* smisms_link_name_char = new char[512];
			sprintf(smisms_link_name_char, "%s_to_%s", smi_node_name.data(), sms_node_name.data());
			std::string smisms_link_name(smisms_link_name_char);
			Link smisms_link(smisms_link_name, smi_node_name, sms_node_name, unity_weight_name);
			smisms_link.setModuleName(module_name);
			model.addLinks({ smisms_link });
      delete[] smisms_link_name_char;

			// Create the weights and links for the softmax input layer to softmax output layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", smi_node_name, smo_node_name);
			char* smismo_link_name_char = new char[512];
			sprintf(smismo_link_name_char, "%s_to_%s", smi_node_name.data(), smo_node_name.data());
			std::string smismo_link_name(smismo_link_name_char);
			Link smismo_link(smismo_link_name, smi_node_name, smo_node_name, unity_weight_name);
			smismo_link.setModuleName(module_name);
			model.addLinks({ smismo_link });
      delete[] smismo_link_name_char;

			// Create the weights and links for the softmax sum layer to softmax output layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", sms_node_name, smo_node_name);
			char* smssmo_link_name_char = new char[512];
			sprintf(smssmo_link_name_char, "%s_to_%s", sms_node_name.data(), smo_node_name.data());
			std::string smssmo_link_name(smssmo_link_name_char);
			Link smssmo_link(smssmo_link_name, sms_node_name, smo_node_name, unity_weight_name);
			smssmo_link.setModuleName(module_name);
			model.addLinks({ smssmo_link });
      delete[] smssmo_link_name_char;
		}

		return node_names;
	}
	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addStableSoftMax(Model<TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names, const bool& specify_layer)
	{
		std::vector<std::string> node_names;
		std::string unity_weight_name, negunity_weight_name;

		// Create the Softmax Max offset node
		char* smm_node_name_char = new char[512];
		sprintf(smm_node_name_char, "%s-Max", name.data());
		std::string smm_node_name(smm_node_name_char);
		Node<TensorT> smm_node(smm_node_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), 
      std::make_shared<MaxOp<TensorT>>(MaxOp<TensorT>()), std::make_shared<MaxErrorOp<TensorT>>(MaxErrorOp<TensorT>()), std::make_shared<MaxWeightGradOp<TensorT>>(MaxWeightGradOp<TensorT>()));
		smm_node.setModuleName(module_name);
		if (specify_layer) smm_node.setLayerName(module_name +"-Max");
		model.addNodes({ smm_node });
    delete[] smm_node_name_char;

		// Create the Softmax Inverse/Sum node
		char* sms_node_name_char = new char[512];
		sprintf(sms_node_name_char, "%s-Sum", name.data());
		std::string sms_node_name(sms_node_name_char);
		Node<TensorT> sms_node(sms_node_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<InverseOp<TensorT>>(InverseOp<TensorT>()), std::make_shared<InverseGradOp<TensorT>>(InverseGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		sms_node.setModuleName(module_name);
		if (specify_layer) sms_node.setLayerName(module_name + "-Sum");
		model.addNodes({ sms_node });
    delete[] sms_node_name_char;

		// Create the Softmax input/output layer
		for (int i = 0; i < source_node_names.size(); ++i)
		{
			// Create the input layer
			char* smi_node_name_char = new char[512];
			sprintf(smi_node_name_char, "%s-In_%012d", name.data(), i);
			std::string smi_node_name(smi_node_name_char);
			Node<TensorT> smi_node(smi_node_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<ExponentialOp<TensorT>>(ExponentialOp<TensorT>()), std::make_shared<ExponentialGradOp<TensorT>>(ExponentialGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			smi_node.setModuleName(module_name);
			if (specify_layer) smi_node.setLayerName(module_name + "-SoftMaxIn");
      delete[] smi_node_name_char;

			// Create the output layer
			char* smo_node_name_char = new char[512];
			sprintf(smo_node_name_char, "%s-Out_%012d", name.data(), i);
			std::string smo_node_name(smo_node_name_char);
			node_names.push_back(smo_node_name);
			Node<TensorT> smo_node(smo_node_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			smo_node.setModuleName(module_name);
			if (specify_layer) smo_node.setLayerName(module_name + "-SoftMaxOut");
      delete[] smo_node_name_char;

			model.addNodes({ smi_node, smo_node });

			// Create the weights and links for the input to softmax Max node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", source_node_names[i], smm_node_name);
			char* ismm_link_name_char = new char[512];
			sprintf(ismm_link_name_char, "%s_to_%s", source_node_names[i].data(), smm_node_name.data());
			std::string ismm_link_name(ismm_link_name_char);
			Link ismm_link(ismm_link_name, source_node_names[i], smm_node_name, unity_weight_name);
			ismm_link.setModuleName(module_name);
			model.addLinks({ ismm_link });
      delete[] ismm_link_name_char;

			// Create the weights and links for the softmax Max node softmax input layer
			negunity_weight_name = makeUnityWeight(model, -1.0, module_name, "%s_to_%s", smm_node_name, smi_node_name);
			char* smmsmi_link_name_char = new char[512];
			sprintf(smmsmi_link_name_char, "%s_to_%s", smm_node_name.data(), smi_node_name.data());
			std::string smmsmi_link_name(smmsmi_link_name_char);
			Link smmsmi_link(smmsmi_link_name, smm_node_name, smi_node_name, negunity_weight_name);
			smmsmi_link.setModuleName(module_name);
			model.addLinks({ smmsmi_link });
      delete[] smmsmi_link_name_char;

			// Create the weights and links for the input to softmax input layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", source_node_names[i], smi_node_name);
			char* ismi_link_name_char = new char[512];
			sprintf(ismi_link_name_char, "%s_to_%s", source_node_names[i].data(), smi_node_name.data());
			std::string ismi_link_name(ismi_link_name_char);
			Link ismi_link(ismi_link_name, source_node_names[i], smi_node_name, unity_weight_name);
			ismi_link.setModuleName(module_name);
			model.addLinks({ ismi_link });
      delete[] ismi_link_name_char;

			// Create the weights and links for the softmax input layer to softmax sum layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", smi_node_name, sms_node_name);
			char* smisms_link_name_char = new char[512];
			sprintf(smisms_link_name_char, "%s_to_%s", smi_node_name.data(), sms_node_name.data());
			std::string smisms_link_name(smisms_link_name_char);
			Link smisms_link(smisms_link_name, smi_node_name, sms_node_name, unity_weight_name);
			smisms_link.setModuleName(module_name);
			model.addLinks({ smisms_link });
      delete[] smisms_link_name_char;

			// Create the weights and links for the softmax input layer to softmax output layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", smi_node_name, smo_node_name);
			char* smismo_link_name_char = new char[512];
			sprintf(smismo_link_name_char, "%s_to_%s", smi_node_name.data(), smo_node_name.data());
			std::string smismo_link_name(smismo_link_name_char);
			Link smismo_link(smismo_link_name, smi_node_name, smo_node_name, unity_weight_name);
			smismo_link.setModuleName(module_name);
			model.addLinks({ smismo_link });
      delete[] smismo_link_name_char;

			// Create the weights and links for the softmax sum layer to softmax output layer
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", sms_node_name, smo_node_name);
			char* smssmo_link_name_char = new char[512];
			sprintf(smssmo_link_name_char, "%s_to_%s", sms_node_name.data(), smo_node_name.data());
			std::string smssmo_link_name(smssmo_link_name_char);
			Link smssmo_link(smssmo_link_name, sms_node_name, smo_node_name, unity_weight_name);
			smssmo_link.setModuleName(module_name);
			model.addLinks({ smssmo_link });
      delete[] smssmo_link_name_char;
		}

		return node_names;
	}
	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addConvolution(Model<TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names,
		const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
		const int & extent_width, const int & extent_height, const int & stride,
		const int & output_width_zero_padding, const int& output_height_zero_padding,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
		const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
		const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, bool split_filter_layers, bool share_weights)
	{
		std::vector<std::string> node_names;

		// Parameters for the Convolution layer
		assert(source_node_names.size() == input_width * input_height);
		int input_padded_width = input_width + 2 * input_width_zero_padding;
		//assert((input_padded_width - extent_width) % stride == 0);
		if ((input_padded_width - extent_width) % stride != 0)
			std::cout << "Warning: input width, filter width, and stride lengths will not allow for uniform coverage during convolution." << std::endl;
		int strides_width = std::floor((input_padded_width - extent_width) / stride) +1; // includes the starting stride
		int input_padded_height = input_height + 2 * input_height_zero_padding;
		//assert((input_padded_height - extent_height) % stride == 0);
		if ((input_padded_height - extent_height) % stride != 0)
			std::cout << "Warning: input height, filter height, and stride lengths will not allow for uniform coverage during convolution." << std::endl;
		int strides_height = std::floor((input_padded_height - extent_height) / stride) +1; // includes the starting stride
		int output_nodes = strides_width + strides_height;
		int output_padded_width = strides_width + 2 * output_width_zero_padding;
		int output_padded_height = strides_height + 2 * output_height_zero_padding;

		std::string bias_name;
		std::string weight_bias_name;
		if (biases && share_weights) {
			// Create the filter bias
			char* bias_name_char = new char[512];
			sprintf(bias_name_char, "%s-bias", name.data());
			bias_name = std::string(bias_name_char);
			Node<TensorT> bias(bias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			bias.setModuleName(module_name);
			model.addNodes({ bias });
      delete[] bias_name_char;

			// Create the shared weights for each bias to output node
			char* weight_bias_name_char = new char[512];
			sprintf(weight_bias_name_char, "%s_to_out", bias_name.data());
			weight_bias_name = std::string(weight_bias_name_char);
			Weight<TensorT> weight_bias(weight_bias_name, weight_init, solver);
			weight_bias.setModuleName(module_name);
			model.addWeights({ weight_bias });
      delete[] weight_bias_name_char;
		}

		// Create the output zero padding nodes
		for (size_t output_width_iter = 0; output_width_iter < output_padded_width; ++output_width_iter) {
			for (size_t output_height_iter = 0; output_height_iter < output_padded_height; ++output_height_iter) {
				if (output_height_iter < output_height_zero_padding || output_height_iter >= output_padded_height - output_height_zero_padding) {
					char* bias_name_char = new char[512];
					sprintf(bias_name_char, "%s-out-padding_H%012d-W%012d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node<TensorT> bias(bias_name, NodeType::zero, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
					bias.setModuleName(module_name);
					if (split_filter_layers) bias.setLayerName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
          delete[] bias_name_char;
				}
				else if (output_width_iter < output_width_zero_padding || output_width_iter >= output_padded_width - output_width_zero_padding) {
					char* bias_name_char = new char[512];
					sprintf(bias_name_char, "%s-out-padding_H%012d-W%012d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node<TensorT> bias(bias_name, NodeType::zero, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
					bias.setModuleName(module_name);
					if (split_filter_layers) bias.setLayerName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
          delete[] bias_name_char;
				}
				else {
					char* output_name_char = new char[512];
					sprintf(output_name_char, "%s-out_H%012d-W%012d", name.data(), output_height_iter, output_width_iter);
					std::string output_name(output_name_char);
					Node<TensorT> output(output_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
					output.setModuleName(module_name);
					output.setDropProbability(drop_out_prob);
					if (split_filter_layers) output.setLayerName(module_name);
					model.addNodes({ output });
					node_names.push_back(output_name);
          delete[] output_name_char;

					if (biases && share_weights) {
						// Create the links between the bias and output nodes
						char* link_bias_name_char = new char[512];
						sprintf(link_bias_name_char, "%s_to_%s_%s", bias_name.data(), output_name.data(), module_name.data());
						std::string link_bias_name(link_bias_name_char);
						Link link_bias(link_bias_name, bias_name, output_name, weight_bias_name);
						link_bias.setModuleName(module_name);
						model.addLinks({ link_bias });
            delete[] link_bias_name_char;
					}
          else if (biases) {
            // Create the filter bias
            char* bias_name_char = new char[512];
            sprintf(bias_name_char, "%s-bias", output_name.data());
            bias_name = std::string(bias_name_char);
            Node<TensorT> bias(bias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
            bias.setModuleName(module_name);
            model.addNodes({ bias });
            delete[] bias_name_char;

            // Create the shared weights for each bias to output node
            char* weight_bias_name_char = new char[512];
            sprintf(weight_bias_name_char, "%s_to_%s_%s", bias_name.data(), output_name.data(), module_name.data());
            weight_bias_name = std::string(weight_bias_name_char);
            Weight<TensorT> weight_bias(weight_bias_name, weight_init, solver);
            weight_bias.setModuleName(module_name);
            model.addWeights({ weight_bias });
            delete[] weight_bias_name_char;

            // Create the links between the bias and output nodes
            char* link_bias_name_char = new char[512];
            sprintf(link_bias_name_char, "%s_to_%s_%s", bias_name.data(), output_name.data(), module_name.data());
            std::string link_bias_name(link_bias_name_char);
            Link link_bias(link_bias_name, bias_name, output_name, weight_bias_name);
            link_bias.setModuleName(module_name);
            model.addLinks({ link_bias });
            delete[] link_bias_name_char;
          }
				}
			}
		}

    if (share_weights) {
      // Create the shared weights for each filter link
      for (size_t filter_height_iter = 0; filter_height_iter < extent_height; ++filter_height_iter) {
        for (size_t filter_width_iter = 0; filter_width_iter < extent_width; ++filter_width_iter) {
          char* weight_filter_name_char = new char[512];
          sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
          std::string weight_filter_name(weight_filter_name_char);
          Weight<TensorT> weight_filter(weight_filter_name, weight_init, solver);
          weight_filter.setModuleName(module_name);
          weight_filter.setDropProbability(drop_connection_prob);
          if (split_filter_layers) weight_filter.setLayerName(module_name);
          model.addWeights({ weight_filter });
          delete[] weight_filter_name_char;
        }
      }
    }

		// Create the convolution links between input and output					
		int tmp = 0;
		int output_width_iter = 0;
		for (size_t width_stride_iter = 0; width_stride_iter < strides_width; ++width_stride_iter) {
			// check if the filter is in the left input width zero padding
			const int filter_width_end = stride * width_stride_iter + extent_width;
			if (filter_width_end <= input_width_zero_padding)
				continue;

			// check if the filter is in the right input width zero padding
			const int filter_width_start = stride * width_stride_iter;
			if (filter_width_start >= input_width_zero_padding + input_width)
				continue;

			// offset the starting width filter for the input zero padding
			int filter_width_offset_start_tmp = input_width_zero_padding - stride * width_stride_iter;
			int filter_width_offset_start = maxFunc(filter_width_offset_start_tmp, 0);
			int filter_width_offset_end_tmp = -input_width_zero_padding + stride * strides_width - stride * width_stride_iter + extent_width;
			int filter_width_offset_end = minFunc(filter_width_offset_end_tmp, extent_width);

			int output_height_iter = 0;
			for (size_t height_stride_iter = 0; height_stride_iter < strides_height; ++height_stride_iter) {
				// check if the filter is in the top input height zero padding
				const int filter_height_end = stride * height_stride_iter + extent_height;
				if (filter_height_end <= input_height_zero_padding)
					continue;

				// check if the filter is in the bottom input height zero padding
				const int filter_height_start = stride * height_stride_iter;
				if (filter_height_start >= input_height_zero_padding + input_height)
					continue;

				// offset starting height filter for the input zero padding
				int filter_height_offset_start_tmp = input_height_zero_padding - stride * height_stride_iter;
				int filter_height_offset_start = maxFunc(filter_height_offset_start_tmp, 0);
				int filter_height_offset_end_tmp = -input_height_zero_padding + stride * strides_height - stride * height_stride_iter + extent_height;
				int filter_height_offset_end = minFunc(filter_height_offset_end_tmp, extent_height);

				// create the links between input and output
				int width_iter_tmp = stride * width_stride_iter - input_width_zero_padding;
				int width_iter = maxFunc(width_iter_tmp, 0);
				for (size_t filter_width_iter = filter_width_offset_start; filter_width_iter < filter_width_offset_end; ++filter_width_iter) {
					int height_iter_tmp = stride * height_stride_iter - input_height_zero_padding; 
					int height_iter = maxFunc(height_iter_tmp, 0);
					for (size_t filter_height_iter = filter_height_offset_start; filter_height_iter < filter_height_offset_end; ++filter_height_iter) {
						int source_node_iter = height_iter + width_iter * input_height;

						if (source_node_iter >= source_node_names.size()) {
							//std::cout << "WARNING: node size has been exceeded!" << std::endl;
							break;
						}

            // Output node name
            char* output_name_char = new char[512];
            sprintf(output_name_char, "%s-out_H%012d-W%012d", name.data(), output_height_iter + output_height_zero_padding, output_width_iter + output_width_zero_padding);
            std::string output_name(output_name_char);
            delete[] output_name_char;

            // Weight<TensorT> name
            std::string weight_filter_name;
            if (share_weights) {
              char* weight_filter_name_char = new char[512];
              sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
              weight_filter_name = std::string(weight_filter_name_char);
              delete[] weight_filter_name_char;
            }
            else {
              char* weight_filter_name_char = new char[512];
              sprintf(weight_filter_name_char, "%s_to_%s_%s", source_node_names.at(source_node_iter).data(), output_name.data(), module_name.data());
              weight_filter_name = std::string(weight_filter_name_char);
              Weight<TensorT> weight_filter(weight_filter_name, weight_init, solver);
              weight_filter.setModuleName(module_name);
              weight_filter.setDropProbability(drop_connection_prob);
              if (split_filter_layers) weight_filter.setLayerName(module_name);
              model.addWeights({ weight_filter });
              delete[] weight_filter_name_char;
            }

						// Link name
						char* link_filter_name_char = new char[512];
						sprintf(link_filter_name_char, "%s_to_%s_%s", source_node_names[source_node_iter].data(), output_name.data(), module_name.data());
						std::string link_filter_name(link_filter_name_char);
            delete[] link_filter_name_char;

						Link link_filter(link_filter_name, source_node_names[source_node_iter], output_name, weight_filter_name);
						link_filter.setModuleName(module_name);
						model.addLinks({ link_filter });

						++height_iter;
					}
					++width_iter;
				}
				++output_height_iter;
			}
			++output_width_iter;
		}

		return node_names;
	}
	template<typename TensorT>
	void ModelBuilder<TensorT>::addConvolution(Model<TensorT> & model, const std::string & name, const std::string& module_name,
		const std::vector<std::string>& source_node_names,
		const std::vector<std::string>& sink_node_names,
		const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
		const int & extent_width, const int & extent_height, const int & stride,
		const int & output_width_zero_padding, const int& output_height_zero_padding,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, bool split_filter_layers)
	{
		// Parameters for the Convolution layer
		assert(source_node_names.size() == input_width * input_height);
		int input_padded_width = input_width + 2 * input_width_zero_padding;
		//assert((input_padded_width - extent_width) % stride == 0);
		if ((input_padded_width - extent_width) % stride != 0)
			std::cout << "Warning: input width, filter width, and stride lengths will not allow for uniform coverage during convolution." << std::endl;
		int strides_width = std::floor((input_padded_width - extent_width) / stride) + 1; // includes the starting stride
		int input_padded_height = input_height + 2 * input_height_zero_padding;
		//assert((input_padded_height - extent_height) % stride == 0);
		if ((input_padded_height - extent_height) % stride != 0)
			std::cout << "Warning: input height, filter height, and stride lengths will not allow for uniform coverage during convolution." << std::endl;
		int strides_height = std::floor((input_padded_height - extent_height) / stride) + 1; // includes the starting stride
		int output_nodes = strides_width + strides_height;
		int output_padded_width = strides_width + 2 * output_width_zero_padding;
		int output_padded_height = strides_height + 2 * output_height_zero_padding;
		assert(sink_node_names.size() == output_padded_width * output_padded_height);
		
		// Create the shared weights for each filter link
		for (size_t filter_height_iter = 0; filter_height_iter < extent_height; ++filter_height_iter) {
			for (size_t filter_width_iter = 0; filter_width_iter < extent_width; ++filter_width_iter) {
				char* weight_filter_name_char = new char[512];
				sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
				std::string weight_filter_name(weight_filter_name_char);
				Weight<TensorT> weight_filter(weight_filter_name, weight_init, solver);
				weight_filter.setModuleName(module_name);
				weight_filter.setDropProbability(drop_connection_prob);
				if (split_filter_layers) weight_filter.setLayerName(module_name);
				model.addWeights({ weight_filter });
        delete[] weight_filter_name_char;
			}
		}

		// Create the convolution links between input and output					
		int tmp = 0;
		int output_width_iter = 0;
		for (size_t width_stride_iter = 0; width_stride_iter < strides_width; ++width_stride_iter) {
			// check if the filter is in the left input width zero padding
			const int filter_width_end = stride * width_stride_iter + extent_width;
			if (filter_width_end <= input_width_zero_padding)
				continue;

			// check if the filter is in the right input width zero padding
			const int filter_width_start = stride * width_stride_iter;
			if (filter_width_start >= input_width_zero_padding + input_width)
				continue;

			// offset the starting width filter for the input zero padding
			int filter_width_offset_start_tmp = input_width_zero_padding - stride * width_stride_iter;
			int filter_width_offset_start = maxFunc(filter_width_offset_start_tmp, 0);
			int filter_width_offset_end_tmp = -input_width_zero_padding + stride * strides_width - stride * width_stride_iter + extent_width;
			int filter_width_offset_end = minFunc(filter_width_offset_end_tmp, extent_width);

			int output_height_iter = 0;
			for (size_t height_stride_iter = 0; height_stride_iter < strides_height; ++height_stride_iter) {
				// check if the filter is in the top input height zero padding
				const int filter_height_end = stride * height_stride_iter + extent_height;
				if (filter_height_end <= input_height_zero_padding)
					continue;

				// check if the filter is in the bottom input height zero padding
				const int filter_height_start = stride * height_stride_iter;
				if (filter_height_start >= input_height_zero_padding + input_height)
					continue;

				// offset starting height filter for the input zero padding
				int filter_height_offset_start_tmp = input_height_zero_padding - stride * height_stride_iter;
				int filter_height_offset_start = maxFunc(filter_height_offset_start_tmp, 0);
				int filter_height_offset_end_tmp = -input_height_zero_padding + stride * strides_height - stride * height_stride_iter + extent_height;
				int filter_height_offset_end = minFunc(filter_height_offset_end_tmp, extent_height);

				// create the links between input and output
				int width_iter_tmp = stride * width_stride_iter - input_width_zero_padding;
				int width_iter = maxFunc(width_iter_tmp, 0);
				for (size_t filter_width_iter = filter_width_offset_start; filter_width_iter < filter_width_offset_end; ++filter_width_iter) {
					int height_iter_tmp = stride * height_stride_iter - input_height_zero_padding;
					int height_iter = maxFunc(height_iter_tmp, 0);
					for (size_t filter_height_iter = filter_height_offset_start; filter_height_iter < filter_height_offset_end; ++filter_height_iter) {
						int source_node_iter = height_iter + width_iter * input_height;

						if (source_node_iter >= source_node_names.size()) {
							//std::cout << "WARNING: node size has been exceeded!" << std::endl;
							break;
						}

						// Weight<TensorT> name
						char* weight_filter_name_char = new char[512];
						sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
						std::string weight_filter_name(weight_filter_name_char);
            delete[] weight_filter_name_char;

						// Output node name
						char* output_name_char = new char[512];
						sprintf(output_name_char, "%s-out_H%012d-W%012d", name.data(), output_height_iter + output_height_zero_padding, output_width_iter + output_width_zero_padding);
						std::string output_name(output_name_char);
						assert(std::count(sink_node_names.begin(), sink_node_names.end(), output_name) == 1);
            delete[] output_name_char;

						// Link name
						char* link_filter_name_char = new char[512];
						sprintf(link_filter_name_char, "%s_to_%s_%s", source_node_names[source_node_iter].data(), output_name.data(), module_name.data());
						std::string link_filter_name(link_filter_name_char);
            delete[] link_filter_name_char;

						Link link_filter(link_filter_name, source_node_names[source_node_iter], output_name, weight_filter_name);
						link_filter.setModuleName(module_name);
						model.addLinks({ link_filter });

						++height_iter;
					}
					++width_iter;
				}
				++output_height_iter;
			}
			++output_width_iter;
		}
	}
	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addProjection(Model<TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names,
		const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
		const int & extent_width, const int & extent_height, const int & spacing,
		const int & output_width_zero_padding, const int& output_height_zero_padding,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
		const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
		const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, bool split_filter_layers, bool share_weights)
	{
		std::vector<std::string> node_names;

		// Parameters for the Convolution layer
		assert(source_node_names.size() == input_width * input_height);
		int input_padded_width = input_width + 2 * input_width_zero_padding;
		int input_padded_height = input_height + 2 * input_height_zero_padding;
		int strides_width = input_padded_width;
		int strides_height = input_padded_height;
		int output_width = input_padded_width + (extent_width - 1) + input_padded_width * (spacing - 1);
		int output_height = input_padded_height + (extent_height - 1) + input_padded_height * (spacing - 1);
		int output_padded_width = output_width + 2 * output_width_zero_padding;
		int output_padded_height = output_height + 2 * output_height_zero_padding;

		// [TODO: would need to be refactored to add a bias for each filter output (i.e., extent_width * extent_height)]
		std::string bias_name;
		//std::string weight_bias_name;
		//if (biases) {
		//	// Create the filter bias
		//	char* bias_name_char = new char[512];
		//	sprintf(bias_name_char, "%s-bias", name.data());
		//	bias_name = std::string(bias_name_char);
		//	Node<TensorT> bias(bias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		//	bias.setModuleName(module_name);
		//	model.addNodes({ bias });

		//	// Create the shared weights for each bias to output node
		//	char* weight_bias_name_char = new char[512];
		//	sprintf(weight_bias_name_char, "%s_to_out", bias_name.data());
		//	weight_bias_name = std::string(weight_bias_name_char);
		//	Weight<TensorT> weight_bias(weight_bias_name, weight_init, solver);
		//	weight_bias.setModuleName(module_name);
		//	weight_bias.setDropProbability(drop_connection_prob);
		//	model.addWeights({ weight_bias });
		//}

		// Create the output zero padding nodes
		for (size_t output_width_iter = 0; output_width_iter < output_padded_width; ++output_width_iter) {
			for (size_t output_height_iter = 0; output_height_iter < output_padded_height; ++output_height_iter) {
				if (output_height_iter < output_height_zero_padding || output_height_iter >= output_padded_height - output_height_zero_padding) {
					char* bias_name_char = new char[512];
					sprintf(bias_name_char, "%s-out-padding_H%012d-W%012d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node<TensorT> bias(bias_name, NodeType::zero, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
					bias.setModuleName(module_name);
					if (split_filter_layers) bias.setLayerName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
          delete[] bias_name_char;
				}
				else if (output_width_iter < output_width_zero_padding || output_width_iter >= output_padded_width - output_width_zero_padding) {
					char* bias_name_char = new char[512];
					sprintf(bias_name_char, "%s-out-padding_H%012d-W%012d", name.data(), output_height_iter, output_width_iter);
					std::string bias_name(bias_name_char);
					Node<TensorT> bias(bias_name, NodeType::zero, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
					bias.setModuleName(module_name);
					if (split_filter_layers) bias.setLayerName(module_name);
					model.addNodes({ bias });
					node_names.push_back(bias_name);
          delete[] bias_name_char;
				}
				else {
					char* output_name_char = new char[512];
					sprintf(output_name_char, "%s-out_H%012d-W%012d", name.data(), output_height_iter, output_width_iter);
					std::string output_name(output_name_char);
					Node<TensorT> output(output_name, NodeType::hidden, NodeStatus::activated, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
					output.setModuleName(module_name);
					output.setDropProbability(drop_out_prob);
					if (split_filter_layers) output.setLayerName(module_name);
					model.addNodes({ output });
					node_names.push_back(output_name);
          delete[] output_name_char;

					//if (biases) {
					//	// Create the links between the bias and output nodes
					//	char* link_bias_name_char = new char[512];
					//	sprintf(link_bias_name_char, "%s_to_%s_%s", bias_name.data(), output_name.data(), module_name.data());
					//	std::string link_bias_name(link_bias_name_char);
					//	Link link_bias(link_bias_name, bias_name, output_name, weight_bias_name);
					//	link_bias.setModuleName(module_name);
					//	model.addLinks({ link_bias });
					//}
				}
			}
		}

    if (share_weights) {
      // Create the shared weights for each filter link
      for (size_t filter_height_iter = 0; filter_height_iter < extent_height; ++filter_height_iter) {
        for (size_t filter_width_iter = 0; filter_width_iter < extent_width; ++filter_width_iter) {
          char* weight_filter_name_char = new char[512];
          sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
          std::string weight_filter_name(weight_filter_name_char);
          Weight<TensorT> weight_filter(weight_filter_name, weight_init, solver);
          weight_filter.setModuleName(module_name);
          weight_filter.setDropProbability(drop_connection_prob);
          if (split_filter_layers) weight_filter.setLayerName(module_name);
          model.addWeights({ weight_filter });
          delete[] weight_filter_name_char;
        }
      }
    }

		// Create the projection links between input and output					
		int tmp = 0;
		for (size_t width_stride_iter = 0; width_stride_iter < strides_width; ++width_stride_iter) {
			// check if the filter is in the left input width zero padding
			const int filter_width_end = (spacing - 1) * width_stride_iter + width_stride_iter + extent_width - 1;
			if (width_stride_iter < input_width_zero_padding)
				continue;

			// check if the filter is in the right input width zero padding
			const int filter_width_start = (spacing - 1) * width_stride_iter + width_stride_iter;
			if (width_stride_iter >= input_width_zero_padding + input_width)
				continue;

			for (size_t height_stride_iter = 0; height_stride_iter < strides_height; ++height_stride_iter) {
				// check if the filter is in the top input height zero padding
				const int filter_height_end = (spacing - 1) * height_stride_iter + height_stride_iter + extent_height - 1;
				if (height_stride_iter < input_height_zero_padding)
					continue;

				// check if the filter is in the bottom input height zero padding
				const int filter_height_start = (spacing - 1) * height_stride_iter + height_stride_iter;
				if (height_stride_iter >= input_height_zero_padding + input_height)
					continue;

				// create the links between input and output
				int width_iter = width_stride_iter - input_width_zero_padding;
				int height_iter = height_stride_iter - input_height_zero_padding;
				int source_node_iter = height_iter + width_iter * input_height;

				if (source_node_iter >= source_node_names.size()) {
					//std::cout << "WARNING: node size has been exceeded!" << std::endl;
					break;
				}

				int filter_width_iter = 0;
				for (size_t filter_width_pos = filter_width_start; filter_width_pos <= filter_width_end; ++filter_width_pos) {
					int filter_height_iter = 0;
					for (size_t filter_height_pos = filter_height_start; filter_height_pos <= filter_height_end; ++filter_height_pos) {

            // Output node name
            char* output_name_char = new char[512];
            sprintf(output_name_char, "%s-out_H%012d-W%012d", name.data(), filter_height_pos + output_height_zero_padding, filter_width_pos + output_width_zero_padding);
            std::string output_name(output_name_char);
            delete[] output_name_char;

						// Weight name
            std::string weight_filter_name;
            if (share_weights) {
              char* weight_filter_name_char = new char[512];
              sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
              weight_filter_name = std::string(weight_filter_name_char);
              delete[] weight_filter_name_char;
            }
            else {
              char* weight_filter_name_char = new char[512];
              sprintf(weight_filter_name_char, "%s_to_%s_%s", source_node_names[source_node_iter].data(), output_name.data(), module_name.data());
              weight_filter_name = std::string(weight_filter_name_char);
              Weight<TensorT> weight_filter(weight_filter_name, weight_init, solver);
              weight_filter.setModuleName(module_name);
              weight_filter.setDropProbability(drop_connection_prob);
              if (split_filter_layers) weight_filter.setLayerName(module_name);
              model.addWeights({ weight_filter });
              delete[] weight_filter_name_char;
            }

						// Link name
						char* link_filter_name_char = new char[512];
						sprintf(link_filter_name_char, "%s_to_%s_%s", source_node_names[source_node_iter].data(), output_name.data(), module_name.data());
						std::string link_filter_name(link_filter_name_char);
            delete[] link_filter_name_char;

						Link link_filter(link_filter_name, source_node_names[source_node_iter], output_name, weight_filter_name);
						link_filter.setModuleName(module_name);
						model.addLinks({ link_filter });

						++filter_height_iter;
					}
					++filter_width_iter;
				}
			}
		}

		return node_names;
	}

	template<typename TensorT>
	inline void ModelBuilder<TensorT>::addProjection(Model<TensorT>& model, const std::string & name, const std::string & module_name, 
		const std::vector<std::string>& source_node_names, const std::vector<std::string>& output_node_names, const int & input_width, const int & input_height, const int & input_width_zero_padding, const int & input_height_zero_padding, const int & extent_width, const int & extent_height, const int & spacing, const int & output_width_zero_padding, const int & output_height_zero_padding, const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver, const TensorT& drop_out_prob, const TensorT& drop_connection_prob, bool split_filter_layers)
	{

		// Parameters for the Convolution layer
		assert(source_node_names.size() == input_width * input_height);
		int input_padded_width = input_width + 2 * input_width_zero_padding;
		int input_padded_height = input_height + 2 * input_height_zero_padding;
		int strides_width = input_padded_width;
		int strides_height = input_padded_height;
		int output_width = input_padded_width + (extent_width - 1) + input_padded_width * (spacing - 1);
		int output_height = input_padded_height + (extent_height - 1) + input_padded_height * (spacing - 1);
		int output_padded_width = output_width + 2 * output_width_zero_padding;
		int output_padded_height = output_height + 2 * output_height_zero_padding;

		// Create the shared weights for each filter link
		for (size_t filter_height_iter = 0; filter_height_iter < extent_height; ++filter_height_iter) {
			for (size_t filter_width_iter = 0; filter_width_iter < extent_width; ++filter_width_iter) {
				char* weight_filter_name_char = new char[512];
				sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
				std::string weight_filter_name(weight_filter_name_char);
				Weight<TensorT> weight_filter(weight_filter_name, weight_init, solver);
				weight_filter.setModuleName(module_name);
				weight_filter.setDropProbability(drop_connection_prob);
				if (split_filter_layers) weight_filter.setLayerName(module_name);
				model.addWeights({ weight_filter });
        delete[] weight_filter_name_char;
			}
		}

		// Create the projection links between input and output					
		int tmp = 0;
		for (size_t width_stride_iter = 0; width_stride_iter < strides_width; ++width_stride_iter) {
			// check if the filter is in the left input width zero padding
			const int filter_width_end = (spacing - 1) * width_stride_iter + width_stride_iter + extent_width - 1;
			if (width_stride_iter < input_width_zero_padding)
				continue;

			// check if the filter is in the right input width zero padding
			const int filter_width_start = (spacing - 1) * width_stride_iter + width_stride_iter;
			if (width_stride_iter >= input_width_zero_padding + input_width)
				continue;

			for (size_t height_stride_iter = 0; height_stride_iter < strides_height; ++height_stride_iter) {
				// check if the filter is in the top input height zero padding
				const int filter_height_end = (spacing - 1) * height_stride_iter + height_stride_iter + extent_height - 1;
				if (height_stride_iter < input_height_zero_padding)
					continue;

				// check if the filter is in the bottom input height zero padding
				const int filter_height_start = (spacing - 1) * height_stride_iter + height_stride_iter;
				if (height_stride_iter >= input_height_zero_padding + input_height)
					continue;

				// create the links between input and output
				int width_iter = width_stride_iter - input_width_zero_padding;
				int height_iter = height_stride_iter - input_height_zero_padding;
				int source_node_iter = height_iter + width_iter * input_height;

				if (source_node_iter >= source_node_names.size()) {
					//std::cout << "WARNING: node size has been exceeded!" << std::endl;
					break;
				}

				int filter_width_iter = 0;
				for (size_t filter_width_pos = filter_width_start; filter_width_pos <= filter_width_end; ++filter_width_pos) {
					int filter_height_iter = 0;
					for (size_t filter_height_pos = filter_height_start; filter_height_pos <= filter_height_end; ++filter_height_pos) {

						// Weight name
						char* weight_filter_name_char = new char[512];
						sprintf(weight_filter_name_char, "%s-%s_H%012d-W%012d", name.data(), module_name.data(), filter_height_iter, filter_width_iter);
						std::string weight_filter_name(weight_filter_name_char);
            delete[] weight_filter_name_char;

						// Output node name
						char* output_name_char = new char[512];
						sprintf(output_name_char, "%s-out_H%012d-W%012d", name.data(), filter_height_pos + output_height_zero_padding, filter_width_pos + output_width_zero_padding);
						std::string output_name(output_name_char);
            delete[] output_name_char;

						// Link name
						char* link_filter_name_char = new char[512];
						sprintf(link_filter_name_char, "%s_to_%s_%s", source_node_names[source_node_iter].data(), output_name.data(), module_name.data());
						std::string link_filter_name(link_filter_name_char);
            delete[] link_filter_name_char;

						Link link_filter(link_filter_name, source_node_names[source_node_iter], output_name, weight_filter_name);
						link_filter.setModuleName(module_name);
						model.addLinks({ link_filter });

						++filter_height_iter;
					}
					++filter_width_iter;
				}
			}
		}
	}

	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addNormalization(Model<TensorT> & model, const std::string & name, const std::string & module_name, const std::vector<std::string>& source_node_names, const bool& specify_layers)
	{
		std::vector<std::string> node_names;
		std::string unity_weight_name, negunity_weight_name;

		// Make the mean/linear node
		char* mean_name_char = new char[512];
		sprintf(mean_name_char, "%s-Mean", name.data());
		std::string mean_name(mean_name_char);
		Node<TensorT> mean(mean_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), 
      std::make_shared<MeanOp<TensorT>>(MeanOp<TensorT>()), std::make_shared<MeanErrorOp<TensorT>>(MeanErrorOp<TensorT>()), std::make_shared<MeanWeightGradOp<TensorT>>(MeanWeightGradOp<TensorT>()));
		mean.setModuleName(module_name);
    if (specify_layers) mean.setLayerName(module_name + "-Mean");
		model.addNodes({ mean });
    delete[] mean_name_char;
		//node_names.push_back(mean_name);

		// Make the variance/inverse sqrt node
		char* variance_name_char = new char[512];
		sprintf(variance_name_char, "%s-Variance", name.data());
		std::string variance_name(variance_name_char);
		Node<TensorT> variance(variance_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<PowOp<TensorT>>(PowOp<TensorT>(-0.5)), std::make_shared<PowGradOp<TensorT>>(PowGradOp<TensorT>(-0.5)), 
      std::make_shared<VarModOp<TensorT>>(VarModOp<TensorT>()), std::make_shared<VarModErrorOp<TensorT>>(VarModErrorOp<TensorT>()), std::make_shared<VarModWeightGradOp<TensorT>>(VarModWeightGradOp<TensorT>()));
		variance.setModuleName(module_name);
    if (specify_layers) mean.setLayerName(module_name + "-Variance");
		model.addNodes({ variance });
    delete[] variance_name_char;

		for (const std::string& node_name : source_node_names) {
			// Make the source-mean nodes
			char* sourceMinMean_name_char = new char[512];
			sprintf(sourceMinMean_name_char, "%s-SourceMinMean", node_name.data());
			std::string sourceMinMean_name(sourceMinMean_name_char);
			Node<TensorT> sourceMinMean(sourceMinMean_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			sourceMinMean.setModuleName(module_name);
      if (specify_layers) sourceMinMean.setLayerName(module_name + "-SourceMinMean");
			model.addNodes({ sourceMinMean });
      delete[] sourceMinMean_name_char;

			// Make the normalized nodes
			char* normalized_name_char = new char[512];
			sprintf(normalized_name_char, "%s-Normalized", node_name.data());
			std::string normalized_name(normalized_name_char);
			Node<TensorT> normalized(normalized_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			normalized.setModuleName(module_name);
      if (specify_layers) normalized.setLayerName(module_name + "-Normalized");
			model.addNodes({ normalized });
			node_names.push_back(normalized_name);
      delete[] normalized_name_char;

			// Make the weights/links from source to mean
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", node_name, mean_name, specify_layers);
			char* sToM_link_name_char = new char[512];
			sprintf(sToM_link_name_char, "%s_to_%s", node_name.data(), mean_name.data());
			std::string sToM_link_name(sToM_link_name_char);
			Link sToM_link(sToM_link_name, node_name, mean_name, unity_weight_name);
			sToM_link.setModuleName(module_name);
			model.addLinks({ sToM_link });
      delete[] sToM_link_name_char;

			// Make the links from source to sourceMinMean
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", node_name, sourceMinMean_name, specify_layers);
			char* sToSMinM_link_name_char = new char[512];
			sprintf(sToSMinM_link_name_char, "%s_to_%s", node_name.data(), sourceMinMean_name.data());
			std::string sToSMinM_link_name(sToSMinM_link_name_char);
			Link sToSMinM_link(sToSMinM_link_name, node_name, sourceMinMean_name, unity_weight_name);
			sToSMinM_link.setModuleName(module_name);
			model.addLinks({ sToSMinM_link });
      delete[] sToSMinM_link_name_char;

			// Make the links from the mean to sourceMinMean
			negunity_weight_name = makeUnityWeight(model, -1.0, module_name, "%s_to_%s", mean_name, sourceMinMean_name, specify_layers);
			char* mToSMinM_link_name_char = new char[512];
			sprintf(mToSMinM_link_name_char, "%s_to_%s", mean_name.data(), sourceMinMean_name.data());
			std::string mToSMinM_link_name(mToSMinM_link_name_char);
			Link mToSMinM_link(mToSMinM_link_name, mean_name, sourceMinMean_name, negunity_weight_name);
			mToSMinM_link.setModuleName(module_name);
			model.addLinks({ mToSMinM_link });
      delete[] mToSMinM_link_name_char;

			// Make the links from sourceMinMean to variance
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", sourceMinMean_name, variance_name, specify_layers);
			char* sMinMToV_link_name_char = new char[512];
			sprintf(sMinMToV_link_name_char, "%s_to_%s", sourceMinMean_name.data(), variance_name.data());
			std::string sMinMToV_link_name(sMinMToV_link_name_char);
			Link sMinMToV_link(sMinMToV_link_name, sourceMinMean_name, variance_name, unity_weight_name);
			sMinMToV_link.setModuleName(module_name);
			model.addLinks({ sMinMToV_link });
      delete[] sMinMToV_link_name_char;

			// Make the weights/links from sourceMinMean to normalized
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", sourceMinMean_name, normalized_name, specify_layers);
			char* sMinMToN_link_name_char = new char[512];
			sprintf(sMinMToN_link_name_char, "%s_to_%s", sourceMinMean_name.data(), normalized_name.data());
			std::string sMinMToN_link_name(sMinMToN_link_name_char);
			Link sMinMToN_link(sMinMToN_link_name, sourceMinMean_name, normalized_name, unity_weight_name);
			sMinMToN_link.setModuleName(module_name);
			model.addLinks({ sMinMToN_link });
      delete[] sMinMToN_link_name_char;

			// Make the links from variance to normalized
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", variance_name, normalized_name);
			char* vToN_link_name_char = new char[512];
			sprintf(vToN_link_name_char, "%s_to_%s", variance_name.data(), normalized_name.data());
			std::string vToN_link_name(vToN_link_name_char);
			Link vToN_link(vToN_link_name, variance_name, normalized_name, unity_weight_name);
			vToN_link.setModuleName(module_name);
			model.addLinks({ vToN_link });
      delete[] vToN_link_name_char;
		}
		return node_names;
	}

  template<typename TensorT>
  std::vector<std::string> ModelBuilder<TensorT>::addUnitScale(Model<TensorT> & model, const std::string & name, const std::string & module_name, const std::vector<std::string>& source_node_names, const bool& specify_layers)
  {
    std::vector<std::string> node_names;
    std::string unity_weight_name, negunity_weight_name;

    // Make the max/linear node
    char* max_name_char = new char[512];
    sprintf(max_name_char, "%s-Max", name.data());
    std::string max_name(max_name_char);
    Node<TensorT> max(max_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<MaxOp<TensorT>>(MaxOp<TensorT>()), std::make_shared<MaxErrorOp<TensorT>>(MaxErrorOp<TensorT>()), std::make_shared<MaxWeightGradOp<TensorT>>(MaxWeightGradOp<TensorT>()));
    max.setModuleName(module_name);
    if (specify_layers) max.setLayerName(module_name + "-MinMax");
    model.addNodes({ max });
    delete[] max_name_char;

    // Make the min/linear node
    char* min_name_char = new char[512];
    sprintf(min_name_char, "%s-Min", name.data());
    std::string min_name(min_name_char);
    Node<TensorT> min(min_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<MinOp<TensorT>>(MinOp<TensorT>()), std::make_shared<MinErrorOp<TensorT>>(MinErrorOp<TensorT>()), std::make_shared<MinWeightGradOp<TensorT>>(MinWeightGradOp<TensorT>()));
    min.setModuleName(module_name);
    if (specify_layers) min.setLayerName(module_name + "-MinMax");
    model.addNodes({ min });
    delete[] min_name_char;

    // Make the sum/inverse sqrt node
    char* scalar_name_char = new char[512];
    sprintf(scalar_name_char, "%s-Scalar", name.data());
    std::string scalar_name(scalar_name_char);
    Node<TensorT> scalar(scalar_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<PowOp<TensorT>>(PowOp<TensorT>(-0.5)), std::make_shared<PowGradOp<TensorT>>(PowGradOp<TensorT>(-0.5)), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    scalar.setModuleName(module_name);
    if (specify_layers) max.setLayerName(module_name + "-Scalar");
    model.addNodes({ scalar });
    delete[] scalar_name_char;

    // Make the link from the max to the scalar node
    unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", max_name, scalar_name, specify_layers);
    char* maxToScalar_link_name_char = new char[512];
    sprintf(maxToScalar_link_name_char, "%s_to_%s", max_name.data(), scalar_name.data());
    std::string maxToScalar_link_name(maxToScalar_link_name_char);
    Link maxToScalar_link(maxToScalar_link_name, max_name, scalar_name, unity_weight_name);
    maxToScalar_link.setModuleName(module_name);
    model.addLinks({ maxToScalar_link });
    delete[] maxToScalar_link_name_char;

    // Make the link from the min to the scalar node
    unity_weight_name = makeUnityWeight(model, -1.0, module_name, "%s_to_%s", min_name, scalar_name, specify_layers);
    char* minToScalar_link_name_char = new char[512];
    sprintf(minToScalar_link_name_char, "%s_to_%s", min_name.data(), scalar_name.data());
    std::string minToScalar_link_name(minToScalar_link_name_char);
    Link minToScalar_link(minToScalar_link_name, min_name, scalar_name, unity_weight_name);
    minToScalar_link.setModuleName(module_name);
    model.addLinks({ minToScalar_link });
    delete[] minToScalar_link_name_char;

    for (const std::string& node_name : source_node_names) {
      // Make the unitScale nodes
      char* unitScale_name_char = new char[512];
      sprintf(unitScale_name_char, "%s-UnitScaled", node_name.data());
      std::string unitScale_name(unitScale_name_char);
      Node<TensorT> unitScale(unitScale_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
      unitScale.setModuleName(module_name);
      if (specify_layers) unitScale.setLayerName(module_name + "-UnitScaled");
      model.addNodes({ unitScale });
      node_names.push_back(unitScale_name);
      delete[] unitScale_name_char;

      // Make the weights/links from source to max
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", node_name, max_name, specify_layers);
      char* sToM_link_name_char = new char[512];
      sprintf(sToM_link_name_char, "%s_to_%s", node_name.data(), max_name.data());
      std::string sToM_link_name(sToM_link_name_char);
      Link sToM_link(sToM_link_name, node_name, max_name, unity_weight_name);
      sToM_link.setModuleName(module_name);
      model.addLinks({ sToM_link });
      delete[] sToM_link_name_char;

      // Make the weights/links from source to min
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", node_name, min_name, specify_layers);
      char* sToMin_link_name_char = new char[512];
      sprintf(sToMin_link_name_char, "%s_to_%s", node_name.data(), min_name.data());
      std::string sToMin_link_name(sToMin_link_name_char);
      Link sToMin_link(sToMin_link_name, node_name, min_name, unity_weight_name);
      sToMin_link.setModuleName(module_name);
      model.addLinks({ sToMin_link });
      delete[] sToMin_link_name_char;

      // Make the links from scalar to unitScale
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", scalar_name, unitScale_name);
      char* vToN_link_name_char = new char[512];
      sprintf(vToN_link_name_char, "%s_to_%s", scalar_name.data(), unitScale_name.data());
      std::string vToN_link_name(vToN_link_name_char);
      Link vToN_link(vToN_link_name, scalar_name, unitScale_name, unity_weight_name);
      vToN_link.setModuleName(module_name);
      model.addLinks({ vToN_link });
      delete[] vToN_link_name_char;
    }
    return node_names;
  }


  template<typename TensorT>
  std::vector<std::string> ModelBuilder<TensorT>::addLinearScale(Model<TensorT> & model, const std::string & name, const std::string & module_name, const std::vector<std::string>& source_node_names, const TensorT& range_min, const TensorT& range_max, const bool& specify_layers)
  {
    std::vector<std::string> node_names;
    std::string unity_weight_name, negunity_weight_name;

    // Make the max/linear node
    char* max_name_char = new char[512];
    sprintf(max_name_char, "%s-Max", name.data());
    std::string max_name(max_name_char);
    Node<TensorT> max(max_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<MaxOp<TensorT>>(MaxOp<TensorT>()), std::make_shared<MaxErrorOp<TensorT>>(MaxErrorOp<TensorT>()), std::make_shared<MaxWeightGradOp<TensorT>>(MaxWeightGradOp<TensorT>()));
    max.setModuleName(module_name);
    if (specify_layers) max.setLayerName(module_name + "-MinMax");
    model.addNodes({ max });
    delete[] max_name_char;

    // Make the min/linear node
    char* min_name_char = new char[512];
    sprintf(min_name_char, "%s-Min", name.data());
    std::string min_name(min_name_char);
    Node<TensorT> min(min_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<MinOp<TensorT>>(MinOp<TensorT>()), std::make_shared<MinErrorOp<TensorT>>(MinErrorOp<TensorT>()), std::make_shared<MinWeightGradOp<TensorT>>(MinWeightGradOp<TensorT>()));
    min.setModuleName(module_name);
    if (specify_layers) min.setLayerName(module_name + "-DomainMinMax");
    model.addNodes({ min });
    delete[] min_name_char;

    // Make the sum/inverse node
    char* scalar_name_char = new char[512];
    sprintf(scalar_name_char, "%s-Scalar", name.data());
    std::string scalar_name(scalar_name_char);
    Node<TensorT> scalar(scalar_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<InverseOp<TensorT>>(InverseOp<TensorT>()), std::make_shared<InverseGradOp<TensorT>>(InverseGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    scalar.setModuleName(module_name);
    if (specify_layers) max.setLayerName(module_name + "-DomainScalar");
    model.addNodes({ scalar });
    delete[] scalar_name_char;

    // Make the link from the max to the scalar node
    unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", max_name, scalar_name, specify_layers);
    char* maxToScalar_link_name_char = new char[512];
    sprintf(maxToScalar_link_name_char, "%s_to_%s", max_name.data(), scalar_name.data());
    std::string maxToScalar_link_name(maxToScalar_link_name_char);
    Link maxToScalar_link(maxToScalar_link_name, max_name, scalar_name, unity_weight_name);
    maxToScalar_link.setModuleName(module_name);
    model.addLinks({ maxToScalar_link });
    delete[] maxToScalar_link_name_char;

    // Make the link from the min to the scalar node
    unity_weight_name = makeUnityWeight(model, -1.0, module_name, "%s_to_%s", min_name, scalar_name, specify_layers);
    char* minToScalar_link_name_char = new char[512];
    sprintf(minToScalar_link_name_char, "%s_to_%s", min_name.data(), scalar_name.data());
    std::string minToScalar_link_name(minToScalar_link_name_char);
    Link minToScalar_link(minToScalar_link_name, min_name, scalar_name, unity_weight_name);
    minToScalar_link.setModuleName(module_name);
    model.addLinks({ minToScalar_link });
    delete[] minToScalar_link_name_char;

    // Make the range max minus min bias
    char* rangeMaxMinBias_name_char = new char[512];
    sprintf(rangeMaxMinBias_name_char, "%s-RangeMaxMinBias", module_name.data());
    std::string rangeMaxMinBias_name(rangeMaxMinBias_name_char);
    Node<TensorT> rangeMaxMinBias(rangeMaxMinBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    rangeMaxMinBias.setModuleName(module_name);
    if (specify_layers) rangeMaxMinBias.setLayerName(module_name + "-RangeMaxMinBias");
    model.addNodes({ rangeMaxMinBias });
    delete[] rangeMaxMinBias_name_char;

    // Make the range min bias
    char* rangeMinBias_name_char = new char[512];
    sprintf(rangeMinBias_name_char, "%s-RangeMinBias", module_name.data());
    std::string rangeMinBias_name(rangeMinBias_name_char);
    Node<TensorT> rangeMinBias(rangeMinBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
    rangeMinBias.setModuleName(module_name);
    if (specify_layers) rangeMinBias.setLayerName(module_name + "-RangeMinBias");
    model.addNodes({ rangeMinBias });
    delete[] rangeMinBias_name_char;

    for (const std::string& node_name : source_node_names) {
      // Make the dMinOffset nodes
      char* dMinOffset_name_char = new char[512];
      sprintf(dMinOffset_name_char, "%s-DomainMinOffset", node_name.data());
      std::string dMinOffset_name(dMinOffset_name_char);
      Node<TensorT> dMinOffset(dMinOffset_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
      dMinOffset.setModuleName(module_name);
      if (specify_layers) dMinOffset.setLayerName(module_name + "-DomainMinOffset");
      model.addNodes({ dMinOffset });
      delete[] dMinOffset_name_char;

      // Make the weights/links from source to max
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", node_name, max_name, specify_layers);
      char* sToM_link_name_char = new char[512];
      sprintf(sToM_link_name_char, "%s_to_%s", node_name.data(), max_name.data());
      std::string sToM_link_name(sToM_link_name_char);
      Link sToM_link(sToM_link_name, node_name, max_name, unity_weight_name);
      sToM_link.setModuleName(module_name);
      model.addLinks({ sToM_link });
      delete[] sToM_link_name_char;

      // Make the weights/links from source to min
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", node_name, min_name, specify_layers);
      char* sToMin_link_name_char = new char[512];
      sprintf(sToMin_link_name_char, "%s_to_%s", node_name.data(), min_name.data());
      std::string sToMin_link_name(sToMin_link_name_char);
      Link sToMin_link(sToMin_link_name, node_name, min_name, unity_weight_name);
      sToMin_link.setModuleName(module_name);
      model.addLinks({ sToMin_link });
      delete[] sToMin_link_name_char;

      // Make the weights/links from the min to the dMinOffset
      unity_weight_name = makeUnityWeight(model, -1.0, module_name, "%s_to_%s", min_name, dMinOffset_name, specify_layers);
      char* minToDMinOffset_link_name_char = new char[512];
      sprintf(minToDMinOffset_link_name_char, "%s_to_%s", min_name.data(), dMinOffset_name.data());
      std::string minToDMinOffset_link_name(minToDMinOffset_link_name_char);
      Link minToDMinOffset_link(minToDMinOffset_link_name, min_name, dMinOffset_name, unity_weight_name);
      minToDMinOffset_link.setModuleName(module_name);
      model.addLinks({ minToDMinOffset_link });
      delete[] minToDMinOffset_link_name_char;

      // Make the weights/links from the source to the dMinOffset
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", node_name, dMinOffset_name, specify_layers);
      char* sToDMinOffset_link_name_char = new char[512];
      sprintf(sToDMinOffset_link_name_char, "%s_to_%s", node_name.data(), dMinOffset_name.data());
      std::string sToDMinOffset_link_name(sToDMinOffset_link_name_char);
      Link sToDMinOffset_link(sToDMinOffset_link_name, node_name, dMinOffset_name, unity_weight_name);
      sToDMinOffset_link.setModuleName(module_name);
      model.addLinks({ sToDMinOffset_link });
      delete[] sToDMinOffset_link_name_char;

      // Make the domainScale node
      char* dScale_name_char = new char[512];
      sprintf(dScale_name_char, "%s-DomainScaled", node_name.data());
      std::string dScale_name(dScale_name_char);
      Node<TensorT> dScale(dScale_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
      dScale.setModuleName(module_name);
      if (specify_layers) dScale.setLayerName(module_name + "-DomainScaled");
      model.addNodes({ dScale });
      delete[] dScale_name_char;

      // Make the links/weights from the scalar to the domainScale node
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", scalar_name, dScale_name, specify_layers);
      char* scalarToDScale_link_name_char = new char[512];
      sprintf(scalarToDScale_link_name_char, "%s_to_%s", scalar_name.data(), dScale_name.data());
      std::string scalarToDScale_link_name(scalarToDScale_link_name_char);
      Link scalarToDScale_link(scalarToDScale_link_name, scalar_name, dScale_name, unity_weight_name);
      scalarToDScale_link.setModuleName(module_name);
      model.addLinks({ scalarToDScale_link });
      delete[] scalarToDScale_link_name_char;

      // Make the links/weights from the dMinOffset to the domainScale nodes
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", dMinOffset_name, dScale_name, specify_layers);
      char* dMinOffsetToDScale_link_name_char = new char[512];
      sprintf(dMinOffsetToDScale_link_name_char, "%s_to_%s", dMinOffset_name.data(), dScale_name.data());
      std::string dMinOffsetToDScale_link_name(dMinOffsetToDScale_link_name_char);
      Link dMinOffsetToDScale_link(dMinOffsetToDScale_link_name, dMinOffset_name, dScale_name, unity_weight_name);
      dMinOffsetToDScale_link.setModuleName(module_name);
      model.addLinks({ dMinOffsetToDScale_link });
      delete[] dMinOffsetToDScale_link_name_char;

      // Make the rangeMaxMinScale node
      char* rangeMaxMinScale_name_char = new char[512];
      sprintf(rangeMaxMinScale_name_char, "%s-RangeMaxMinScale", node_name.data());
      std::string rangeMaxMinScale_name(rangeMaxMinScale_name_char);
      Node<TensorT> rangeMaxMinScale(rangeMaxMinScale_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
      rangeMaxMinScale.setModuleName(module_name);
      if (specify_layers) rangeMaxMinScale.setLayerName(module_name + "-RangeMaxMinScale");
      model.addNodes({ rangeMaxMinScale });
      delete[] rangeMaxMinScale_name_char;

      // Make the links/weights from the rangeMaxMin to the rangeMaxMinScale node
      char* rangeMaxMinToRangeMaxMinScale_weight_name_char = new char[512];
      sprintf(rangeMaxMinToRangeMaxMinScale_weight_name_char, "%s_to_%s", rangeMaxMinBias_name.data(), rangeMaxMinScale_name.data());
      std::string rangeMaxMinToRangeMaxMinScale_weight_name(rangeMaxMinToRangeMaxMinScale_weight_name_char);
      std::shared_ptr<WeightInitOp<TensorT>> rangeMaxMinToRangeMaxMinScale_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(range_max - range_min));
      std::shared_ptr<SolverOp<TensorT>>  rangeMaxMinToRangeMaxMinScale_solver = std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>());
      Weight<TensorT> rangeMaxMinToRangeMaxMinScale_weight(rangeMaxMinToRangeMaxMinScale_weight_name, rangeMaxMinToRangeMaxMinScale_weight_init, rangeMaxMinToRangeMaxMinScale_solver);
      rangeMaxMinToRangeMaxMinScale_weight.setModuleName(module_name);
      model.addWeights({ rangeMaxMinToRangeMaxMinScale_weight });
      delete[] rangeMaxMinToRangeMaxMinScale_weight_name_char;

      char* rangeMaxMinToRangeMaxMinScale_link_name_char = new char[512];
      sprintf(rangeMaxMinToRangeMaxMinScale_link_name_char, "%s_to_%s", rangeMaxMinBias_name.data(), rangeMaxMinScale_name.data());
      std::string rangeMaxMinToRangeMaxMinScale_link_name(rangeMaxMinToRangeMaxMinScale_link_name_char);
      Link rangeMaxMinToRangeMaxMinScale_link(rangeMaxMinToRangeMaxMinScale_link_name, rangeMaxMinBias_name, rangeMaxMinScale_name, rangeMaxMinToRangeMaxMinScale_weight_name);
      rangeMaxMinToRangeMaxMinScale_link.setModuleName(module_name);
      model.addLinks({ rangeMaxMinToRangeMaxMinScale_link });
      delete[] rangeMaxMinToRangeMaxMinScale_link_name_char;

      // Make the links/weights from the domainScale nodes to the rangeMaxMinScale node
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", dScale_name, rangeMaxMinScale_name, specify_layers);
      char* dScaleToRMaxMinScale_link_name_char = new char[512];
      sprintf(dScaleToRMaxMinScale_link_name_char, "%s_to_%s", dScale_name.data(), rangeMaxMinScale_name.data());
      std::string dScaleToRMaxMinScale_link_name(dScaleToRMaxMinScale_link_name_char);
      Link dScaleToRMaxMinScale_link(dScaleToRMaxMinScale_link_name, dScale_name, rangeMaxMinScale_name, unity_weight_name);
      dScaleToRMaxMinScale_link.setModuleName(module_name);
      model.addLinks({ dScaleToRMaxMinScale_link });
      delete[] dScaleToRMaxMinScale_link_name_char;

      // Make the LinearScaleFunctor node
      char* linearScale_name_char = new char[512];
      sprintf(linearScale_name_char, "%s-LinearScaleFunctor", node_name.data());
      std::string linearScale_name(linearScale_name_char);
      Node<TensorT> linearScale(linearScale_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
      linearScale.setModuleName(module_name);
      if (specify_layers) linearScale.setLayerName(module_name + "-LinearScaleFunctor");
      model.addNodes({ linearScale });
      node_names.push_back(linearScale_name);
      delete[] linearScale_name_char;

      // Make the links/weights from the rangeMin node to the LinearScaleFunctor node
      char* rangeMinToLinearScale_weight_name_char = new char[512];
      sprintf(rangeMinToLinearScale_weight_name_char, "%s_to_%s", rangeMinBias_name.data(), linearScale_name.data());
      std::string rangeMinToLinearScale_weight_name(rangeMinToLinearScale_weight_name_char);
      std::shared_ptr<WeightInitOp<TensorT>> rangeMinToLinearScale_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(range_min));
      std::shared_ptr<SolverOp<TensorT>>  rangeMinToLinearScale_solver = std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>());
      Weight<TensorT> rangeMinToLinearScale_weight(rangeMinToLinearScale_weight_name, rangeMinToLinearScale_weight_init, rangeMinToLinearScale_solver);
      rangeMinToLinearScale_weight.setModuleName(module_name);
      model.addWeights({ rangeMinToLinearScale_weight });
      delete[] rangeMinToLinearScale_weight_name_char;

      char* rangeMinToLinearScale_link_name_char = new char[512];
      sprintf(rangeMinToLinearScale_link_name_char, "%s_to_%s", rangeMinBias_name.data(), linearScale_name.data());
      std::string rangeMinToLinearScale_link_name(rangeMinToLinearScale_link_name_char);
      Link rangeMinToLinearScale_link(rangeMinToLinearScale_link_name, rangeMinBias_name, linearScale_name, rangeMinToLinearScale_weight_name);
      rangeMinToLinearScale_link.setModuleName(module_name);
      model.addLinks({ rangeMinToLinearScale_link });
      delete[] rangeMinToLinearScale_link_name_char;

      // Make the links/weights from the rangeMaxMinScale node to the LinearScaleFunctor node
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", rangeMaxMinScale_name, linearScale_name, specify_layers);
      char* rMaxMinScaleToLinearScale_link_name_char = new char[512];
      sprintf(rMaxMinScaleToLinearScale_link_name_char, "%s_to_%s", rangeMaxMinScale_name.data(), linearScale_name.data());
      std::string rMaxMinScaleToLinearScale_link_name(rMaxMinScaleToLinearScale_link_name_char);
      Link rMaxMinScaleToLinearScale_link(rMaxMinScaleToLinearScale_link_name, rangeMaxMinScale_name, linearScale_name, unity_weight_name);
      rMaxMinScaleToLinearScale_link.setModuleName(module_name);
      model.addLinks({ rMaxMinScaleToLinearScale_link });
      delete[] rMaxMinScaleToLinearScale_link_name_char;

    }
    return node_names;
  }

	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addGaussianEncoding(Model<TensorT> & model, const std::string & name, const std::string & module_name, const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names, const bool& specify_layer)
	{
		std::vector<std::string> node_names;
		std::string unity_weight_name, scalar_weight_name;

		assert(mu_node_names.size() == logvar_node_names.size());

		// Specify the layer names for the mu and logvar nodes in order
		// to ensure they are placed on different tensors during model interpretation
		for (const std::string& node_name : mu_node_names) {
			model.nodes_.at(node_name)->setLayerName("VAE_Mu");
		}
		for (const std::string& node_name : logvar_node_names) {
			model.nodes_.at(node_name)->setLayerName("VAE_LogVar");
		}

		for (size_t i = 0; i < logvar_node_names.size(); ++i) {
			// Make the logVar scalar nodes
			char* logvarScale_name_char = new char[512];
			sprintf(logvarScale_name_char, "%s-Scalar", logvar_node_names[i].data());
			std::string logvarScale_name(logvarScale_name_char);
			Node<TensorT> logvarScale(logvarScale_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<ExponentialOp<TensorT>>(ExponentialOp<TensorT>()), std::make_shared<ExponentialGradOp<TensorT>>(ExponentialGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			logvarScale.setModuleName(module_name);
			if (specify_layer) logvarScale.setLayerName(module_name + "-Scalar");
			model.addNodes({ logvarScale });
      delete[] logvarScale_name_char;
			//node_names.push_back(logvarScale_name);

			// Make the links from logvar to the scalar node
			scalar_weight_name = makeUnityWeight(model, 0.5, module_name, "%s_to_%s", logvar_node_names[i], logvarScale_name);
			char* lvToS_link_name_char = new char[512];
			sprintf(lvToS_link_name_char, "%s_to_%s", logvar_node_names[i].data(), logvarScale_name.data());
			std::string lvToS_link_name(lvToS_link_name_char);
			Link lvToS_link(lvToS_link_name, logvar_node_names[i], logvarScale_name, scalar_weight_name);
			lvToS_link.setModuleName(module_name);
			model.addLinks({ lvToS_link });
      delete[] lvToS_link_name_char;

			// Make the sampler nodes
			char* sampler_name_char = new char[512];
			sprintf(sampler_name_char, "%s_%012d-Sampler", name.data(), i);
			std::string sampler_name(sampler_name_char);
			Node<TensorT> sampler(sampler_name, NodeType::input, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			sampler.setModuleName(module_name);
			if (specify_layer) sampler.setLayerName(module_name + "-Sampler");
			model.addNodes({ sampler });
      delete[] sampler_name_char;
			//node_names.push_back(sampler_name);

			// Make the stddev nodes
			char* stddev_name_char = new char[512];
			sprintf(stddev_name_char, "%s-StdDev", logvar_node_names[i].data());
			std::string stddev_name(stddev_name_char);
			Node<TensorT> stddev(stddev_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			stddev.setModuleName(module_name);
			if (specify_layer) stddev.setLayerName(module_name + "-StdDev");
			model.addNodes({ stddev });
      delete[] stddev_name_char;
			//node_names.push_back(stddev_name);

			// Make the links from logvar scalar node to the std dev node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", logvarScale_name, stddev_name);
			char* ScToStdev_link_name_char = new char[512];
			sprintf(ScToStdev_link_name_char, "%s_to_%s", logvarScale_name.data(), stddev_name.data());
			std::string ScToStdev_link_name(ScToStdev_link_name_char);
			Link ScToStdev_link(ScToStdev_link_name, logvarScale_name, stddev_name, unity_weight_name);
			ScToStdev_link.setModuleName(module_name);
			model.addLinks({ ScToStdev_link });
      delete[] ScToStdev_link_name_char;

			// Make the links from sampler to the std dev node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", sampler_name, stddev_name);
			char* SToStdev_link_name_char = new char[512];
			sprintf(SToStdev_link_name_char, "%s_to_%s", sampler_name.data(), stddev_name.data());
			std::string SToStdev_link_name(SToStdev_link_name_char);
			Link SToStdev_link(SToStdev_link_name, sampler_name, stddev_name, unity_weight_name);
			SToStdev_link.setModuleName(module_name);
			model.addLinks({ SToStdev_link });
      delete[] SToStdev_link_name_char;

			// Make the output nodes
			char* output_name_char = new char[512];
			sprintf(output_name_char, "%s_%012d", name.data(), i);
			std::string output_name(output_name_char);
			Node<TensorT> output(output_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			output.setModuleName(module_name);
			if (specify_layer) output.setLayerName(module_name);
			model.addNodes({ output });
			node_names.push_back(output_name);
      delete[] output_name_char;

			// Make the links from std dev node to the output node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", stddev_name, output_name);
			char* StDevToOutput_link_name_char = new char[512];
			sprintf(StDevToOutput_link_name_char, "%s_to_%s", stddev_name.data(), output_name.data());
			std::string StDevToOutput_link_name(StDevToOutput_link_name_char);
			Link StDevToOutput_link(StDevToOutput_link_name, stddev_name, output_name, unity_weight_name);
			StDevToOutput_link.setModuleName(module_name);
			model.addLinks({ StDevToOutput_link });
      delete[] StDevToOutput_link_name_char;

			// Make the links from mean to the output node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", mu_node_names[i], output_name);
			char* muToOutput_link_name_char = new char[512];
			sprintf(muToOutput_link_name_char, "%s_to_%s", mu_node_names[i].data(), output_name.data());
			std::string muToOutput_link_name(muToOutput_link_name_char);
			Link muToOutput_link(muToOutput_link_name, mu_node_names[i], output_name, unity_weight_name);
			muToOutput_link.setModuleName(module_name);
			model.addLinks({ muToOutput_link });
      delete[] muToOutput_link_name_char;
		}
		return node_names;
	}

	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addCategoricalEncoding(Model<TensorT> & model, const std::string & name, const std::string & module_name, 
		const std::vector<std::string>& alpha_node_names, const bool& specify_layer)
	{
		std::vector<std::string> softmax_args_names;
		std::string unity_weight_name, scalar_weight_name;

		for (size_t i = 0; i < alpha_node_names.size(); ++i) {

			// Make the sampler nodes
			char* sampler_name_char = new char[512];
			sprintf(sampler_name_char, "%s_%012d-GumbelSampler", name.data(), i);
			std::string sampler_name(sampler_name_char);
			Node<TensorT> sampler(sampler_name, NodeType::input, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			sampler.setModuleName(module_name);
      if (specify_layer) sampler.setLayerName(module_name + "-GumbelSampler");
			model.addNodes({ sampler });
      delete[] sampler_name_char;
			//node_names.push_back(sampler_name);

			// Make the logAlphaScale node
			char* logalphaScale_name_char = new char[512];
			sprintf(logalphaScale_name_char, "%s_%012d-Scalar", name.data(), i);
			std::string logalphaScale_name(logalphaScale_name_char);
			Node<TensorT> logalphaScale(logalphaScale_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LogOp<TensorT>>(LogOp<TensorT>()), std::make_shared<LogGradOp<TensorT>>(LogGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			logalphaScale.setModuleName(module_name);
			if (specify_layer) logalphaScale.setLayerName(module_name + "-Scalar");
			model.addNodes({ logalphaScale });
			delete[] logalphaScale_name_char;
			//node_names.push_back(logalphaScale_name);

			// Make the LogAlphaSampler nodes
			char* logAlphaSampler_name_char = new char[512];
			sprintf(logAlphaSampler_name_char, "%s_%012d-LogAlphaSampler", name.data(), i);
			std::string logAlphaSampler_name(logAlphaSampler_name_char);
			Node<TensorT> logAlphaSampler(logAlphaSampler_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			logAlphaSampler.setModuleName(module_name);
			if (specify_layer) logAlphaSampler.setLayerName(module_name + "-LogAlphaSampler");
			model.addNodes({ logAlphaSampler });
      delete[] logAlphaSampler_name_char;
			//node_names.push_back(logAlphaSampler);

			// Make the links from the logAlpha node and sampler node to the logAlphaSamplerSum node
			scalar_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", sampler_name, logAlphaSampler_name);
			char* lsToLAS_link_name_char = new char[512];
			sprintf(lsToLAS_link_name_char, "%s_to_%s", sampler_name.data(), logAlphaSampler_name.data());
			std::string lsToLAS_link_name(lsToLAS_link_name_char);
			Link lsToLAS_link(lsToLAS_link_name, sampler_name, logAlphaSampler_name, scalar_weight_name);
			lsToLAS_link.setModuleName(module_name);
			model.addLinks({ lsToLAS_link });
      delete[] lsToLAS_link_name_char;

			// Make the links from the alpha node to the logAlpha node
			scalar_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", alpha_node_names[i], logalphaScale_name);
			char* laToLA_link_name_char = new char[512];
			sprintf(laToLA_link_name_char, "%s_to_%s", alpha_node_names[i].data(), logalphaScale_name.data());
			std::string laToLA_link_name(laToLA_link_name_char);
			Link laToLA_link(laToLA_link_name, alpha_node_names[i], logalphaScale_name, scalar_weight_name);
			laToLA_link.setModuleName(module_name);
			model.addLinks({ laToLA_link });
			delete[] laToLA_link_name_char;

			// Make the links from the logAlpha node and sampler node to the logAlphaSamplerSum node
			//scalar_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", alpha_node_names[i], logAlphaSampler_name);
			scalar_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", logalphaScale_name, logAlphaSampler_name);
			char* laToLAS_link_name_char = new char[512];
			//sprintf(laToLAS_link_name_char, "%s_to_%s", alpha_node_names[i].data(), logAlphaSampler_name.data());
			sprintf(laToLAS_link_name_char, "%s_to_%s", logalphaScale_name.data(), logAlphaSampler_name.data());
			std::string laToLAS_link_name(laToLAS_link_name_char);
			//Link laToLAS_link(laToLAS_link_name, alpha_node_names[i], logAlphaSampler_name, scalar_weight_name);
			Link laToLAS_link(laToLAS_link_name, logalphaScale_name, logAlphaSampler_name, scalar_weight_name);
			laToLAS_link.setModuleName(module_name);
			model.addLinks({ laToLAS_link });
      delete[] laToLAS_link_name_char;

			// Make the inverse tau nodes
			char* tau_name_char = new char[512];
			sprintf(tau_name_char, "%s_%012d-InverseTau", name.data(), i);
			std::string tau_name(tau_name_char);
			Node<TensorT> tau(tau_name, NodeType::input, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			tau.setModuleName(module_name);
			if (specify_layer) tau.setLayerName(module_name + "-InverseTau");
			model.addNodes({ tau });
      delete[] tau_name_char;
			//node_names.push_back(tau_name)

			// Make the intermediate nodes before the softmax
			char* softmaxArgs_name_char = new char[512];
			sprintf(softmaxArgs_name_char, "%s_%012d-SoftmaxArgs", name.data(), i);
			std::string softmaxArgs_name(softmaxArgs_name_char);
			Node<TensorT> softmaxArgs(softmaxArgs_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			softmaxArgs.setModuleName(module_name);
			if (specify_layer) softmaxArgs.setLayerName(module_name + "-SoftmaxArgs");
			model.addNodes({ softmaxArgs });
			softmax_args_names.push_back(softmaxArgs_name);
      delete[] softmaxArgs_name_char;

			// Make the links from the LogAlphaSampler node to the SoftmaxArgs node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", logAlphaSampler_name, softmaxArgs_name);
			char* LasToSa_link_name_char = new char[512];
			sprintf(LasToSa_link_name_char, "%s_to_%s", logAlphaSampler_name.data(), softmaxArgs_name.data());
			std::string LasToSa_link_name(LasToSa_link_name_char);
			Link LasToSa_link(LasToSa_link_name, logAlphaSampler_name, softmaxArgs_name, unity_weight_name);
			LasToSa_link.setModuleName(module_name);
			model.addLinks({ LasToSa_link });
      delete[] LasToSa_link_name_char;

			// Make the links from the inverseTau node to the SoftmaxArgs node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", tau_name, softmaxArgs_name);
			char* ItToSa_link_name_char = new char[512];
			sprintf(ItToSa_link_name_char, "%s_to_%s", tau_name.data(), softmaxArgs_name.data());
			std::string ItToSa_link_name(ItToSa_link_name_char);
			Link ItToSa_link(ItToSa_link_name, tau_name, softmaxArgs_name, unity_weight_name);
			ItToSa_link.setModuleName(module_name);
			model.addLinks({ ItToSa_link });
      delete[] ItToSa_link_name_char;
		}

		// Make the softmax layer
		std::vector<std::string> node_names = addStableSoftMax(model, name + "-" + "SoftMax", module_name, softmax_args_names, true);

		return node_names;
	}

	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addDiscriminator(Model<TensorT> & model, const std::string & name, const std::string & module_name, const std::vector<std::string>& encoding_node_names)
	{
    // TODO: add specify layer

		std::vector<std::string> node_names;
		std::string unity_weight_name, negative_weight_name;

		//// Create the unity weight
		//char* unity_weight_name_char = new char[512];
		//sprintf(unity_weight_name_char, "%s_Unity", name.data());
		//std::string unity_weight_name(unity_weight_name_char);
		//Weight<TensorT> unity_weight(unity_weight_name, std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
		//unity_weight.setModuleName(module_name);
		//model.addWeights({ unity_weight });

		//// Create the negative unity weight
		//char* negative_weight_name_char = new char[512];
		//sprintf(negative_weight_name_char, "%s_NegUnity", name.data());
		//std::string negative_weight_name(negative_weight_name_char);
		//Weight<TensorT> negative_weight(negative_weight_name, std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(-1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
		//negative_weight.setModuleName(module_name);
		//model.addWeights({ negative_weight });

		for (size_t i = 0; i < encoding_node_names.size(); ++i) {
			// Make the output node
			char* output_name_char = new char[512];
			sprintf(output_name_char, "%s-Output-%012d", name.data(), i);
			std::string output_name(output_name_char);
			Node<TensorT> output(output_name, NodeType::output, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			output.setModuleName(module_name);
			model.addNodes({ output });
			node_names.push_back(output_name);
      delete[] output_name_char;

			// Make the links from the encoding to the output node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", encoding_node_names[i], output_name);
			char* lvToS_link_name_char = new char[512];
			sprintf(lvToS_link_name_char, "%s_to_%s", encoding_node_names[i].data(), output_name.data());
			std::string lvToS_link_name(lvToS_link_name_char);
			Link lvToS_link(lvToS_link_name, encoding_node_names[i], output_name, unity_weight_name);
			lvToS_link.setModuleName(module_name);
			model.addLinks({ lvToS_link });
      delete[] lvToS_link_name_char;

			// Make the sampler nodes
			char* sampler_name_char = new char[512];
			sprintf(sampler_name_char, "%s-Sampler-%012d", name.data(), i);
			std::string sampler_name(sampler_name_char);
			Node<TensorT> sampler(sampler_name, NodeType::input, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			sampler.setModuleName(module_name);
			model.addNodes({ sampler });
      delete[] sampler_name_char;

			// Make the links from the sampler node to the output node
			negative_weight_name = makeUnityWeight(model, -1.0, module_name, "%s_to_%s", sampler_name, output_name);
			char* ScToStdev_link_name_char = new char[512];
			sprintf(ScToStdev_link_name_char, "%s_to_%s", sampler_name.data(), output_name.data());
			std::string ScToStdev_link_name(ScToStdev_link_name_char);
			Link ScToStdev_link(ScToStdev_link_name, sampler_name, output_name, negative_weight_name);
			ScToStdev_link.setModuleName(module_name);
			model.addLinks({ ScToStdev_link });
      delete[] ScToStdev_link_name_char;
		}
		return node_names;
	}

	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addLSTM(Model<TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names, const int & n_blocks, const int & n_cells,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& node_integration, const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error, const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, bool forget_gate, int block_version, const bool& specify_layer, bool specify_cyclic_pairs)
	{
		std::vector<std::string> node_names;

		for (int block_iter = 0; block_iter < n_blocks; ++block_iter) {
			// Make the LSTM cell
			char* name_char = new char[512];
			sprintf(name_char, "%s-%012d", name.data(), block_iter);
			std::string node_name(name_char);
      delete[] name_char;
			if (block_version == 1) {
				std::vector<std::string> output_node_names = addLSTMBlock1(model, node_name, module_name, source_node_names, n_cells, node_activation, node_activation_grad,
					node_integration, node_integration_error, node_integration_weight_grad,
					weight_init, solver, drop_out_prob, drop_connection_prob, biases, forget_gate, specify_layer, specify_cyclic_pairs);
				for (const std::string& node_name : output_node_names) node_names.push_back(node_name);
			}
			else if (block_version == 2) {
				std::vector<std::string> output_node_names = addLSTMBlock2(model, node_name, module_name, source_node_names, n_cells, node_activation, node_activation_grad,
					node_integration, node_integration_error, node_integration_weight_grad,
					weight_init, solver, drop_out_prob, drop_connection_prob, biases, forget_gate, specify_layer, specify_cyclic_pairs);
				for (const std::string& node_name : output_node_names) node_names.push_back(node_name);
			}
		}
		return node_names;
	}

	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addLSTMBlock1(
		Model<TensorT> & model, const std::string & name, const std::string& module_name,
		const std::vector<std::string>& source_node_names,
		const int & n_cells,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& node_integration, const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error, const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, bool forget_gate, const bool& specify_layer, bool specify_cyclic_pairs)
	{
		std::vector<std::string> node_names;
		std::string unity_weight_name;

		// Make the input gate node
		char* blockGateInput_name_char = new char[512];
		sprintf(blockGateInput_name_char, "%s-BlockGateInput", name.data());
		std::string blockGateInput_name(blockGateInput_name_char);
		Node<TensorT> blockGateInput(blockGateInput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()), node_integration, node_integration_error, node_integration_weight_grad);
		blockGateInput.setModuleName(module_name);
		if (specify_layer) blockGateInput.setLayerName(module_name + "-BlockGateInput");
		model.addNodes({ blockGateInput });
    delete[] blockGateInput_name_char;

		// Make the output gate node 
		char* blockGateOutput_name_char = new char[512];
		sprintf(blockGateOutput_name_char, "%s-BlockGateOutput", name.data());
		std::string blockGateOutput_name(blockGateOutput_name_char);
		Node<TensorT> blockGateOutput(blockGateOutput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()), node_integration, node_integration_error, node_integration_weight_grad);
		blockGateOutput.setModuleName(module_name);
		if (specify_layer) blockGateOutput.setLayerName(module_name + "-BlockGateOutput");
		model.addNodes({ blockGateOutput });
    delete[] blockGateOutput_name_char;

		std::string blockGateForget_name;
		if (forget_gate) {
			// Make the forget gate node
			char* blockGateForget_name_char = new char[512];
			sprintf(blockGateForget_name_char, "%s-BlockGateForget", name.data());
			blockGateForget_name = std::string(blockGateForget_name_char);
			Node<TensorT> blockGateForget(blockGateForget_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()), node_integration, node_integration_error, node_integration_weight_grad);
			blockGateForget.setModuleName(module_name);
			if (specify_layer) blockGateForget.setLayerName(module_name + "-BlockGateForget");
			model.addNodes({ blockGateForget });
      delete[] blockGateForget_name_char;
		}

		if (biases) {  // biases, links, and weights for input gate, forget gate, and output gate
			// Make the input gate bias nodes
			char* iGateBias_name_char = new char[512];
			sprintf(iGateBias_name_char, "%s-bias", blockGateInput_name.data());
			std::string iGateBias_name(iGateBias_name_char);
			Node<TensorT> iGateBias(iGateBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			iGateBias.setModuleName(module_name);
			model.addNodes({ iGateBias });
      delete[] iGateBias_name_char;

			// Make the link between input gate bias node to input gate node
			char* weight_iGateBias_name_char = new char[512];
			sprintf(weight_iGateBias_name_char, "%s_to_%s", iGateBias_name.data(), blockGateInput_name.data());
			std::string weight_iGateBias_name(weight_iGateBias_name_char);
      delete[] weight_iGateBias_name_char;

			char* link_iGateBias_name_char = new char[512];
			sprintf(link_iGateBias_name_char, "%s_to_%s", iGateBias_name.data(), blockGateInput_name.data());
			std::string link_iGateBias_name(link_iGateBias_name_char);
      delete[] link_iGateBias_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  iGateBias_weight_init;
			iGateBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
			std::shared_ptr<SolverOp<TensorT>>  iGateBias_solver = solver;
			Weight<TensorT> weight_iGateBias(weight_iGateBias_name, iGateBias_weight_init, iGateBias_solver);
			weight_iGateBias.setModuleName(module_name);
			Link link_iGateBias(link_iGateBias_name, iGateBias_name, blockGateInput_name, weight_iGateBias_name);
			link_iGateBias.setModuleName(module_name);

			model.addWeights({ weight_iGateBias });
			model.addLinks({ link_iGateBias });

			if (forget_gate) {
				// Make the forget gate bias nodes
				char* fGateBias_name_char = new char[512];
				sprintf(fGateBias_name_char, "%s-bias", blockGateForget_name.data());
				std::string fGateBias_name(fGateBias_name_char);
				Node<TensorT> fGateBias(fGateBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
				fGateBias.setModuleName(module_name);
				model.addNodes({ fGateBias });
        delete[] fGateBias_name_char;

				// Make the link between forget gate bias node to forget gate node
				char* weight_fGateBias_name_char = new char[512];
				sprintf(weight_fGateBias_name_char, "%s_to_%s", fGateBias_name.data(), blockGateForget_name.data());
				std::string weight_fGateBias_name(weight_fGateBias_name_char);
        delete[] weight_fGateBias_name_char;

				char* link_fGateBias_name_char = new char[512];
				sprintf(link_fGateBias_name_char, "%s_to_%s", fGateBias_name.data(), blockGateForget_name.data());
				std::string link_fGateBias_name(link_fGateBias_name_char);
        delete[] link_fGateBias_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  fGateBias_weight_init;
				fGateBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
				std::shared_ptr<SolverOp<TensorT>>  fGateBias_solver = solver;
				Weight<TensorT> weight_fGateBias(weight_fGateBias_name, fGateBias_weight_init, fGateBias_solver);
				weight_fGateBias.setModuleName(module_name);
				Link link_fGateBias(link_fGateBias_name, fGateBias_name, blockGateForget_name, weight_fGateBias_name);
				link_fGateBias.setModuleName(module_name);

				model.addWeights({ weight_fGateBias });
				model.addLinks({ link_fGateBias });
			}

			// Make the output gate bias nodes
			char* oGateBias_name_char = new char[512];
			sprintf(oGateBias_name_char, "%s-bias", blockGateOutput_name.data());
			std::string oGateBias_name(oGateBias_name_char);
			Node<TensorT> oGateBias(oGateBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			oGateBias.setModuleName(module_name);
			model.addNodes({ oGateBias });
      delete[] oGateBias_name_char;

			// Make the link between output gate bias node to output gate node
			char* weight_oGateBias_name_char = new char[512];
			sprintf(weight_oGateBias_name_char, "%s_to_%s", oGateBias_name.data(), blockGateOutput_name.data());
			std::string weight_oGateBias_name(weight_oGateBias_name_char);
      delete[] weight_oGateBias_name_char;

			char* link_oGateBias_name_char = new char[512];
			sprintf(link_oGateBias_name_char, "%s_to_%s", oGateBias_name.data(), blockGateOutput_name.data());
			std::string link_oGateBias_name(link_oGateBias_name_char);
      delete[] link_oGateBias_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  oGateBias_weight_init;
			oGateBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
			std::shared_ptr<SolverOp<TensorT>>  oGateBias_solver = solver;
			Weight<TensorT> weight_oGateBias(weight_oGateBias_name, oGateBias_weight_init, oGateBias_solver);
			weight_oGateBias.setModuleName(module_name);
			Link link_oGateBias(link_oGateBias_name, oGateBias_name, blockGateOutput_name, weight_oGateBias_name);
			link_oGateBias.setModuleName(module_name);

			model.addWeights({ weight_oGateBias });
			model.addLinks({ link_oGateBias });
		}

		for (const std::string& node_name : source_node_names) {
			// Make the link from input node to input gate
			char* weight_iToIGate_name_char = new char[512];
			sprintf(weight_iToIGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string weight_iToIGate_name(weight_iToIGate_name_char);
      delete[] weight_iToIGate_name_char;

			char* link_iToIGate_name_char = new char[512];
			sprintf(link_iToIGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string link_iToIGate_name(link_iToIGate_name_char);
      delete[] link_iToIGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  iToIGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  iToIGate_solver = solver;
			Weight<TensorT> weight_iToIGate(weight_iToIGate_name, iToIGate_weight_init, iToIGate_solver);
			weight_iToIGate.setModuleName(module_name);
			Link link_iToIGate(link_iToIGate_name, node_name, blockGateInput_name, weight_iToIGate_name);
			link_iToIGate.setModuleName(module_name);

			model.addWeights({ weight_iToIGate });
			model.addLinks({ link_iToIGate });

			// Make the link from input node to output gate
			char* weight_iToOGate_name_char = new char[512];
			sprintf(weight_iToOGate_name_char, "%s_to_%s", node_name.data(), blockGateOutput_name.data());
			std::string weight_iToOGate_name(weight_iToOGate_name_char);
      delete[] weight_iToOGate_name_char;

			char* link_iToOGate_name_char = new char[512];
			sprintf(link_iToOGate_name_char, "%s_to_%s", node_name.data(), blockGateOutput_name.data());
			std::string link_iToOGate_name(link_iToOGate_name_char);
      delete[] link_iToOGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  iToOGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  iToOGate_solver = solver;
			Weight<TensorT> weight_iToOGate(weight_iToOGate_name, iToOGate_weight_init, iToOGate_solver);
			weight_iToOGate.setModuleName(module_name);
			Link link_iToOGate(link_iToOGate_name, node_name, blockGateOutput_name, weight_iToOGate_name);
			link_iToOGate.setModuleName(module_name);

			model.addWeights({ weight_iToOGate });
			model.addLinks({ link_iToOGate });

			if (forget_gate) {
				// Make the link from input node to forget gate
				char* weight_iToFGate_name_char = new char[512];
				sprintf(weight_iToFGate_name_char, "%s_to_%s", node_name.data(), blockGateForget_name.data());
				std::string weight_iToFGate_name(weight_iToFGate_name_char);
        delete[] weight_iToFGate_name_char;

				char* link_iToFGate_name_char = new char[512];
				sprintf(link_iToFGate_name_char, "%s_to_%s", node_name.data(), blockGateForget_name.data());
				std::string link_iToFGate_name(link_iToFGate_name_char);
        delete[] link_iToFGate_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  iToFGate_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  iToFGate_solver = solver;
				Weight<TensorT> weight_iToFGate(weight_iToFGate_name, iToFGate_weight_init, iToFGate_solver);
				weight_iToFGate.setModuleName(module_name);
				Link link_iToFGate(link_iToFGate_name, node_name, blockGateForget_name, weight_iToFGate_name);
				link_iToFGate.setModuleName(module_name);

				model.addWeights({ weight_iToFGate });
				model.addLinks({ link_iToFGate });
			}
		}

		for (int cell_iter = 0; cell_iter < n_cells; ++cell_iter) {

			// Make the input node
			char* blockInput_name_char = new char[512];
			sprintf(blockInput_name_char, "%s-BlockInput-%012d", name.data(), cell_iter);
			std::string blockInput_name(blockInput_name_char);
			Node<TensorT> blockInput(blockInput_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
			blockInput.setModuleName(module_name);
			if (specify_layer) blockInput.setLayerName(module_name + "-BlockInput");
			blockInput.setDropProbability(drop_out_prob);
			model.addNodes({ blockInput });
      delete[] blockInput_name_char;

			// Make the input multiplier node
			char* blockMultInput_name_char = new char[512];
			sprintf(blockMultInput_name_char, "%s-BlockMultInput-%012d", name.data(), cell_iter);
			std::string blockMultInput_name(blockMultInput_name_char);
			Node<TensorT> blockMultInput(blockMultInput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			blockMultInput.setModuleName(module_name);
			if (specify_layer) blockMultInput.setLayerName(module_name + "-BlockMultInput");
			model.addNodes({ blockMultInput });
      delete[] blockMultInput_name_char;

			// Make the output multiplier node
			char* blockOutput_name_char = new char[512];
			sprintf(blockOutput_name_char, "%s-BlockMultOutput-%012d", name.data(), cell_iter);
			std::string blockOutput_name(blockOutput_name_char);
			Node<TensorT> blockOutput(blockOutput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			blockOutput.setModuleName(module_name);
			if (specify_layer) blockOutput.setLayerName(module_name + "-BlockMultOut");
			blockOutput.setDropProbability(drop_out_prob);
			model.addNodes({ blockOutput });
			node_names.push_back(blockOutput_name);
      delete[] blockOutput_name_char;

			// Make the memory cell
			char* blockMemoryCell_name_char = new char[512];
			sprintf(blockMemoryCell_name_char, "%s-BlockMemoryCell-%012d", name.data(), cell_iter);
			std::string blockMemoryCell_name(blockMemoryCell_name_char);
			Node<TensorT> blockMemoryCell(blockMemoryCell_name, NodeType::recursive, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			blockMemoryCell.setModuleName(module_name);
			model.addNodes({ blockMemoryCell });
      delete[] blockMemoryCell_name_char;

			// Make the link from memory cell to output multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMemoryCell_name, blockOutput_name);
			char* link_MemCellToOMult_name_char = new char[512];
			sprintf(link_MemCellToOMult_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockOutput_name.data());
			std::string link_MemCellToOMult_name(link_MemCellToOMult_name_char);
			Link link_MemCellToOMult(link_MemCellToOMult_name, blockMemoryCell_name, blockOutput_name, unity_weight_name);
			link_MemCellToOMult.setModuleName(module_name);
			model.addLinks({ link_MemCellToOMult });
      delete[] link_MemCellToOMult_name_char;

			// Make the link from input multiplier node to memory cell
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMultInput_name, blockMemoryCell_name);
			char* link_iMultToMemCell_name_char = new char[512];
			sprintf(link_iMultToMemCell_name_char, "%s_to_%s", blockMultInput_name.data(), blockMemoryCell_name.data());
			std::string link_iMultToMemCell_name(link_iMultToMemCell_name_char);
			Link link_iMultToMemCell(link_iMultToMemCell_name, blockMultInput_name, blockMemoryCell_name, unity_weight_name);
			link_iMultToMemCell.setModuleName(module_name);
			model.addLinks({ link_iMultToMemCell });
      delete[] link_iMultToMemCell_name_char;

			// Make the link between the input and the input multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockInput_name, blockMultInput_name);
			char* link_iToIMult_name_char = new char[512];
			sprintf(link_iToIMult_name_char, "%s_to_%s", blockInput_name.data(), blockMultInput_name.data());
			std::string link_iToIMult_name(link_iToIMult_name_char);
			Link link_iToIMult(link_iToIMult_name, blockInput_name, blockMultInput_name, unity_weight_name);
			link_iToIMult.setModuleName(module_name);
			model.addLinks({ link_iToIMult });
      delete[] link_iToIMult_name_char;

			// Make the link between the input gate and the input multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockGateInput_name, blockMultInput_name);
			char* link_iGateToIMult_name_char = new char[512];
			sprintf(link_iGateToIMult_name_char, "%s_to_%s", blockGateInput_name.data(), blockMultInput_name.data());
			std::string link_iGateToIMult_name(link_iGateToIMult_name_char);
			Link link_iGateToIMult(link_iGateToIMult_name, blockGateInput_name, blockMultInput_name, unity_weight_name);
			link_iGateToIMult.setModuleName(module_name);
			model.addLinks({ link_iGateToIMult });
      delete[] link_iGateToIMult_name_char;

			// Make the link between the output gate and the output gate multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockGateOutput_name, blockOutput_name);
			char* link_oGateToOMult_name_char = new char[512];
			sprintf(link_oGateToOMult_name_char, "%s_to_%s", blockGateOutput_name.data(), blockOutput_name.data());
			std::string link_oGateToOMult_name(link_oGateToOMult_name_char);
			Link link_oGateToOMult(link_oGateToOMult_name, blockGateOutput_name, blockOutput_name, unity_weight_name);
			link_oGateToOMult.setModuleName(module_name);
			model.addLinks({ link_oGateToOMult });
      delete[] link_oGateToOMult_name_char;

			// Make the link between the output multiplier node and the input
			char* weight_OMultToI_name_char = new char[512];
			sprintf(weight_OMultToI_name_char, "%s_to_%s", blockOutput_name.data(), blockInput_name.data());
			std::string weight_OMultToI_name(weight_OMultToI_name_char);
      delete[] weight_OMultToI_name_char;

			char* link_OMultToI_name_char = new char[512];
			sprintf(link_OMultToI_name_char, "%s_to_%s", blockOutput_name.data(), blockInput_name.data());
			std::string link_OMultToI_name(link_OMultToI_name_char);
      delete[] link_OMultToI_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  OMultToI_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  OMultToI_solver = solver;
			Weight<TensorT> weight_OMultToI(weight_OMultToI_name, OMultToI_weight_init, OMultToI_solver);
			weight_OMultToI.setModuleName(module_name);
			weight_OMultToI.setDropProbability(drop_connection_prob);
			Link link_OMultToI(link_OMultToI_name, blockOutput_name, blockInput_name, weight_OMultToI_name);
			link_OMultToI.setModuleName(module_name);

			model.addWeights({ weight_OMultToI });
			model.addLinks({ link_OMultToI });
      if (specify_cyclic_pairs) model.addCyclicPairs(std::make_pair(blockOutput_name, blockInput_name));

			// Make the link between the output multiplier node and the input gate
			char* weight_OMultToIGate_name_char = new char[512];
			sprintf(weight_OMultToIGate_name_char, "%s_to_%s", blockOutput_name.data(), blockGateInput_name.data());
			std::string weight_OMultToIGate_name(weight_OMultToIGate_name_char);
      delete[] weight_OMultToIGate_name_char;

			char* link_OMultToIGate_name_char = new char[512];
			sprintf(link_OMultToIGate_name_char, "%s_to_%s", blockOutput_name.data(), blockGateInput_name.data());
			std::string link_OMultToIGate_name(link_OMultToIGate_name_char);
      delete[] link_OMultToIGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  OMultToIGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  OMultToIGate_solver = solver;
			Weight<TensorT> weight_OMultToIGate(weight_OMultToIGate_name, OMultToIGate_weight_init, OMultToIGate_solver);
			weight_OMultToIGate.setModuleName(module_name);
			Link link_OMultToIGate(link_OMultToIGate_name, blockOutput_name, blockGateInput_name, weight_OMultToIGate_name);
			link_OMultToIGate.setModuleName(module_name);

			model.addWeights({ weight_OMultToIGate });
			model.addLinks({ link_OMultToIGate });
      if (specify_cyclic_pairs) model.addCyclicPairs(std::make_pair(blockOutput_name, blockGateInput_name));

      // Make the forget gate multiplier node
      char* blockMultForget_name_char = new char[512];
      sprintf(blockMultForget_name_char, "%s-BlockMultForget-%012d", name.data(), cell_iter);
      std::string blockMultForget_name(blockMultForget_name_char);
      Node<TensorT> blockMultForget(blockMultForget_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()), std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
      blockMultForget.setModuleName(module_name);
      if (specify_layer) blockMultForget.setLayerName(module_name + "-BlockMultForget");
      model.addNodes({ blockMultForget });
      delete[] blockMultForget_name_char;

			if (forget_gate) {
				// Make the link between the forget gate and the forget gate multiplier node
				unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockGateForget_name, blockMultForget_name);
				char* link_fGateToFMult_name_char = new char[512];
				sprintf(link_fGateToFMult_name_char, "%s_to_%s", blockGateForget_name.data(), blockMultForget_name.data());
				std::string link_fGateToFMult_name(link_fGateToFMult_name_char);
				Link link_fGateToFMult(link_fGateToFMult_name, blockGateForget_name, blockMultForget_name, unity_weight_name);
				link_fGateToFMult.setModuleName(module_name);
				model.addLinks({ link_fGateToFMult });
        delete[] link_fGateToFMult_name_char;

				// Make the link between the output multiplier node and the forget gate
				char* weight_OMultToFGate_name_char = new char[512];
				sprintf(weight_OMultToFGate_name_char, "%s_to_%s", blockOutput_name.data(), blockGateForget_name.data());
				std::string weight_OMultToFGate_name(weight_OMultToFGate_name_char);
        delete[] weight_OMultToFGate_name_char;

				char* link_OMultToFGate_name_char = new char[512];
				sprintf(link_OMultToFGate_name_char, "%s_to_%s", blockOutput_name.data(), blockGateForget_name.data());
				std::string link_OMultToFGate_name(link_OMultToFGate_name_char);
        delete[] link_OMultToFGate_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  OMultToFGate_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  OMultToFGate_solver = solver;
				Weight<TensorT> weight_OMultToFGate(weight_OMultToFGate_name, OMultToFGate_weight_init, OMultToFGate_solver);
				weight_OMultToFGate.setModuleName(module_name);
				Link link_OMultToFGate(link_OMultToFGate_name, blockOutput_name, blockGateForget_name, weight_OMultToFGate_name);
				link_OMultToFGate.setModuleName(module_name);

				model.addWeights({ weight_OMultToFGate });
				model.addLinks({ link_OMultToFGate });
        if (specify_cyclic_pairs) model.addCyclicPairs(std::make_pair(blockOutput_name, blockGateForget_name));
			}

      // Make the link from forget gate multiplier node to memory cell
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMultForget_name, blockMemoryCell_name);
      char* link_fMultToMemCell_name_char = new char[512];
      sprintf(link_fMultToMemCell_name_char, "%s_to_%s", blockMultForget_name.data(), blockMemoryCell_name.data());
      std::string link_fMultToMemCell_name(link_fMultToMemCell_name_char);
      Link link_fMultToMemCell(link_fMultToMemCell_name, blockMultForget_name, blockMemoryCell_name, unity_weight_name);
      link_fMultToMemCell.setModuleName(module_name);
      model.addLinks({ link_fMultToMemCell });
      if (specify_cyclic_pairs) model.addCyclicPairs(std::make_pair(blockMultForget_name, blockMemoryCell_name));
      delete[] link_fMultToMemCell_name_char;

      // Make the link from memory cell to forget gate multiplier node
      unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMemoryCell_name, blockMultForget_name);
      char* link_MemCellToFMult_name_char = new char[512];
      sprintf(link_MemCellToFMult_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockMultForget_name.data());
      std::string link_MemCellToFMult_name(link_MemCellToFMult_name_char);
      Link link_MemCellToFMult(link_MemCellToFMult_name, blockMemoryCell_name, blockMultForget_name, unity_weight_name);
      link_MemCellToFMult.setModuleName(module_name);
      model.addLinks({ link_MemCellToFMult });
      delete[] link_MemCellToFMult_name_char;

			// Make the link between the output multiplier node and the output gate
			char* weight_OMultToOGate_name_char = new char[512];
			sprintf(weight_OMultToOGate_name_char, "%s_to_%s", blockOutput_name.data(), blockGateOutput_name.data());
			std::string weight_OMultToOGate_name(weight_OMultToOGate_name_char);
      delete[] weight_OMultToOGate_name_char;

			char* link_OMultToOGate_name_char = new char[512];
			sprintf(link_OMultToOGate_name_char, "%s_to_%s", blockOutput_name.data(), blockGateOutput_name.data());
			std::string link_OMultToOGate_name(link_OMultToOGate_name_char);
      delete[] link_OMultToOGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  OMultToOGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  OMultToOGate_solver = solver;
			Weight<TensorT> weight_OMultToOGate(weight_OMultToOGate_name, OMultToOGate_weight_init, OMultToOGate_solver);
			weight_OMultToOGate.setModuleName(module_name);
			Link link_OMultToOGate(link_OMultToOGate_name, blockOutput_name, blockGateOutput_name, weight_OMultToOGate_name);
			link_OMultToOGate.setModuleName(module_name);

			model.addWeights({ weight_OMultToOGate });
			model.addLinks({ link_OMultToOGate });
      if (specify_cyclic_pairs) model.addCyclicPairs(std::make_pair(blockOutput_name, blockGateOutput_name));

			if (biases) {  // biases, links, and weights for input
				// Make the input bias nodes
				char* iBias_name_char = new char[512];
				sprintf(iBias_name_char, "%s-bias-%012d", blockInput_name.data(), cell_iter);
				std::string iBias_name(iBias_name_char);
				Node<TensorT> iBias(iBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
				iBias.setDropProbability(drop_out_prob);
				iBias.setModuleName(module_name);
				model.addNodes({ iBias });
        delete[] iBias_name_char;

				// Make the link between input bias node to input node
				char* weight_iBias_name_char = new char[512];
				sprintf(weight_iBias_name_char, "%s_to_%s", iBias_name.data(), blockInput_name.data());
				std::string weight_iBias_name(weight_iBias_name_char);
        delete[] weight_iBias_name_char;

				char* link_iBias_name_char = new char[512];
				sprintf(link_iBias_name_char, "%s_to_%s", iBias_name.data(), blockInput_name.data());
				std::string link_iBias_name(link_iBias_name_char);
        delete[] link_iBias_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  iBias_weight_init;
				iBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
				std::shared_ptr<SolverOp<TensorT>>  iBias_solver = solver;
				Weight<TensorT> weight_iBias(weight_iBias_name, iBias_weight_init, iBias_solver);
				weight_iBias.setModuleName(module_name);
				weight_iBias.setDropProbability(drop_connection_prob);
				Link link_iBias(link_iBias_name, iBias_name, blockInput_name, weight_iBias_name);
				link_iBias.setModuleName(module_name);

				model.addWeights({ weight_iBias });
				model.addLinks({ link_iBias });
			}

			for (const std::string& node_name : source_node_names) {
				// Make the link form input to block input
				char* weight_iToIBlock_name_char = new char[512];
				sprintf(weight_iToIBlock_name_char, "%s_to_%s", node_name.data(), blockInput_name.data());
				std::string weight_iToIBlock_name(weight_iToIBlock_name_char);
        delete[] weight_iToIBlock_name_char;

				char* link_iToIBlock_name_char = new char[512];
				sprintf(link_iToIBlock_name_char, "%s_to_%s", node_name.data(), blockInput_name.data());
				std::string link_iToIBlock_name(link_iToIBlock_name_char);
        delete[] link_iToIBlock_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  iToIBlock_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  iToIBlock_solver = solver;
				Weight<TensorT> weight_iToIBlock(weight_iToIBlock_name, iToIBlock_weight_init, iToIBlock_solver);
				weight_iToIBlock.setModuleName(module_name);
				weight_iToIBlock.setDropProbability(drop_connection_prob);
				Link link_iToIBlock(link_iToIBlock_name, node_name, blockInput_name, weight_iToIBlock_name);
				link_iToIBlock.setModuleName(module_name);

				model.addWeights({ weight_iToIBlock });
				model.addLinks({ link_iToIBlock });
			}
		}

		return node_names;
	}
	template<typename TensorT>
	std::vector<std::string> ModelBuilder<TensorT>::addLSTMBlock2(
		Model<TensorT> & model, const std::string & name, const std::string& module_name,
		const std::vector<std::string>& source_node_names,
		const int & n_cells,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<IntegrationOp<TensorT>>& node_integration, const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error, const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
		const std::shared_ptr<WeightInitOp<TensorT>> & weight_init, const std::shared_ptr<SolverOp<TensorT>> & solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, bool forget_gate, const bool& specify_layer, bool specify_cyclic_pairs)
	{
		std::vector<std::string> node_names;
		std::string unity_weight_name;

		// Make the input gate node
		char* blockGateInput_name_char = new char[512];
		sprintf(blockGateInput_name_char, "%s-BlockGateInput", name.data());
		std::string blockGateInput_name(blockGateInput_name_char);
		Node<TensorT> blockGateInput(blockGateInput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()), node_integration, node_integration_error, node_integration_weight_grad);
		blockGateInput.setModuleName(module_name);
    if (specify_layer) blockGateInput.setLayerName(module_name + "-BlockGateInput");
		model.addNodes({ blockGateInput });
    delete[] blockGateInput_name_char;

		// Make the output gate node 
		char* blockGateOutput_name_char = new char[512];
		sprintf(blockGateOutput_name_char, "%s-BlockGateOutput", name.data());
		std::string blockGateOutput_name(blockGateOutput_name_char);
		Node<TensorT> blockGateOutput(blockGateOutput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()), node_integration, node_integration_error, node_integration_weight_grad);
		blockGateOutput.setModuleName(module_name);
    if (specify_layer) blockGateOutput.setLayerName(module_name + "-BlockGateOutput");
		model.addNodes({ blockGateOutput });
    delete[] blockGateOutput_name_char;

		std::string blockGateForget_name;
		if (forget_gate) {
			// Make the forget gate node
			char* blockGateForget_name_char = new char[512];
			sprintf(blockGateForget_name_char, "%s-BlockGateForget", name.data());
			blockGateForget_name = std::string(blockGateForget_name_char);
			Node<TensorT> blockGateForget(blockGateForget_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<SigmoidOp<TensorT>>(SigmoidOp<TensorT>()), std::make_shared<SigmoidGradOp<TensorT>>(SigmoidGradOp<TensorT>()), node_integration, node_integration_error, node_integration_weight_grad);
			blockGateForget.setModuleName(module_name);
      if (specify_layer) blockGateForget.setLayerName(module_name + "-BlockGateForget");
			model.addNodes({ blockGateForget });
      delete[] blockGateForget_name_char;
		}

		if (biases) {  // biases, links, and weights for input gate, forget gate, and output gate
			// Make the input gate bias nodes
			char* iGateBias_name_char = new char[512];
			sprintf(iGateBias_name_char, "%s-bias", blockGateInput_name.data());
			std::string iGateBias_name(iGateBias_name_char);
			Node<TensorT> iGateBias(iGateBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			iGateBias.setModuleName(module_name);
			model.addNodes({ iGateBias });
      delete[] iGateBias_name_char;

			// Make the link between input gate bias node to input gate node
			char* weight_iGateBias_name_char = new char[512];
			sprintf(weight_iGateBias_name_char, "%s_to_%s", iGateBias_name.data(), blockGateInput_name.data());
			std::string weight_iGateBias_name(weight_iGateBias_name_char);
      delete[] weight_iGateBias_name_char;

			char* link_iGateBias_name_char = new char[512];
			sprintf(link_iGateBias_name_char, "%s_to_%s", iGateBias_name.data(), blockGateInput_name.data());
			std::string link_iGateBias_name(link_iGateBias_name_char);
      delete[] link_iGateBias_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  iGateBias_weight_init;
			iGateBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
			std::shared_ptr<SolverOp<TensorT>>  iGateBias_solver = solver;
			Weight<TensorT> weight_iGateBias(weight_iGateBias_name, iGateBias_weight_init, iGateBias_solver);
			weight_iGateBias.setModuleName(module_name);
			Link link_iGateBias(link_iGateBias_name, iGateBias_name, blockGateInput_name, weight_iGateBias_name);
			link_iGateBias.setModuleName(module_name);

			model.addWeights({ weight_iGateBias });
			model.addLinks({ link_iGateBias });

			if (forget_gate) {
				// Make the forget gate bias nodes
				char* fGateBias_name_char = new char[512];
				sprintf(fGateBias_name_char, "%s-bias", blockGateForget_name.data());
				std::string fGateBias_name(fGateBias_name_char);
				Node<TensorT> fGateBias(fGateBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
				fGateBias.setModuleName(module_name);
				model.addNodes({ fGateBias });
        delete[] fGateBias_name_char;

				// Make the link between forget gate bias node to forget gate node
				char* weight_fGateBias_name_char = new char[512];
				sprintf(weight_fGateBias_name_char, "%s_to_%s", fGateBias_name.data(), blockGateForget_name.data());
				std::string weight_fGateBias_name(weight_fGateBias_name_char);
        delete[] weight_fGateBias_name_char;

				char* link_fGateBias_name_char = new char[512];
				sprintf(link_fGateBias_name_char, "%s_to_%s", fGateBias_name.data(), blockGateForget_name.data());
				std::string link_fGateBias_name(link_fGateBias_name_char);
        delete[] link_fGateBias_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  fGateBias_weight_init;
				fGateBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
				std::shared_ptr<SolverOp<TensorT>>  fGateBias_solver = solver;
				Weight<TensorT> weight_fGateBias(weight_fGateBias_name, fGateBias_weight_init, fGateBias_solver);
				weight_fGateBias.setModuleName(module_name);
				Link link_fGateBias(link_fGateBias_name, fGateBias_name, blockGateForget_name, weight_fGateBias_name);
				link_fGateBias.setModuleName(module_name);

				model.addWeights({ weight_fGateBias });
				model.addLinks({ link_fGateBias });
			}

			// Make the output gate bias nodes
			char* oGateBias_name_char = new char[512];
			sprintf(oGateBias_name_char, "%s-bias", blockGateOutput_name.data());
			std::string oGateBias_name(oGateBias_name_char);
			Node<TensorT> oGateBias(oGateBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			oGateBias.setModuleName(module_name);
			model.addNodes({ oGateBias });
      delete[] oGateBias_name_char;

			// Make the link between output gate bias node to output gate node
			char* weight_oGateBias_name_char = new char[512];
			sprintf(weight_oGateBias_name_char, "%s_to_%s", oGateBias_name.data(), blockGateOutput_name.data());
			std::string weight_oGateBias_name(weight_oGateBias_name_char);
      delete[] weight_oGateBias_name_char;

			char* link_oGateBias_name_char = new char[512];
			sprintf(link_oGateBias_name_char, "%s_to_%s", oGateBias_name.data(), blockGateOutput_name.data());
			std::string link_oGateBias_name(link_oGateBias_name_char);
      delete[] link_oGateBias_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  oGateBias_weight_init;
			oGateBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
			std::shared_ptr<SolverOp<TensorT>>  oGateBias_solver = solver;
			Weight<TensorT> weight_oGateBias(weight_oGateBias_name, oGateBias_weight_init, oGateBias_solver);
			weight_oGateBias.setModuleName(module_name);
			Link link_oGateBias(link_oGateBias_name, oGateBias_name, blockGateOutput_name, weight_oGateBias_name);
			link_oGateBias.setModuleName(module_name);

			model.addWeights({ weight_oGateBias });
			model.addLinks({ link_oGateBias });
		}
		for (const std::string& node_name : source_node_names) {
			// Make the link from input node to input gate
			char* weight_iToIGate_name_char = new char[512];
			sprintf(weight_iToIGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string weight_iToIGate_name(weight_iToIGate_name_char);
      delete[] weight_iToIGate_name_char;

			char* link_iToIGate_name_char = new char[512];
			sprintf(link_iToIGate_name_char, "%s_to_%s", node_name.data(), blockGateInput_name.data());
			std::string link_iToIGate_name(link_iToIGate_name_char);
      delete[] link_iToIGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  iToIGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  iToIGate_solver = solver;
			Weight<TensorT> weight_iToIGate(weight_iToIGate_name, iToIGate_weight_init, iToIGate_solver);
			weight_iToIGate.setModuleName(module_name);
			Link link_iToIGate(link_iToIGate_name, node_name, blockGateInput_name, weight_iToIGate_name);
			link_iToIGate.setModuleName(module_name);

			model.addWeights({ weight_iToIGate });
			model.addLinks({ link_iToIGate });

			// Make the link from input node to output gate
			char* weight_iToOGate_name_char = new char[512];
			sprintf(weight_iToOGate_name_char, "%s_to_%s", node_name.data(), blockGateOutput_name.data());
			std::string weight_iToOGate_name(weight_iToOGate_name_char);
      delete[] weight_iToOGate_name_char;

			char* link_iToOGate_name_char = new char[512];
			sprintf(link_iToOGate_name_char, "%s_to_%s", node_name.data(), blockGateOutput_name.data());
			std::string link_iToOGate_name(link_iToOGate_name_char);
      delete[] link_iToOGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  iToOGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  iToOGate_solver = solver;
			Weight<TensorT> weight_iToOGate(weight_iToOGate_name, iToOGate_weight_init, iToOGate_solver);
			weight_iToOGate.setModuleName(module_name);
			Link link_iToOGate(link_iToOGate_name, node_name, blockGateOutput_name, weight_iToOGate_name);
			link_iToOGate.setModuleName(module_name);

			model.addWeights({ weight_iToOGate });
			model.addLinks({ link_iToOGate });

			if (forget_gate) {
				// Make the link from input node to forget gate
				char* weight_iToFGate_name_char = new char[512];
				sprintf(weight_iToFGate_name_char, "%s_to_%s", node_name.data(), blockGateForget_name.data());
				std::string weight_iToFGate_name(weight_iToFGate_name_char);
        delete[] weight_iToFGate_name_char;

				char* link_iToFGate_name_char = new char[512];
				sprintf(link_iToFGate_name_char, "%s_to_%s", node_name.data(), blockGateForget_name.data());
				std::string link_iToFGate_name(link_iToFGate_name_char);
        delete[] link_iToFGate_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  iToFGate_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  iToFGate_solver = solver;
				Weight<TensorT> weight_iToFGate(weight_iToFGate_name, iToFGate_weight_init, iToFGate_solver);
				weight_iToFGate.setModuleName(module_name);
				Link link_iToFGate(link_iToFGate_name, node_name, blockGateForget_name, weight_iToFGate_name);
				link_iToFGate.setModuleName(module_name);

				model.addWeights({ weight_iToFGate });
				model.addLinks({ link_iToFGate });
			}
		}
		for (int cell_iter = 0; cell_iter < n_cells; ++cell_iter) {
			// Make the input node
			char* blockInput_name_char = new char[512];
			sprintf(blockInput_name_char, "%s-BlockInput-%012d", name.data(), cell_iter);
			std::string blockInput_name(blockInput_name_char);
			Node<TensorT> blockInput(blockInput_name, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
			blockInput.setModuleName(module_name);
			blockInput.setDropProbability(drop_out_prob);
      if (specify_layer) blockInput.setLayerName(module_name + "-BlockInput");
			model.addNodes({ blockInput });
      delete[] blockInput_name_char;

			// Make the input multiplier node
			char* blockMultInput_name_char = new char[512];
			sprintf(blockMultInput_name_char, "%s-BlockMultInput-%012d", name.data(), cell_iter);
			std::string blockMultInput_name(blockMultInput_name_char);
			Node<TensorT> blockMultInput(blockMultInput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			blockMultInput.setModuleName(module_name);
      if (specify_layer) blockMultInput.setLayerName(module_name + "-BlockMultInput");
			model.addNodes({ blockMultInput });
      delete[] blockMultInput_name_char;

			// Make the output multiplier node[add drop prob]
			char* blockOutput_name_char = new char[512];
			sprintf(blockOutput_name_char, "%s-BlockMultOutput-%012d", name.data(), cell_iter);
			std::string blockOutput_name(blockOutput_name_char);
			Node<TensorT> blockOutput(blockOutput_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
			blockOutput.setModuleName(module_name);
			blockOutput.setDropProbability(drop_out_prob);
      if (specify_layer) blockOutput.setLayerName(module_name + "-BlockMultOutput");
			model.addNodes({ blockOutput });
			node_names.push_back(blockOutput_name);
      delete[] blockOutput_name_char;

			// Make the memory cell
			char* blockMemoryCell_name_char = new char[512];
			sprintf(blockMemoryCell_name_char, "%s-BlockMemoryCell-%012d", name.data(), cell_iter);
			std::string blockMemoryCell_name(blockMemoryCell_name_char);
			Node<TensorT> blockMemoryCell(blockMemoryCell_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
			blockMemoryCell.setModuleName(module_name);
      if (specify_layer) blockMemoryCell.setLayerName(module_name + "-BlockMemoryCell");
			model.addNodes({ blockMemoryCell });
      delete[] blockMemoryCell_name_char;

			// Make the link from memory cell to output multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMemoryCell_name, blockOutput_name);
			char* link_MemCellToOMult_name_char = new char[512];
			sprintf(link_MemCellToOMult_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockOutput_name.data());
			std::string link_MemCellToOMult_name(link_MemCellToOMult_name_char);
			Link link_MemCellToOMult(link_MemCellToOMult_name, blockMemoryCell_name, blockOutput_name, unity_weight_name);
			link_MemCellToOMult.setModuleName(module_name);
			model.addLinks({ link_MemCellToOMult });
      delete[] link_MemCellToOMult_name_char;

			// Make the link from input multiplier node to memory cell
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMultInput_name, blockMemoryCell_name);
			char* link_iMultToMemCell_name_char = new char[512];
			sprintf(link_iMultToMemCell_name_char, "%s_to_%s", blockMultInput_name.data(), blockMemoryCell_name.data());
			std::string link_iMultToMemCell_name(link_iMultToMemCell_name_char);
			Link link_iMultToMemCell(link_iMultToMemCell_name, blockMultInput_name, blockMemoryCell_name, unity_weight_name);
			link_iMultToMemCell.setModuleName(module_name);
			model.addLinks({ link_iMultToMemCell });
      delete[] link_iMultToMemCell_name_char;

			// Make the link between the input and the input multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockInput_name, blockMultInput_name);
			char* link_iToIMult_name_char = new char[512];
			sprintf(link_iToIMult_name_char, "%s_to_%s", blockInput_name.data(), blockMultInput_name.data());
			std::string link_iToIMult_name(link_iToIMult_name_char);
			Link link_iToIMult(link_iToIMult_name, blockInput_name, blockMultInput_name, unity_weight_name);
			link_iToIMult.setModuleName(module_name);
			model.addLinks({ link_iToIMult });
      delete[] link_iToIMult_name_char;

			// Make the link between the input gate and the input multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockGateInput_name, blockMultInput_name);
			char* link_iGateToIMult_name_char = new char[512];
			sprintf(link_iGateToIMult_name_char, "%s_to_%s", blockGateInput_name.data(), blockMultInput_name.data());
			std::string link_iGateToIMult_name(link_iGateToIMult_name_char);
			Link link_iGateToIMult(link_iGateToIMult_name, blockGateInput_name, blockMultInput_name, unity_weight_name);
			link_iGateToIMult.setModuleName(module_name);
			model.addLinks({ link_iGateToIMult });
      delete[] link_iGateToIMult_name_char;

			// Make the link between the output gate and the output gate multiplier node
			unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockGateOutput_name, blockOutput_name);
			char* link_oGateToOMult_name_char = new char[512];
			sprintf(link_oGateToOMult_name_char, "%s_to_%s", blockGateOutput_name.data(), blockOutput_name.data());
			std::string link_oGateToOMult_name(link_oGateToOMult_name_char);
			Link link_oGateToOMult(link_oGateToOMult_name, blockGateOutput_name, blockOutput_name, unity_weight_name);
			link_oGateToOMult.setModuleName(module_name);
			model.addLinks({ link_oGateToOMult });
      delete[] link_oGateToOMult_name_char;

			// Make the link between the memory cell node and the input gate
			char* weight_OMultToIGate_name_char = new char[512];
			sprintf(weight_OMultToIGate_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockGateInput_name.data());
			std::string weight_OMultToIGate_name(weight_OMultToIGate_name_char);
      delete[] weight_OMultToIGate_name_char;

			char* link_OMultToIGate_name_char = new char[512];
			sprintf(link_OMultToIGate_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockGateInput_name.data());
			std::string link_OMultToIGate_name(link_OMultToIGate_name_char);
      delete[] link_OMultToIGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  OMultToIGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  OMultToIGate_solver = solver;
			Weight<TensorT> weight_OMultToIGate(weight_OMultToIGate_name, OMultToIGate_weight_init, OMultToIGate_solver);
			weight_OMultToIGate.setModuleName(module_name);
			Link link_OMultToIGate(link_OMultToIGate_name, blockMemoryCell_name, blockGateInput_name, weight_OMultToIGate_name);
			link_OMultToIGate.setModuleName(module_name);

			model.addWeights({ weight_OMultToIGate });
			model.addLinks({ link_OMultToIGate });

			if (forget_gate) {
				// Make the forget gate multiplier node
				char* blockMultForget_name_char = new char[512];
				sprintf(blockMultForget_name_char, "%s-BlockMultForget-%012d", name.data(), cell_iter);
				std::string blockMultForget_name(blockMultForget_name_char);
				Node<TensorT> blockMultForget(blockMultForget_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()));
				blockMultForget.setModuleName(module_name);
        if (specify_layer) blockMultForget.setLayerName(module_name + "-BlockMultForget");
				model.addNodes({ blockMultForget });
        delete[] blockMultForget_name_char;

				// Make the link between the forget gate and the forget gate multiplier node
				unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockGateForget_name, blockMultForget_name);
				char* link_fGateToFMult_name_char = new char[512];
				sprintf(link_fGateToFMult_name_char, "%s_to_%s", blockGateForget_name.data(), blockMultForget_name.data());
				std::string link_fGateToFMult_name(link_fGateToFMult_name_char);
				Link link_fGateToFMult(link_fGateToFMult_name, blockGateForget_name, blockMultForget_name, unity_weight_name);
				link_fGateToFMult.setModuleName(module_name);
				model.addLinks({ link_fGateToFMult });
        delete[] link_fGateToFMult_name_char;

				// Make the link between the memory cell node and the forget gate
				char* weight_OMultToFGate_name_char = new char[512];
				sprintf(weight_OMultToFGate_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockGateForget_name.data());
				std::string weight_OMultToFGate_name(weight_OMultToFGate_name_char);
        delete[] weight_OMultToFGate_name_char;

				char* link_OMultToFGate_name_char = new char[512];
				sprintf(link_OMultToFGate_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockGateForget_name.data());
				std::string link_OMultToFGate_name(link_OMultToFGate_name_char);
        delete[] link_OMultToFGate_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  OMultToFGate_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  OMultToFGate_solver = solver;
				Weight<TensorT> weight_OMultToFGate(weight_OMultToFGate_name, OMultToFGate_weight_init, OMultToFGate_solver);
				weight_OMultToFGate.setModuleName(module_name);
				Link link_OMultToFGate(link_OMultToFGate_name, blockMemoryCell_name, blockGateForget_name, weight_OMultToFGate_name);
				link_OMultToFGate.setModuleName(module_name);

				model.addWeights({ weight_OMultToFGate });
				model.addLinks({ link_OMultToFGate });

				// Make the link from forget gate multiplier node to memory cell
				unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMultForget_name, blockMemoryCell_name);
				char* link_fMultToMemCell_name_char = new char[512];
				sprintf(link_fMultToMemCell_name_char, "%s_to_%s", blockMultForget_name.data(), blockMemoryCell_name.data());
				std::string link_fMultToMemCell_name(link_fMultToMemCell_name_char);
				Link link_fMultToMemCell(link_fMultToMemCell_name, blockMultForget_name, blockMemoryCell_name, unity_weight_name);
				link_fMultToMemCell.setModuleName(module_name);
				model.addLinks({ link_fMultToMemCell });
        delete[] link_fMultToMemCell_name_char;

				// Make the link from memory cell to forget gate multiplier node
				unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMemoryCell_name, blockMultForget_name);
				char* link_MemCellToFMult_name_char = new char[512];
				sprintf(link_MemCellToFMult_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockMultForget_name.data());
				std::string link_MemCellToFMult_name(link_MemCellToFMult_name_char);
				Link link_MemCellToFMult(link_MemCellToFMult_name, blockMemoryCell_name, blockMultForget_name, unity_weight_name);
				link_MemCellToFMult.setModuleName(module_name);
				model.addLinks({ link_MemCellToFMult });
        delete[] link_MemCellToFMult_name_char;
			}
			else {
				// Make the link from forget gate multiplier node to memory cell
				unity_weight_name = makeUnityWeight(model, 1.0, module_name, "%s_to_%s", blockMemoryCell_name, blockMemoryCell_name);
				char* link_fMultToMemCell_name_char = new char[512];
				sprintf(link_fMultToMemCell_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockMemoryCell_name.data());
				std::string link_fMultToMemCell_name(link_fMultToMemCell_name_char);
				Link link_fMultToMemCell(link_fMultToMemCell_name, blockMemoryCell_name, blockMemoryCell_name, unity_weight_name);
				link_fMultToMemCell.setModuleName(module_name);
				model.addLinks({ link_fMultToMemCell });
        delete[] link_fMultToMemCell_name_char;
			}

			// Make the link between the output multiplier node and the output gate
			char* weight_OMultToOGate_name_char = new char[512];
			sprintf(weight_OMultToOGate_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockGateOutput_name.data());
			std::string weight_OMultToOGate_name(weight_OMultToOGate_name_char);
      delete[] weight_OMultToOGate_name_char;

			char* link_OMultToOGate_name_char = new char[512];
			sprintf(link_OMultToOGate_name_char, "%s_to_%s", blockMemoryCell_name.data(), blockGateOutput_name.data());
			std::string link_OMultToOGate_name(link_OMultToOGate_name_char);
      delete[] link_OMultToOGate_name_char;

			std::shared_ptr<WeightInitOp<TensorT>>  OMultToOGate_weight_init = weight_init;
			std::shared_ptr<SolverOp<TensorT>>  OMultToOGate_solver = solver;
			Weight<TensorT> weight_OMultToOGate(weight_OMultToOGate_name, OMultToOGate_weight_init, OMultToOGate_solver);
			weight_OMultToOGate.setModuleName(module_name);
			Link link_OMultToOGate(link_OMultToOGate_name, blockMemoryCell_name, blockGateOutput_name, weight_OMultToOGate_name);
			link_OMultToOGate.setModuleName(module_name);

			model.addWeights({ weight_OMultToOGate });
			model.addLinks({ link_OMultToOGate });

			if (biases) {  // biases, links, and weights for input
				// Make the input bias nodes
				char* iBias_name_char = new char[512];
				sprintf(iBias_name_char, "%s-bias-%012d", blockInput_name.data(), cell_iter);
				std::string iBias_name(iBias_name_char);
				Node<TensorT> iBias(iBias_name, NodeType::bias, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
				iBias.setDropProbability(drop_out_prob);
				iBias.setModuleName(module_name);
				model.addNodes({ iBias });
        delete[] iBias_name_char;

				// Make the link between input bias node to input node
				char* weight_iBias_name_char = new char[512];
				sprintf(weight_iBias_name_char, "%s_to_%s", iBias_name.data(), blockInput_name.data());
				std::string weight_iBias_name(weight_iBias_name_char);
        delete[] weight_iBias_name_char;

				char* link_iBias_name_char = new char[512];
				sprintf(link_iBias_name_char, "%s_to_%s", iBias_name.data(), blockInput_name.data());
				std::string link_iBias_name(link_iBias_name_char);
        delete[] link_iBias_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  iBias_weight_init;
				iBias_weight_init = std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(0));
				std::shared_ptr<SolverOp<TensorT>>  iBias_solver = solver;
				Weight<TensorT> weight_iBias(weight_iBias_name, iBias_weight_init, iBias_solver);
				weight_iBias.setModuleName(module_name);
				weight_iBias.setDropProbability(drop_connection_prob);
				Link link_iBias(link_iBias_name, iBias_name, blockInput_name, weight_iBias_name);
				link_iBias.setModuleName(module_name);

				model.addWeights({ weight_iBias });
				model.addLinks({ link_iBias });
			}

			for (const std::string& node_name : source_node_names) {
				// Make the link form input to block input
				char* weight_iToIBlock_name_char = new char[512];
				sprintf(weight_iToIBlock_name_char, "%s_to_%s", node_name.data(), blockInput_name.data());
				std::string weight_iToIBlock_name(weight_iToIBlock_name_char);
        delete[] weight_iToIBlock_name_char;

				char* link_iToIBlock_name_char = new char[512];
				sprintf(link_iToIBlock_name_char, "%s_to_%s", node_name.data(), blockInput_name.data());
				std::string link_iToIBlock_name(link_iToIBlock_name_char);
        delete[] link_iToIBlock_name_char;

				std::shared_ptr<WeightInitOp<TensorT>>  iToIBlock_weight_init = weight_init;
				std::shared_ptr<SolverOp<TensorT>>  iToIBlock_solver = solver;
				Weight<TensorT> weight_iToIBlock(weight_iToIBlock_name, iToIBlock_weight_init, iToIBlock_solver);
				weight_iToIBlock.setModuleName(module_name);
				weight_iToIBlock.setDropProbability(drop_connection_prob);
				Link link_iToIBlock(link_iToIBlock_name, node_name, blockInput_name, weight_iToIBlock_name);
				link_iToIBlock.setModuleName(module_name);

				model.addWeights({ weight_iToIBlock });
				model.addLinks({ link_iToIBlock });
			}
		}

		return node_names;
	}
	template<typename TensorT>
	inline std::vector<std::string> ModelBuilder<TensorT>::addMultiHeadAttention(Model<TensorT>& model, const std::string & name, const std::string & module_name, 
		const std::vector<std::string>& query_node_names, const std::vector<std::string>& key_node_names, const std::vector<std::string>& values_node_names, 
		const int & n_heads, const std::string & attention_type, const int & model_length, const int & key_length, const int & values_length,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad, 
		const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver, const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, bool split_attention_layers)
	{

		// Create each head and concatenate the results
		std::vector<std::string> node_names_heads;
		for (size_t i = 0; i < n_heads; ++i) {
			std::vector<std::string> node_names_attention;
			char* name_char = new char[512];
			sprintf(name_char, "%s-%012d", name.data(), i);
			std::string node_name(name_char);
      delete[] name_char;
			if (attention_type == "DotProd") {
				node_names_attention = addDotProdAttention(model, node_name, module_name,
					query_node_names, key_node_names, values_node_names, key_length, values_length, node_activation, node_activation_grad,
					weight_init, solver, drop_out_prob, drop_connection_prob, biases, split_attention_layers);
			}
			else {
				std::cout << "Attention type " << attention_type << " was not recognized." << std::endl;
			}
			for (std::string& node_name : node_names_attention)
				node_names_heads.push_back(node_name);
		}

		// Matrix multiply the concatenated heads to create the output
		std::vector<std::string> node_names = addFullyConnected(model, name + "_MultiHead", module_name, node_names_heads, model_length, node_activation, node_activation_grad, 
			std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
			weight_init, solver, drop_out_prob, drop_connection_prob, biases, split_attention_layers);

		return node_names;
	}
	template<typename TensorT>
	inline std::vector<std::string> ModelBuilder<TensorT>::addDotProdAttention(Model<TensorT>& model, const std::string& name, const std::string& module_name,
		const std::vector<std::string>& query_node_names, const std::vector<std::string>& key_node_names, const std::vector<std::string>& values_node_names,
		const int& key_length, const int& values_length,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
		const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver,
		const TensorT& drop_out_prob, const TensorT& drop_connection_prob, const bool& biases, bool split_attention_layers)
	{
		std::vector<std::string> node_names;

		// Make the query network
		std::vector<std::string> node_names_query = addFullyConnected(model, name + "_query", module_name + "_query", query_node_names, key_length,
			std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
			std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
			weight_init, solver, drop_out_prob, drop_connection_prob, false, split_attention_layers);

		// Make the key network
		std::vector<std::string> node_names_key = addFullyConnected(model, name + "_keys", module_name + "_keys", key_node_names, key_length,
			std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
			std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
			weight_init, solver, drop_out_prob, drop_connection_prob, false, split_attention_layers);

		// Make the values network
		std::vector<std::string> node_names_value = addFullyConnected(model, name + "_values", module_name + "_values", values_node_names, values_length,
			std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
			std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
			weight_init, solver, drop_out_prob, drop_connection_prob, false, split_attention_layers);

		// Multiply the key with the values and scale by the squared of the keys_length
		std::vector<std::string> node_names_scores = addSinglyConnected(model, name + "_scores", module_name + "_scores", node_names_key, node_names_key.size(),
			std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()),
			std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
			std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, split_attention_layers);
		addSinglyConnected(model, module_name + "_scores", node_names_query, node_names_scores,
			std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, split_attention_layers);

		// Add the scalar
		TensorT scalar_value = 1/std::sqrt((TensorT)key_length);
		char* scalar_name_char = new char[512];
		sprintf(scalar_name_char, "%s-scalar", name.data());
		std::string scalar_name(scalar_name_char);
		Node<TensorT> scalar(scalar_name, NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		scalar.setModuleName(module_name);
    if (split_attention_layers) scalar.setLayerName(module_name + "-scalar");
		model.addNodes({ scalar });
    delete[] scalar_name_char;

		std::vector<std::string> scalar_nodes = { scalar_name };
		addFullyConnected(model, module_name + "_scalar", scalar_nodes, node_names_scores,
			std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(scalar_value)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, split_attention_layers);

		// Add a stable softmax to create the attention
		std::vector<std::string> node_names_attention = addStableSoftMax(model, name + "_softMax", module_name, node_names_scores, split_attention_layers);

		// Multiply the attention with the values
		node_names = addSinglyConnected(model, name + "_attention", module_name + "_attention", node_names_value, node_names_value.size(),
			node_activation, node_activation_grad,
			std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
			std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), drop_out_prob, 0.0, false, split_attention_layers);
		addSinglyConnected(model, module_name, node_names_attention, node_names,
			std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, split_attention_layers);

		return node_names;
	}
	template<typename TensorT>
	inline std::vector<std::string> ModelBuilder<TensorT>::addScalar(Model<TensorT>& model, const std::string & name, const std::string & module_name, 
		const std::vector<std::string>& source_node_names, const TensorT & scalar_value, 
		const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad, const bool& specify_layer)
	{
		// Multiply the key with the values and scale by the squared of the keys_length
		std::vector<std::string> node_names = addSinglyConnected(model, name, module_name, source_node_names, source_node_names.size(),
			node_activation, node_activation_grad,
			std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()),std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
			std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1.0)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, 0.0, false, specify_layer);

		// Add the scalar
		char* scalar_name_char = new char[512];
		sprintf(scalar_name_char, "%s-scalar", name.data());
		std::string scalar_name(scalar_name_char);
		Node<TensorT> scalar(scalar_name, NodeType::input, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
		scalar.setModuleName(module_name);
    if (specify_layer) scalar.setLayerName(module_name + "-scalar");
		model.addNodes({ scalar });
    delete[] scalar_name_char;

		std::vector<std::string> scalar_nodes = { scalar_name };
		addFullyConnected(model, module_name, scalar_nodes, node_names,
			std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(scalar_value)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0.0, specify_layer);

		return node_names;
	}
  template<typename TensorT>
  inline std::vector<std::string> ModelBuilder<TensorT>::addGaussianPosterior(Model<TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names, const std::vector<std::string>& gaussian_node_names, const bool& specify_layer)
  {
    // Add the gaussian difference
    std::vector<std::string> gaussian_posterior_node_names = addGaussian_(model, name, module_name, mu_node_names, logvar_node_names, gaussian_node_names, specify_layer);
    return gaussian_posterior_node_names;
  }
  template<typename TensorT>
  inline std::vector<std::string> ModelBuilder<TensorT>::addGaussian_(Model<TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names, const std::vector<std::string>& gaussian_node_names, const bool& specify_layer)
  {
    // logvar to sigma **2
    std::vector<std::string> sigma_node_names = addSinglyConnected(model, name + "-GaussianSigma", module_name + "-GaussianSigma", logvar_node_names, logvar_node_names.size(), std::make_shared<ExponentialOp<TensorT>>(ExponentialOp<TensorT>()), std::make_shared<ExponentialGradOp<TensorT>>(ExponentialGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);
    std::vector<std::string> sigma2_node_names = addSinglyConnected(model, name + "-GaussianSigma2", module_name + "-GaussianSigma2", sigma_node_names, sigma_node_names.size(), std::make_shared<PowOp<TensorT>>(PowOp<TensorT>(TensorT(2))), std::make_shared<PowGradOp<TensorT>>(PowGradOp<TensorT>(TensorT(2))), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);

    // sigma**2 to scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
    std::vector<std::string> scaling_node_names = addSinglyConnected(model, name + "-GaussianScale", module_name + "-GaussianScale", sigma2_node_names, sigma2_node_names.size(), std::make_shared<PowOp<TensorT>>(PowOp<TensorT>(TensorT(-0.5))), std::make_shared<PowGradOp<TensorT>>(PowGradOp<TensorT>(TensorT(-0.5))), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(2.0 * 3.14159265359))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);

    // x and mu to (x - mu) ** 2
    std::vector<std::string> xMinMu2_node_names = addSinglyConnected(model, name + "-GaussianXMinMu2", module_name + "-GaussianXMinMu2", gaussian_node_names, gaussian_node_names.size(), std::make_shared<PowOp<TensorT>>(PowOp<TensorT>(TensorT(2))), std::make_shared<PowGradOp<TensorT>>(PowGradOp<TensorT>(TensorT(2))), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);
    addSinglyConnected(model, module_name + "-GaussianXMinMu2", mu_node_names, xMinMu2_node_names,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(-1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), specify_layer);
    
    // sigma ** 2 to 1 / (2.0 * sigma ** 2)
    std::vector<std::string> bellSigma_node_names = addSinglyConnected(model, name + "-GaussianBellSigma", module_name + "-GaussianBellSigma", sigma2_node_names, sigma2_node_names.size(), std::make_shared<InverseOp<TensorT>>(InverseOp<TensorT>()), std::make_shared<InverseGradOp<TensorT>>(InverseGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(2.0))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);
    
    // (x - mu) ** 2 and 1 / (2.0 * sigma ** 2) to bell = exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    std::vector<std::string> bell_node_names = addSinglyConnected(model, name + "-GaussianBell", module_name + "-GaussianBell", xMinMu2_node_names, xMinMu2_node_names.size(), std::make_shared<ExponentialOp<TensorT>>(ExponentialOp<TensorT>()), std::make_shared<ExponentialGradOp<TensorT>>(ExponentialGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()), std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(-1.0))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);
    addSinglyConnected(model, module_name + "-GaussianBell", bellSigma_node_names, bell_node_names,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), specify_layer);
    
    // scaling * bell
    std::vector<std::string> gaussian__node_names = addSinglyConnected(model, name, module_name + "-Gaussian", scaling_node_names, scaling_node_names.size(), std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<ProdOp<TensorT>>(ProdOp<TensorT>()), std::make_shared<ProdErrorOp<TensorT>>(ProdErrorOp<TensorT>()), std::make_shared<ProdWeightGradOp<TensorT>>(ProdWeightGradOp<TensorT>()),
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);
    addSinglyConnected(model, module_name + "-Gaussian", bell_node_names, gaussian__node_names,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), specify_layer);

    return gaussian__node_names;
  }
  template<typename TensorT>
  inline std::vector<std::string> ModelBuilder<TensorT>::addMixedGaussianPior(Model<TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& gaussian_node_names, const TensorT& logvar_1, const TensorT& logvar_2, const TensorT& pi, const bool& specify_layer)
  {
    // Make the mu (i.e., 0) layer and the logvar (i.e., scalar) layers
    std::vector<std::string> mu_node_names, logvar1_node_names, logvar2_node_names;
    for (int i = 0; i < gaussian_node_names.size(); ++i) {
      // Mu
      char* mu_name_char = new char[512];
      sprintf(mu_name_char, "%s-MixedGaussianPriorMu-%012d", name.data(), i);
      std::string mu_name(mu_name_char);
      Node<TensorT> mu(mu_name, NodeType::zero, NodeStatus::activated, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
      mu.setModuleName(module_name + "-MixedGaussianPriorMu");
      if (specify_layer) mu.setLayerName(module_name + "-MixedGaussianPriorMu");
      model.addNodes({ mu });
      mu_node_names.push_back(mu_name);
      delete[] mu_name_char;

      // logvar1
      char* logvar1_name_char = new char[512];
      sprintf(logvar1_name_char, "%s-MixedGaussianPriorLogVar1-%012d", name.data(), i);
      std::string logvar1_name(logvar1_name_char);
      Node<TensorT> logvar1(logvar1_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
      logvar1.setModuleName(module_name + "-MixedGaussianPriorLogVar1");
      if (specify_layer) logvar1.setLayerName(module_name + "-MixedGaussianPriorLogVar1");
      model.addNodes({ logvar1 });
      logvar1_node_names.push_back(logvar1_name);
      delete[] logvar1_name_char;

      // logvar2
      char* logvar2_name_char = new char[512];
      sprintf(logvar2_name_char, "%s-MixedGaussianPriorLogVar2-%012d", name.data(), i);
      std::string logvar2_name(logvar2_name_char);
      Node<TensorT> logvar2(logvar2_name, NodeType::hidden, NodeStatus::initialized, std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()));
      logvar2.setModuleName(module_name + "-MixedGaussianPriorLogVar2");
      if (specify_layer) logvar2.setLayerName(module_name + "-MixedGaussianPriorLogVar2");
      model.addNodes({ logvar2 });
      logvar2_node_names.push_back(logvar2_name);
      delete[] logvar2_name_char;
    }
    addBiases(model, module_name + "-MixedGaussianPriorLogVar1Bias", logvar1_node_names, std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(logvar_1))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), specify_layer);
    addBiases(model, module_name + "-MixedGaussianPriorLogVar2Bias", logvar2_node_names, std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(logvar_2))), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), specify_layer);

    // Make the two Gaussians
    std::vector<std::string> gaussian1_node_names = addGaussian_(model, name + "-Gaussian-1", module_name + "-Gaussian-1", mu_node_names, logvar1_node_names, gaussian_node_names, specify_layer);
    std::vector<std::string> gaussian2_node_names = addGaussian_(model, name + "-Gaussian-2", module_name + "-Gaussian-2", mu_node_names, logvar2_node_names, gaussian_node_names, specify_layer);
    
    // Mix the two Gaussians    
    std::vector<std::string> mixedGaussianPrior_node_names = addSinglyConnected(model, name + "-MixedGaussianPrior", module_name + "-MixedGaussianPrior", gaussian1_node_names, gaussian1_node_names.size(), std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
    std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(pi)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), TensorT(0), false, specify_layer);
    addSinglyConnected(model, module_name + "-MixedGaussianPrior", gaussian2_node_names, mixedGaussianPrior_node_names,
      std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(TensorT(1)-pi)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), TensorT(0), specify_layer);

    return mixedGaussianPrior_node_names;
  }
  template<typename TensorT>
  inline std::vector<std::string> ModelBuilder<TensorT>::addFullyConnectedBayesian(Model<TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& source_node_names, const int& n_nodes, const std::shared_ptr<ActivationOp<TensorT>>& node_activation, const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad, const std::shared_ptr<IntegrationOp<TensorT>>& node_integration, const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error, const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
    const std::shared_ptr<WeightInitOp<TensorT>>& weight_init_mu, const std::shared_ptr<SolverOp<TensorT>>& solver_mu,
    const std::shared_ptr<WeightInitOp<TensorT>>& weight_init_logvar, const std::shared_ptr<SolverOp<TensorT>>& solver_logvar, const TensorT& logvar_1, const TensorT& logvar_2, const TensorT& pi, 
    std::vector<std::string>& node_names_logvar_output, std::vector<std::string>& node_names_posterior_output, std::vector<std::string>& node_names_prior_output, const bool& specify_layer)
  {
    std::vector<std::string> node_names_output;
    node_names_logvar_output.clear();
    node_names_posterior_output.clear();
    node_names_prior_output.clear();

    // Bayesian source layer
    std::vector<std::vector<std::string>> node_names_mu, node_names_logvar, node_names_gaussian;
    for (const auto& node_name_source : source_node_names) {
      // Create the source to mu and logvar nodes, links, and weights
      std::vector<std::string> node_names_mu_source = addFullyConnected(model, name + "-" + node_name_source + "-Mu", module_name + "-source",
        { node_name_source }, n_nodes,
        std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        weight_init_mu, solver_mu, 0.0f, 0.0f, false, specify_layer);
      std::vector<std::string> node_names_logvar_source = addFullyConnected(model, name + "-" + node_name_source + "-LogVar", module_name + "-source",
        { node_name_source }, n_nodes,
        std::make_shared<LinearOp<TensorT>>(LinearOp<TensorT>()), std::make_shared<LinearGradOp<TensorT>>(LinearGradOp<TensorT>()), std::make_shared<SumOp<TensorT>>(SumOp<TensorT>()), std::make_shared<SumErrorOp<TensorT>>(SumErrorOp<TensorT>()), std::make_shared<SumWeightGradOp<TensorT>>(SumWeightGradOp<TensorT>()),
        weight_init_logvar, solver_logvar, 0.0f, 0.0f, false, specify_layer);
      
      // Gaussian node for each source node
      std::vector<std::string> node_names_gaussian_source = addGaussianEncoding(model, name + "-" + node_name_source + "-Gaussian", module_name + "-Gaussian", node_names_mu_source, node_names_logvar_source, specify_layer);
      
      // Gaussian posterior and prior for each source node
      std::vector<std::string> node_names_posterior_source = addGaussianPosterior(model, name + "-" + node_name_source + "-Posterior", module_name + "-Posterior", node_names_mu_source, node_names_logvar_source, node_names_gaussian_source, specify_layer);
      std::vector<std::string> node_names_prior_source = addMixedGaussianPior(model, name + "-" + node_name_source + "-Prior", module_name + "-Prior", node_names_gaussian_source, logvar_1, logvar_2, pi, specify_layer);

      // Update the output node names
      node_names_mu.push_back(node_names_mu_source);
      node_names_logvar.push_back(node_names_logvar_source);
      for (const std::string& node_name : node_names_logvar_source) node_names_logvar_output.push_back(node_name);
      node_names_gaussian.push_back(node_names_gaussian_source);
      for (const std::string& node_name : node_names_posterior_source) node_names_posterior_output.push_back(node_name);
      for (const std::string& node_name : node_names_prior_source) node_names_prior_output.push_back(node_name);
    }

    // Create the output nodes
    for (int i = 0; i < n_nodes; ++i) {
      char* node_name_output_char = new char[512];
      sprintf(node_name_output_char, "%s_%012d", name.data(), i);
      std::string node_name_output(node_name_output_char);
      node_names_output.push_back(node_name_output);
      Node<TensorT> node_output(node_name_output, NodeType::hidden, NodeStatus::initialized, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
      node_output.setModuleName(module_name);
      if (specify_layer) node_output.setLayerName(module_name);
      model.addNodes({ node_output });
      delete[] node_name_output_char;
    }

    // Link the Guassian nodes to the output nodes
    for (int i = 0; i < source_node_names.size(); ++i) {
      addSinglyConnected(model, module_name, node_names_gaussian.at(i), node_names_output, std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(1)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()), 0, specify_layer);
    }

    return node_names_output;
  }
  template<typename TensorT>
	inline std::string ModelBuilder<TensorT>::makeUnityWeight(Model<TensorT>& model, const TensorT & scale, const std::string& module_name, const std::string& name_format, const std::string& lhs, const std::string& rhs, const bool& specify_layer)
	{
		// Create the unity weight
		char* unity_weight_name_char = new char[512];
		sprintf(unity_weight_name_char, name_format.data(), lhs.data(), rhs.data());
		std::string unity_weight_name(unity_weight_name_char);
		Weight<TensorT> unity_weight(unity_weight_name, std::make_shared<ConstWeightInitOp<TensorT>>(ConstWeightInitOp<TensorT>(scale)), std::make_shared<DummySolverOp<TensorT>>(DummySolverOp<TensorT>()));
		unity_weight.setModuleName(module_name);
    if (specify_layer) unity_weight.setLayerName(module_name);
		model.addWeights({ unity_weight });
		delete[] unity_weight_name_char;
		return unity_weight_name;
	}
}

#endif //EVONET_MODELBUILDER_H