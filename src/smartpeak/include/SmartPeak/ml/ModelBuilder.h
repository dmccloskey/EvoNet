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
	template<typename HDelT, typename DDelT, typename TensorT>
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
		std::vector<std::string> addInputNodes(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const int& n_nodes);

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
		std::vector<std::string> addFullyConnected(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name, 
			const std::vector<std::string>& source_node_names, const int& n_nodes,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true);
		void addFullyConnected(Model<HDelT, DDelT, TensorT>& model, const std::string& module_name,
			const std::vector<std::string>& source_node_names, const std::vector<std::string>& sink_node_names,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver, 
			TensorT drop_connection_prob = 0.0f);

		/**
		@brief Add a Soft Max

		def stable_softmax(X):
		exps = np.exp(X)
		return exps / np.sum(exps)

		@param[in, out] Model<HDelT, DDelT, TensorT>
		@param[in] source_node_names Node_names to add the layer to

		@returns vector of output node names
		*/
		std::vector<std::string> addSoftMax(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& source_node_names);

		/**
		@brief Add a Stable Soft Max

		def stable_softmax(X):
			exps = np.exp(X - np.max(X))
			return exps / np.sum(exps)

		@param[in, out] Model
		@param[in] source_node_names Node_names to add the layer to

		@returns vector of output node names
		*/
		std::vector<std::string> addStableSoftMax(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name, const std::vector<std::string>& source_node_names);

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
		std::vector<std::string> addConvolution(Model<HDelT, DDelT, TensorT> & model, const std::string & name, const std::string& module_name, const std::vector<std::string>& source_node_names,
			const int & input_width, const int & input_height, const int& input_width_zero_padding, const int& input_height_zero_padding,
			const int & extent_width, const int & extent_height, const int & stride,
			const int & output_width_zero_padding, const int& output_height_zero_padding,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true);

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
		std::vector<std::string> addNormalization(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true);

		/**
		@brief Add a VAE Encoding layer with input node

		@param[in, out] Model
		@param[in] mu_node_names Node_names from the average layer
		@param[in] logvar_node_names Nodes names from the logvar layer

		@returns vector of output node names
		*/
		std::vector<std::string> addVAEEncoding(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& mu_node_names, const std::vector<std::string>& logvar_node_names);

		/**
		@brief Add a VAE Encoding layer with input node

		@param[in, out] Model
		@param[in] encoding_node_names Node_names for the latent distribution

		@returns vector of output node names
		*/
		std::vector<std::string> addDiscriminator(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
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
		std::vector<std::string> addLSTM(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_blocks, const int& n_cells,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true,
			bool forget_gate = true, int block_version = 1);
		std::vector<std::string> addLSTMBlock1(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_cells,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true,
			bool forget_gate = true);
		std::vector<std::string> addLSTMBlock2(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_cells,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true,
			bool forget_gate = true);

		/**
		@brief Add a LSTM layer

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
		std::vector<std::string> addGRU(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const int& n_blocks, 
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true,
			bool forget_gate = true, int block_version = 1);
		std::vector<std::string> addGRU1(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true, bool input_gate_connection = true);
		std::vector<std::string> addGRU2(Model<HDelT, DDelT, TensorT>& model, const std::string& name, const std::string& module_name,
			const std::vector<std::string>& source_node_names,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation,
			const std::shared_ptr<ActivationOp<TensorT>>& node_activation_grad,
			const std::shared_ptr<IntegrationOp<TensorT>>& node_integration,
			const std::shared_ptr<IntegrationErrorOp<TensorT>>& node_integration_error,
			const std::shared_ptr<IntegrationWeightGradOp<TensorT>>& node_integration_weight_grad,
			const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
			TensorT drop_out_prob = 0.0f, TensorT drop_connection_prob = 0.0f, bool biases = true, bool input_gate_connection = true);

		/**
		@brief Add one model to another

		@param[in, out] Model
		@param[in] source_node_names Node_names in the LH model to add to
		@param[in] sink_node_names Node names in the RH model to join
		@param[in] model_rh The RH model to add to the LH model

		@returns vector of output node names
		*/
		std::vector<std::string> addModel(Model<HDelT, DDelT, TensorT>& model, const std::vector<std::string>& source_node_names,
			const std::vector<std::string>& sink_node_names, const Model<HDelT, DDelT, TensorT>& model_rh);
  };
}

#endif //SMARTPEAK_MODELBUILDER_H