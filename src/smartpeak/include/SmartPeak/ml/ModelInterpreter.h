/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETER_H
#define SMARTPEAK_MODELINTERPRETER_H

// .h
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/NodeTensorData.h>
#include <SmartPeak/ml/WeightTensorData.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <SmartPeak/ml/SolverTensor.h>
#include <SmartPeak/ml/LossFunctionTensor.h>
#include <SmartPeak/ml/OpToTensorOp.h>
#include <SmartPeak/ml/ModelResources.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <set>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/utility.hpp> // std::pair
#include <cereal/types/vector.hpp>

// .cpp
#include <SmartPeak/ml/ModelErrorData.h>
#include <SmartPeak/ml/ModelKernal.h>

#include <stdexcept>

namespace SmartPeak
{
	/*
	Structures required to identify node operations
	*/
	template<typename TensorT>
	struct OperationResult
	{
		std::shared_ptr<Node<TensorT>> sink_node;
		int time_step = 0;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(sink_node, time_step);
		}
	};

	template<typename TensorT>
	struct OperationArguments
	{
		std::shared_ptr<Node<TensorT>> source_node;
		std::shared_ptr<Weight<TensorT>> weight;
		std::string link_name;
		int time_step = 0;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(source_node, weight, link_name, time_step);
		}
	};

	template<typename TensorT>
	struct OperationList
	{
		OperationResult<TensorT> result;
		std::vector<OperationArguments<TensorT>> arguments;
		int operation_index = -1;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(result, arguments, operation_index);
		}
	};

	/*
	Structures required for layer operations
	*/
	template<typename TensorT, typename DeviceT>
	class OperationLayer
	{
	public:
		std::shared_ptr<NodeTensorData<TensorT, DeviceT>> tensor = nullptr;
		int time_step = 0;
		std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> integration = nullptr;
		std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> integration_error = nullptr;
		std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> integration_weight_grad = nullptr;
		std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> activation = nullptr;
		std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> activation_grad = nullptr;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(tensor, time_step, integration, integration_error, integration_weight_grad,	activation, activation_grad);
		}
	};

	template<typename TensorT, typename DeviceT>
	class OperationWeight
	{
	public:
		std::shared_ptr<WeightTensorData<TensorT, DeviceT>> tensor = nullptr;
		std::shared_ptr<WeightInitOp<TensorT>> weight_init = nullptr;
		std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> solver = nullptr;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(tensor, weight_init, solver);
		}
	};

	/*
	Class used for layer operations
	*/
	template<typename TensorT, typename DeviceT>
	class OperationTensorStep
	{
	public:
		OperationLayer<TensorT, DeviceT> sink_layer;
		OperationLayer<TensorT, DeviceT> source_layer;
		OperationWeight<TensorT, DeviceT> weight;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(sink_layer, source_layer, weight);
		}
	};

	/**
		@brief Directed Network Model Interpreter

		Assumptions about the model structure:
		1. Inputs can only be sources
		2. Outputs can only be sinks (will break back propogation algorithm)
	*/
	template<typename TensorT, typename DeviceT>
	class ModelInterpreter
	{
	public:
		ModelInterpreter() = default; ///< Default constructor
		ModelInterpreter(const ModelInterpreter& other); ///< Copy constructor that does not create a shared memory address between model nodes/links/weights
		ModelInterpreter(const ModelResources& model_resources); ///< Copy constructor that does not create a shared memory address between model nodes/links/weights
		~ModelInterpreter() = default; ///< Default destructor

		inline bool operator==(const ModelInterpreter& other) const
		{
			// BUG: 
			// - 'operator __surrogate_func': no matching overloaded function found
			// - Failed to specialize function template 'unknown-type std::equal_to<void>::operator ()(_Ty1 && _Ty2 &&) const'
			return
				std::tie(
					//operation_steps_,
					//layer_tensors_,
					//weight_tensors_,
					//model_error_,
					//model_resources_
				) == std::tie(
					//other.operation_steps_,
					//other.layer_tensors_,
					//other.weight_tensors_,
					//other.model_error_,
					//other.model_resources_
				);
		}

		inline bool operator!=(const ModelInterpreter& other) const
		{
			return !(*this == other);
		}

		/**
		@brief Copy assignment operator that creates a new model with different memory addresses
		*/
		inline ModelInterpreter& operator=(const ModelInterpreter& other)
		{
			model_resources_ = other.model_resources_;
			return *this;
		}

		/**
			@brief Assigns output or error values to the nodes.
				The node statuses are then changed accordingly (i.e.,
				status_update of "activated" will update the output values
				of the node and status_update of "corrected" will update
				the error values of the node.

			dimensions of batch size by memory size by nodes

			@param[in] values Values to assign to the node
			@param[in] node_names
			@param[in] value_type ("output", "derivative", "error", or "dt")
		*/
		void mapValuesToLayers(
			Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 3>& values,
			const std::vector<std::string>& node_names,
			const std::string& value_type);

		/**
			@brief Initializes the bias nodes to an output of 1

			The reason this is currently required is that layers are not seperated
			by NodeType.  This optimization has the side-effect
			that bias nodes may not be initialized to 1, but instead 0.
			To correct for this, we seperately initialize them here.

			dimensions of batch size by memory size by nodes

			@param[in] model
		*/
		void initBiases(Model<TensorT>& model);

		/**
			@brief Initializes Node Output, Input, Derivative, and Error tensors to 0
		*/
		void reInitNodes();

		/**
			@brief Initializes Model Error to 0
		*/
		void reInitModelError();

		/**
			@brief A prelude to a forward propogation step. Returns a vector of links
				and associated nodes that satisfy the following conditions:
				1. all sink output values are unknown (i.e. inactive),
				2. all source node output values are known (i.e. active).
				3. all nodes need not be the same type

			@param[out] FP_operations_map Key/Value pair of sink node name to FP_operations index
			@param[out] FP_operations
		*/
		void getNextInactiveLayer(Model<TensorT>& model,
			std::map<std::string, int>& FP_operations_map,
			std::vector<OperationList<TensorT>>& FP_operations);
		void getNextInactiveLayerWOBiases(Model<TensorT>& model,
			std::map<std::string, int>& FP_operations_map,
			std::vector<OperationList<TensorT>>& FP_operations);

		/**
			@brief Continuation of the forward propogation step that identifies all biases
				for the identified sink nodes. Returns a vector of links
				and associated nodes that satisfy the following conditions:
				1. all sink output values are unknown (i.e. inactive),
				2. all source node output values are known (i.e. active) and biases.

			@param[out] FP_operations_map Key/Value pair of sink node name to FP_peroations index
			@param[out] FP_operations
			@param[out] sink_nodes_with_biases
		*/
		void getNextInactiveLayerBiases(Model<TensorT>& model,
			std::map<std::string, int>& FP_operations_map,
			std::vector<OperationList<TensorT>>& FP_operations,
			std::vector<std::string>& sink_nodes_with_biases
		);

		/**
			@brief Continuation of the forward propogation step that identifies
				all cyclic source nodes for the identified sink nodes. Returns a vector of links
				and associated nodes that satisfy the following conditions:
				1. all sink output values are unknown (i.e. inactive),
				2. all source node output values are unknown (i.e. inactive).

			@param[out] FP_operations_map Key/Value pair of sink node name to FP_peroations index
			@param[out] FP_operations
			@param[out] sink_nodes_with_cycles
		*/
		void getNextInactiveLayerCycles(Model<TensorT>& model,
			std::map<std::string, int>& FP_operations_map,
			std::vector<OperationList<TensorT>>& FP_operations,
			std::vector<std::string>& sink_nodes_with_cycles);

		/**
			@brief Prunes identified cyclic nodes that are not in fact part of a cycle
				but are instead not yet activated and not yet ready to fire.

			[TODO: add tests!]

			@param[out] FP_operations_map Key/Value pair of sink node name to FP_peroations index
			@param[out] FP_operations
			@param[out] sink_nodes_with_cycles
		*/
		void pruneInactiveLayerCycles(Model<TensorT>& model,
			std::map<std::string, int>& FP_operations_map,
			std::map<std::string, int>& FP_operations_map_cycles,
			std::vector<OperationList<TensorT>>& FP_operations,
			std::vector<OperationList<TensorT>>& FP_operations_cycles,
			std::vector<std::string>& sink_nodes_with_cycles);

		/**
			@brief Expands the current operation list to satisfy the following assumptions:
			1. arguments for a given sink node have the same time-step/activation/node_integration
			2. all links have the same solver and weight_init operator
			3. arguments are not a mix of nodes from pre-identified layers and nodes that have not yet been partitioned into a layer

			[TODO: add tests!]

			@param[in] FP_operations
			@param[out] FP_operations_expanded Expanded FP_operations list
		*/
		void expandForwardPropogationOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded);
		void expandForwardPropogationOperationsBySourceNodeKey(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded);
		void expandForwardPropogationOperationsByWeightKey(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded);
		void expandForwardPropogationOperationsByCachedNodes(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded);
		void expandAllForwardPropogationOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded);

		/**
			@brief Re-organizes the identified layers into tensors and attempts to optimizes
				the layer operations to maximize hardware acceleration.

			If possible, layer operations will be partitioned into predefined tensor integration motifs
      in order to optimize matrix size and memory usage
			- Tensor integration motifs: FC/SC/Conv/FanIn/FanOut
			- Custom

			Criteria for FC (Not implemented; Subset of matrix multiplication)
			- all arguments for sinks are equal

			Criteria for FanIn and FanOut (Not implemented; Subset of matrix multiplication)
			- FanIn: 1 sink, multiple sources
			- FanOut: 1 source, multiple sinks

			Criteria for SC
			- unique argument per sinks

			Criteria for Conv/pool
			- shared weights with FanIn

			Critera for Custom
			- Module with optimized computation (e.g., softmax, attention, etc.,)
		*/

		/**
		@brief Identify layer operations

		[TODO: add tests]

		@param[in] FP_operations
		@param[in] identified_sink_nodes Set of identified sink nodes
		@param[in] fast_check Skips the most time intensive check required for models without layer name specifications

		@returns map of identified operations consisting of the identifying sink node name or module name
			for the operation and a list of indices corresponding to the operations in FP_operations
		*/
		std::map<std::string, std::vector<int>> getCustomOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		std::map<std::string, std::vector<int>> GetSinglyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		//std::map<std::string, std::vector<int>> getConvOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		std::map<std::string, std::vector<int>> getTensorOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes, const bool& fast_check);

		/**
		@brief Allocate Node and Weight tensor memory for all model operations.
			Node and weight tensors indices and tensor sizes are registered.

		@param[in] FP_operations
		@param[in] operations_map
		@param[out] source_layer_sizes
		@param[out] sink_layer_sizes
		@param[out] weight_indices
		@param[out] shared_weight_indices
		@param[out] weight_values
		@param[out] make_source_tensors
		@param[out] make_sink_tensors
		@param[out] make_weight_tensors
		@param[out] batch_size
		@param[out] memory_size
		@param[out] train
		*/
		void getForwardPropogationLayerTensorDimensions(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes, std::vector<std::vector<std::pair<int, int>>>& weight_indices, 
			std::vector<std::map<std::string, std::vector<std::pair<int, int>>>>& shared_weight_indices, std::vector<std::vector<TensorT>>& weight_values,
			std::vector<bool>& make_source_tensor, std::vector<bool>& make_sink_tensor, std::vector<bool>& make_weight_tensor);

		/**
		@brief Create a unique key to different nodes by time_step, node_integration, and node_activation methods

		@param[in] time_step
		@param[in] node_type [Currently not used]
		@param[in] node_integration
		@param[in] node_activation
		@param[in] layer_name
		@param[in] layer_index
		*/
		static std::string makeForwardPropogationOperationsKey(const int & time_step, const NodeType& node_type, const std::string & node_integration, const std::string & node_activation,
			const std::string& node_layer_name, const int& node_layer_index, const std::string& weight_layer_name);

		/**
		@brief Convert a graph model to sequence of tensor operations

		@param[in, out] model Network model
		@param[in] batch_size Batch size
		@param[in] memory_size Memory size
		@param[in] train Boolean to indicate training or testing (needed for dropout or drop connection)
		@param[in] fast_check Boolean to use a faster but incomplete tensor compatibility check when manually specifying layers
		@param[in] find_cycles Boolean to search for cyclic nodes
		@param[in] preserve_OoO Boolean to indicate whether the order of operation (OoO) of the model should be preserved (true) or
			the model should be treated as a graph where all operations happen simultaneously (false)
		*/
		void getForwardPropogationOperations(Model<TensorT>& model, const int& batch_size, const int& memory_size, const bool& train, const bool& fast_check, const bool& find_cycles, const bool& preserve_OoO);
		
		/**
		@brief Convert a graph model to sequence of tensor operations
			preserving the order of operations

		@param[in, out] model Network model
		@param[in] find_cycles Boolean to search for cyclic nodes
		@param[out] FP_operations_expanded List of forward (and reverse) operations
		@param[out] iter Number of operations
		*/
		void getFPOpsOoO_(Model<TensorT>& model, const bool& find_cycles, std::vector<OperationList<TensorT>>& FP_operations_expanded, int& iter);

		/**
		@brief Convert a graph model to sequence of tensor operations
			preserving the order of operations

		@param[in, out] model Network model
		@param[in] find_cycles Boolean to search for cyclic nodes
		@param[out] FP_operations_expanded List of forward (and reverse) operations
		@param[out] iter Number of operations
		*/
		void getFPOpsGraph_(Model<TensorT>& model, std::vector<OperationList<TensorT>>& FP_operations_expanded, int& iter);

		/**
		@brief Allocate Node and Weight tensor memory for all model operations.
			Source and sink layer activations are created using the first node in the layers, respecively.
			Source and sink layer integrations are created using the first node in the layers, respectively.
			Weight solver params tensors are created using the first weight in the layer.
			Weight matrix is initialized using the first weight in the layer.

		@param[in] FP_operations
		@param[in] operations_map
		@param[in] source_layer_sizes
		@param[in] sink_layer_sizes
		@param[in] weight_indices
		@param[in] shared_weight_indices
		@param[in] weight_values
		@param[in] make_source_tensors
		@param[in] make_sink_tensors
		@param[in] make_weight_tensors
		@param[in] batch_size
		@param[in] memory_size
		@param[in] train
		*/
		virtual void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, 
			std::vector<std::map<std::string, std::vector<std::pair<int, int>>>>& shared_weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size, const bool& train) = 0;

		/**
		@brief Execute model kernal methods required for forward propogation

		@param[in] time_step The current time-step to operate on
		@param[in] sync_HToD Short circuit for testing that copies all host data to the device
		@param[in] sync_DToH Short circuit for testing that copies all device data to the host
		*/
		virtual void executeForwardPropogationOperations(const int& time_step) = 0;

		/**
		@brief Execute model kernal methods required for calculating the model and output node error

		@param[in] time_step The current time-step to operate on
		@param[in] sync_HToD Short circuit for testing that copies all host data to the device
		@param[in] sync_DToH Short circuit for testing that copies all device data to the host
		*/
		virtual void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const int& layer_id, LossFunctionTensorOp<TensorT, DeviceT>* loss_function, LossFunctionGradTensorOp<TensorT, DeviceT>* loss_function_grad, const int& time_step) = 0;

		/**
		@brief Execute model kernal methods required for backward propogation

		@param[in] time_step The current time-step to operate on
		@param[in] sync_HToD Short circuit for testing that copies all host data to the device
		@param[in] sync_DToH Short circuit for testing that copies all device data to the host
		*/
		virtual void executeBackwardPropogationOperations(const int& time_step) = 0;

		/**
		@brief Execute model kernal methods required for weight error calculations

		@param[in] sync_HToD Short circuit for testing that copies all host data to the device
		@param[in] sync_DToH Short circuit for testing that copies all device data to the host
		*/
		virtual void executeWeightErrorOperations() = 0;

		/**
		@brief Execute model kernal methods required for weight update calculations

		@param[in] sync_HToD Short circuit for testing that copies all host data to the device
		@param[in] sync_DToH Short circuit for testing that copies all device data to the host
		*/
		virtual void executeWeightUpdateOperations() = 0;
		
		void addLayerTensor(NodeTensorData<TensorT, DeviceT>& layer); ///< add a layer to the cache
		void clearLayerTensors(); ///< clear all layers from the cache
		std::shared_ptr<NodeTensorData<TensorT, DeviceT>> getLayerTensor(const int& layer_index); ///< get a layer from the cache

		void addWeightTensor(WeightTensorData<TensorT, DeviceT>& weight); ///< add a weight to the cache
		void clearWeightTensors(); ///< clear all weights from the cache
		std::shared_ptr<WeightTensorData<TensorT, DeviceT>> getWeightTensor(const int& weight_index); ///< get a weight from the cache

		virtual void allocateModelErrorTensor(const int& batch_size, const int& memory_size) = 0; ///< set the model error
		std::shared_ptr<ModelErrorData<TensorT, DeviceT>> getModelError(); ///< get the model error

		void addOperationSteps(const std::vector<OperationTensorStep<TensorT, DeviceT>>& operation_steps);
		std::vector<OperationTensorStep<TensorT, DeviceT>> getOperationSteps(const int& operation_index);
		void clearOperationSteps(); ///< clear the operations caches
 
		/**
		@brief Foward propogation through time (FPTT) of the network model.

		@param[in] time_steps The number of time_steps forward to
			continuously calculate node outputs.
		@param[in] values Input values at each time step where
			dim0: batch_size, dim1: time_step, and dim2: nodes.
		@param[in] node_names
		@param[in] dt Node time resolution
		*/
		void FPTT(const int& time_steps);

		/**
		@brief Calculates the error of the model through time (CETT)
			with respect to the expected values

		@param[in] values Expected node output values
			(dim0: batch_size, dim1: memory_size, dim2: output nodes)
			where t=n to t=0
		@param[in] node_names Output nodes
		*/
		void CETT(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& values, const std::vector<std::string>& node_names, LossFunctionOp<TensorT>* loss_function, LossFunctionGradOp<TensorT>* loss_function_grad, const int& time_steps);

		/**
		@brief Truncated Back Propogation Through Time (TBPTT) of the network model.

		@param[in] time_steps The number of time_steps backwards to
			unfold the network model.
		*/
		void TBPTT(const int& time_steps);

		/**
		@brief Recurrent Real Time Learning (RTRL) of the network model.

		@param[in] time_steps The number of time_steps backwards to
			unfold the network model.
		*/
		void RTRL(const int& time_steps);

		/**
		@brief Update the weights

		*/
		void updateWeights();

		/**
		@brief Transfer Model error, weights, and output node values
			from the model interpreter to the model

		@param[in, out] model The network model
		*/
		virtual void getModelResults(Model<TensorT>& model, bool output_nodes = true, bool weights = true, bool model_error = true) = 0;

		/**
		@brief Update the weight solver params

		NOTE: this method is only safe for updating the learning rate.  More sophisticated checks
			would need to be implemented for updating other paramaters when multiple solvers can
			be used.

		@param[in] param_index The parameter index to update (i.e., 0 for learning rate)
		@param[in] param_value The parameter value to set (i.e., 1e-4)
		*/
		virtual void updateSolverParams(const int& param_index, const TensorT& param_value) = 0;

		void setModelResources(const ModelResources& model_resources); ///< model_resources setter
		ModelResources getModelResources(); ///< model_resources getter

		virtual void checkMemory(const Model<TensorT>& model, const int& batch_size, const int& memory_size) = 0;

		void clear_cache();

		std::vector<std::map<std::string, std::vector<int>>> getTensorOpsSteps() const;

	protected:
		std::vector<std::vector<OperationTensorStep<TensorT, DeviceT>>> operation_steps_;
		std::vector<std::shared_ptr<NodeTensorData<TensorT, DeviceT>>> layer_tensors_;
		std::vector<std::shared_ptr<WeightTensorData<TensorT, DeviceT>>> weight_tensors_;
		std::shared_ptr<ModelErrorData<TensorT, DeviceT>> model_error_;
		ModelResources model_resources_;

	private:
		std::vector<std::map<std::string, std::vector<int>>> tensor_ops_steps_;
		std::vector<OperationList<TensorT>> FP_operations_;
		friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(operation_steps_, layer_tensors_, weight_tensors_, model_error_, model_resources_);
		//}
		template<class Archive>
		void serialize(Archive& archive) {
			archive(tensor_ops_steps_, FP_operations_, model_resources_);
		}
	};

	template<typename TensorT, typename DeviceT>
	ModelInterpreter<TensorT, DeviceT>::ModelInterpreter(const ModelInterpreter<TensorT, DeviceT>& other)
	{
		model_resources_ = other.model_resources_;
	}

	template<typename TensorT, typename DeviceT>
	ModelInterpreter<TensorT, DeviceT>::ModelInterpreter(const ModelResources& model_resources): model_resources_(model_resources)
	{
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::mapValuesToLayers(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& values, const std::vector<std::string>& node_names, const std::string & value_type)
	{
		if (layer_tensors_.size() <= 0) {
			char error_char[512];
			sprintf(error_char, "Tensor layers have not been created.  Cannot map values to layers.");
			std::string error(error_char);
			throw std::runtime_error(error_char);
		}

		// Buffer the input values
		Eigen::Tensor<TensorT, 3> values_buffered = values.pad(Eigen::array<std::pair<int, int>, 3>({std::make_pair(0,0),std::make_pair(0,1),std::make_pair(0,0)}));

		// check dimension mismatches
		if (node_names.size() != values_buffered.dimension(2))
		{
			printf("The number of input features %d and the number of nodes %d do not match.\n", (int)values_buffered.dimension(2), node_names.size());
			return;
		}
		// assumes the tensors have been cached
		else if (layer_tensors_[0]->getBatchSize() != values_buffered.dimension(0))
		{
			printf("The number of input samples %d and the batch size %d does not match.\n", (int)values_buffered.dimension(0), (int)layer_tensors_[0]->getBatchSize());
			return;
		}
		else if (layer_tensors_[0]->getMemorySize() != values_buffered.dimension(1))
		{
			printf("The number of input time steps %d and the memory size %d does not match.\n", (int)values_buffered.dimension(1), (int)layer_tensors_[0]->getMemorySize());
			return;
		}

		for (int i = 0; i < node_names.size(); ++i){
			auto node = model.nodes_.at(node_names[i]);
			if (node->getTensorIndex().first != -1) {
				// copy over the values
				if (value_type == "output")
					getLayerTensor(node->getTensorIndex().first)->getOutput().chip(node->getTensorIndex().second, 2) = values_buffered.chip(i, 2);
				if (value_type == "input")
					getLayerTensor(node->getTensorIndex().first)->getInput().chip(node->getTensorIndex().second, 2) = values_buffered.chip(i, 2);
				else if (value_type == "error")
					getLayerTensor(node->getTensorIndex().first)->getError().chip(node->getTensorIndex().second, 2) = values_buffered.chip(i, 2);
				else if (value_type == "derivative")
					getLayerTensor(node->getTensorIndex().first)->getDerivative().chip(node->getTensorIndex().second, 2) = values_buffered.chip(i, 2);
				else if (value_type == "dt")
					getLayerTensor(node->getTensorIndex().first)->getDt().chip(node->getTensorIndex().second, 2) = values_buffered.chip(i, 2);
			}
			else {
				clear_cache(); // clean up before exiting
				char error_char[512];
				sprintf(error_char, "Node %s has not been assigned a tensor index!", node->getName().data());
				std::string error(error_char);
				throw std::runtime_error(error_char);
				// Error is cause by an added recursive link that "blocks" forward propogation
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::initBiases(Model<TensorT>& model)
	{
		if (layer_tensors_.size() <= 0) {
			char error_char[512];
			sprintf(error_char, "Tensor layers have not been created.  Cannot initiate biases.");
			std::string error(error_char);
			throw std::runtime_error(error_char);
		}
		Eigen::Tensor<TensorT, 2> one((int)layer_tensors_[0]->getBatchSize(), (int)layer_tensors_[0]->getMemorySize());	one.setConstant(1);
		for (auto& node_map : model.nodes_) {
			if (node_map.second->getType() == NodeType::bias) {
				if (node_map.second->getTensorIndex().first != -1) {
					getLayerTensor(node_map.second->getTensorIndex().first)->getOutput().chip(node_map.second->getTensorIndex().second, 2) = one;
				}
				else {
					clear_cache(); // clean up before exiting
					char error_char[512];
					sprintf(error_char, "Node %s has not been assigned a tensor index!", node_map.second->getName().data());
					std::string error(error_char);
					throw std::runtime_error(error_char);
					// Error is cause by an added recursive link that "blocks" forward propogation
				}
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::reInitNodes()
	{
		for (auto& layer_tensor: layer_tensors_) {
			Eigen::Tensor<TensorT, 3> zero((int)layer_tensor->getBatchSize(), (int)layer_tensor->getMemorySize(), (int)layer_tensor->getLayerSize());	zero.setConstant(0);
      Eigen::Tensor<TensorT, 3> one((int)layer_tensor->getBatchSize(), (int)layer_tensor->getMemorySize(), (int)layer_tensor->getLayerSize()); one.setConstant(1);
      if (layer_tensor->getLayerIntegration() == "ProdOp" || layer_tensor->getLayerIntegration() == "ProdSCOp") {
        layer_tensor->setInput(one);
        layer_tensor->setOutput(zero);
        layer_tensor->setDerivative(zero);
        layer_tensor->setError(zero);
        layer_tensor->setDt(zero);
      }
      else {
        layer_tensor->setInput(zero);
        layer_tensor->setOutput(zero);
        layer_tensor->setDerivative(zero);
        layer_tensor->setError(zero);
        layer_tensor->setDt(zero);
      }
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::reInitModelError()
	{
		Eigen::Tensor<TensorT, 2> zero((int)model_error_->getBatchSize(), (int)model_error_->getMemorySize());	zero.setConstant(0);
		model_error_->setError(zero);
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getNextInactiveLayer(Model<TensorT>& model,
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations)
	{
		// get all links where the source node is active and the sink node is inactive
		// except for biases
		for (auto& link_map : model.links_)
		{
			if (
				//model.nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias &&
				model.nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				model.nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = model.nodes_.at(link_map.second->getSourceNodeName());
				arguments.weight = model.weights_.at(link_map.second->getWeightName());
				arguments.time_step = 0;
				arguments.link_name = link_map.first;

				std::string ops_key = link_map.second->getSinkNodeName();
				auto found = FP_operations_map.emplace(ops_key, (int)FP_operations.size());
				if (!found.second)
				{
					FP_operations[FP_operations_map.at(ops_key)].arguments.push_back(arguments);
				}
				else
				{
					OperationList<TensorT> operation_list;
					OperationResult<TensorT> result;
					result.sink_node = model.nodes_.at(link_map.second->getSinkNodeName());
					operation_list.result = result;
					operation_list.arguments.push_back(arguments);
					FP_operations.push_back(operation_list);
				}
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getNextInactiveLayerWOBiases(Model<TensorT>& model,
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations)
	{
		// get all links where the source node is active and the sink node is inactive
		// except for biases
		for (auto& link_map : model.links_)
		{
			if (
				model.nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias &&
				model.nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				model.nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized)
			{
				//if (FP_operations.size() == 680)
				//	std::cout << "check" << std::endl;
				OperationArguments<TensorT> arguments;
				arguments.source_node = model.nodes_.at(link_map.second->getSourceNodeName());
				arguments.weight = model.weights_.at(link_map.second->getWeightName());
				arguments.time_step = 0;
				arguments.link_name = link_map.first;

				std::string ops_key = link_map.second->getSinkNodeName();
				auto found = FP_operations_map.emplace(ops_key, (int)FP_operations.size());
				if (!found.second)
				{
					FP_operations[FP_operations_map.at(ops_key)].arguments.push_back(arguments);
				}
				else
				{
					OperationList<TensorT> operation_list;
					OperationResult<TensorT> result;
					result.sink_node = model.nodes_.at(link_map.second->getSinkNodeName());
					operation_list.result = result;
					operation_list.arguments.push_back(arguments);
					FP_operations.push_back(operation_list);
				}
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getNextInactiveLayerBiases(Model<TensorT>& model,
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations,
		std::vector<std::string>& sink_nodes_with_biases)
	{

		// get all the biases for the sink nodes
		for (auto& link_map : model.links_)
		{
			std::string ops_key = link_map.second->getSinkNodeName();
			if (
				// does not allow for cycles
				model.nodes_.at(link_map.second->getSourceNodeName())->getType() == NodeType::bias &&
				model.nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				// required regardless if cycles are or are not allowed
				model.nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
				FP_operations_map.count(ops_key) != 0 // sink node has already been identified
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = model.nodes_.at(link_map.second->getSourceNodeName());
				arguments.weight = model.weights_.at(link_map.second->getWeightName());
				arguments.time_step = 0;
				arguments.link_name = link_map.first;
				FP_operations[FP_operations_map.at(ops_key)].arguments.push_back(arguments);
				if (std::count(sink_nodes_with_biases.begin(), sink_nodes_with_biases.end(), ops_key) == 0)
				{
					sink_nodes_with_biases.push_back(ops_key);
				}
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getNextInactiveLayerCycles(Model<TensorT>& model,
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations,
		std::vector<std::string>& sink_nodes_with_cycles)
	{

		// get cyclic source nodes
		for (auto& link_map : model.links_)
		{
			std::string ops_key = link_map.second->getSinkNodeName();
			if (
				model.nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::initialized &&
				// required regardless if cycles are or are not allowed
				model.nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
				FP_operations_map.count(ops_key) != 0 // sink node has already been identified
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = model.nodes_.at(link_map.second->getSourceNodeName());
				arguments.weight = model.weights_.at(link_map.second->getWeightName());

				arguments.time_step = 1;
				arguments.link_name = link_map.first;
				FP_operations[FP_operations_map.at(ops_key)].arguments.push_back(arguments);
				sink_nodes_with_cycles.push_back(ops_key);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::pruneInactiveLayerCycles(Model<TensorT>& model, std::map<std::string, int>& FP_operations_map, std::map<std::string, int>& FP_operations_map_cycles, std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_cycles, std::vector<std::string>& sink_nodes_with_cycles)
	{

		// Remove all nodes involved in "cycles" that have arguments
		// involving source to sink node pairs not identified as cycles
		if (sink_nodes_with_cycles.size() > 0)
		{
			std::vector<std::string> sink_nodes_remove;
			std::vector<OperationList<TensorT>> FP_operations_copy = FP_operations;
			for (const std::string& sink_node : sink_nodes_with_cycles) {
				for (size_t i = FP_operations[FP_operations_map.at(sink_node)].arguments.size();
					i < FP_operations_cycles[FP_operations_map_cycles.at(sink_node)].arguments.size(); ++i) {
					// check if the "cyclic" argument is actually involved in a cycle
					bool isCyclicOperation = false;
					for (const auto& cyclic_pair : model.getCyclicPairs()) {
						if (FP_operations_cycles[FP_operations_map_cycles.at(sink_node)].arguments[i].source_node->getName() == cyclic_pair.first &&
							FP_operations_cycles[FP_operations_map_cycles.at(sink_node)].result.sink_node->getName() == cyclic_pair.second) {
							isCyclicOperation = true;
							break;
						}
					}
					// copy over the cyclic operation
					if (isCyclicOperation)
						FP_operations_copy[FP_operations_map_cycles.at(sink_node)].arguments.push_back(FP_operations_cycles[FP_operations_map_cycles.at(sink_node)].arguments[i]);
					// id the sink node for removal
					else {
						sink_nodes_remove.push_back(sink_node);
						break;
					}
				}
			}
			// remove all identified sink nodes
			if (sink_nodes_remove.size() > 0) {
				FP_operations.clear();
				for (const auto& FP_operation : FP_operations_copy)
					if (std::count(sink_nodes_remove.begin(), sink_nodes_remove.end(), FP_operation.result.sink_node->getName()) == 0)
						FP_operations.push_back(FP_operation);
			}
			else
				FP_operations = FP_operations_copy;
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::expandForwardPropogationOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		std::vector<OperationList<TensorT>> FP_operations_1, FP_operations_2;
		expandForwardPropogationOperationsBySourceNodeKey(FP_operations, FP_operations_1); // Pass 1:		 
		expandForwardPropogationOperationsByWeightKey(FP_operations_1, FP_operations_2); // Pass 2:		
		expandForwardPropogationOperationsByCachedNodes(FP_operations_2, FP_operations_expanded); // Pass 3: 
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::expandForwardPropogationOperationsBySourceNodeKey(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		FP_operations_expanded.clear();
		for (const OperationList<TensorT>& FP_operation : FP_operations) {
			// check that all arguments have the same time-step/activation/node_integration
			std::set<std::string> unique_node_types;
			for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
				std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
					argument.source_node->getType(),
					argument.source_node->getIntegration()->getName(),
					argument.source_node->getActivation()->getName(),
					argument.source_node->getLayerName(),
					argument.source_node->getTensorIndex().first,
					"");
				unique_node_types.insert(ops_key);
			}
			for (const std::string& node_types : unique_node_types) {
				OperationList<TensorT> operations_list;
				operations_list.result = FP_operation.result;
				for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.source_node->getType(),
						argument.source_node->getIntegration()->getName(),
						argument.source_node->getActivation()->getName(),
						argument.source_node->getLayerName(),
						argument.source_node->getTensorIndex().first,
						"");
					if (node_types == ops_key) {
						operations_list.arguments.push_back(argument);
					}
				}
				operations_list.operation_index = FP_operation.operation_index;
				FP_operations_expanded.push_back(operations_list);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::expandForwardPropogationOperationsByWeightKey(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		FP_operations_expanded.clear();
		for (const OperationList<TensorT>& FP_operation : FP_operations) {
			// check that all links have the same solver/weight_init operator
			std::set<std::string> unique_weight_types;
			for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
				// Does not account for different solver parameters and weight init op parameters!
				std::string ops_key = argument.weight->getSolverOp()->getName() + "/" + argument.weight->getWeightInitOp()->getName() + "/" + argument.weight->getLayerName();
				unique_weight_types.insert(ops_key);
			}
			for (const std::string& weight_types : unique_weight_types) {
				OperationList<TensorT> operations_list;
				operations_list.result = FP_operation.result;
				for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
					std::string ops_key = argument.weight->getSolverOp()->getName() + "/" + argument.weight->getWeightInitOp()->getName() + "/" + argument.weight->getLayerName();
					if (weight_types == ops_key) {
						operations_list.arguments.push_back(argument);
					}
				}
				operations_list.operation_index = FP_operation.operation_index;
				FP_operations_expanded.push_back(operations_list);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::expandForwardPropogationOperationsByCachedNodes(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		FP_operations_expanded.clear();
		for (const OperationList<TensorT>& FP_operation : FP_operations) {
			// check that all nodes are either cached or not yet cached into a layer
			OperationList<TensorT> operations_list_cached, operations_list;
			operations_list.result = FP_operation.result;
			operations_list.operation_index = FP_operation.operation_index;
			operations_list_cached.result = FP_operation.result;
			operations_list_cached.operation_index = FP_operation.operation_index;
			for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
				if (argument.source_node->getTensorIndex().first == -1) {
					operations_list.arguments.push_back(argument);
				}
				else {
					operations_list_cached.arguments.push_back(argument);
				}
			}
			if (operations_list.arguments.size() > 0)
				FP_operations_expanded.push_back(operations_list);
			if (operations_list_cached.arguments.size() > 0)
				FP_operations_expanded.push_back(operations_list_cached);
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::expandAllForwardPropogationOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		FP_operations_expanded.clear();
		for (const OperationList<TensorT>& FP_operation : FP_operations) {
			for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
				OperationList<TensorT> operations_list;
				operations_list.result = FP_operation.result;
				operations_list.arguments.push_back(argument);
				operations_list.operation_index = FP_operation.operation_index;
				FP_operations_expanded.push_back(operations_list);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getCustomOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		std::set<std::string> supported_custom_module_names = { "SoftMax" }; // [TODO: add support for ModuleType]
		std::map<std::string, std::vector<int>> custom_layers;
		for (size_t operations_iter = 0; operations_iter < FP_operations.size(); ++operations_iter) {
			if (identified_sink_nodes.count(FP_operations[operations_iter].result.sink_node->getName())) continue; // Skip identified sink nodes
			if (supported_custom_module_names.count(FP_operations[operations_iter].result.sink_node->getModuleName())) { // [TODO: replace with comparison after adding support for module types]
				std::string sink_node_key = FP_operations[operations_iter].result.sink_node->getName() + std::to_string(operations_iter);
				identified_sink_nodes.insert(sink_node_key);
				auto found = custom_layers.emplace(FP_operations[operations_iter].result.sink_node->getModuleName(), std::vector<int>({ operations_iter }));
				if (!found.second) {
					custom_layers.at(FP_operations[operations_iter].result.sink_node->getModuleName()).push_back(operations_iter);
				}
			}
		}
		return custom_layers;
	}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::GetSinglyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
    std::map<std::string, std::vector<int>> SC_layers;
		for (size_t operations_iter1 = 0; operations_iter1 < FP_operations.size(); ++operations_iter1) {
      if (FP_operations[operations_iter1].arguments.size() != 1) continue; // Not singly connected
      if (FP_operations[operations_iter1].result.sink_node->getIntegration()->getName() != "ProdOp") continue; // Only supports product integration for now
      std::string sink_node_key1 = FP_operations[operations_iter1].result.sink_node->getName() + "/" + std::to_string(operations_iter1);
      if (identified_sink_nodes.count(sink_node_key1)) continue; // Skip identified sink nodes
			for (size_t operations_iter2 = operations_iter1 + 1; operations_iter2 < FP_operations.size(); ++operations_iter2) {
				if (FP_operations[operations_iter2].arguments.size() != 1) continue; // Not singly connected
        std::string sink_node_key2 = FP_operations[operations_iter2].result.sink_node->getName() + "/" + std::to_string(operations_iter2);
        if (identified_sink_nodes.count(sink_node_key2)) continue; // Skip identified sink nodes

			// check if the sink nodes are compatible
        std::string sink_ops_key_1 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].result.time_step,
          FP_operations[operations_iter1].result.sink_node->getType(),
          FP_operations[operations_iter1].result.sink_node->getIntegration()->getName(),
          FP_operations[operations_iter1].result.sink_node->getActivation()->getName(),
          FP_operations[operations_iter1].result.sink_node->getLayerName(),
          FP_operations[operations_iter1].result.sink_node->getTensorIndex().first,
          FP_operations[operations_iter1].arguments[0].weight->getLayerName());
        std::string sink_ops_key_2 = makeForwardPropogationOperationsKey(FP_operations[operations_iter2].result.time_step,
          FP_operations[operations_iter2].result.sink_node->getType(),
          FP_operations[operations_iter2].result.sink_node->getIntegration()->getName(),
          FP_operations[operations_iter2].result.sink_node->getActivation()->getName(),
          FP_operations[operations_iter2].result.sink_node->getLayerName(),
          FP_operations[operations_iter2].result.sink_node->getTensorIndex().first,
          FP_operations[operations_iter2].arguments[0].weight->getLayerName());
        if (sink_ops_key_1 != sink_ops_key_2) continue;

        // check if the source nodes are compatible
        std::set<std::string> argument1_nodes, argument2_nodes;
        for (const auto& argument : FP_operations[operations_iter1].arguments) {
          std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
            argument.source_node->getType(),
            argument.source_node->getIntegration()->getName(),
            argument.source_node->getActivation()->getName(),
            argument.source_node->getLayerName(),
            argument.source_node->getTensorIndex().first,
            argument.weight->getLayerName());
          argument1_nodes.insert(ops_key);
        }
        for (const auto& argument : FP_operations[operations_iter2].arguments) {
          std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
            argument.source_node->getType(),
            argument.source_node->getIntegration()->getName(),
            argument.source_node->getActivation()->getName(),
            argument.source_node->getLayerName(),
            argument.source_node->getTensorIndex().first,
            argument.weight->getLayerName());
          argument2_nodes.insert(ops_key);
        }
        if (argument1_nodes != argument2_nodes) continue;

				// update the maps
        identified_sink_nodes.insert(sink_node_key1);
        identified_sink_nodes.insert(sink_node_key2);
        std::vector<int> first_operation = { (int)operations_iter1 };
        auto found = SC_layers.emplace(sink_node_key1, first_operation);
        SC_layers.at(sink_node_key1).push_back(operations_iter2);
			}
		}
		return SC_layers;
	}

	//template<typename TensorT, typename DeviceT>
	//inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getConvOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	//{
	//	std::map<std::string, std::vector<int>> Conv_layers;
	//	// getConvOperations (special case of multiple FanIn with shared weights)
	//	for (size_t operations_iter1 = 0; operations_iter1 < FP_operations.size(); ++operations_iter1) {
	//		if (identified_sink_nodes.count(FP_operations[operations_iter1].result.sink_node->getName())) continue; // Skip identified sink nodes
	//		for (size_t operations_iter2 = operations_iter1 + 1; operations_iter2 < FP_operations.size(); ++operations_iter2) {
	//			if (identified_sink_nodes.count(FP_operations[operations_iter2].result.sink_node->getName())) continue; // Skip identified sink nodes

	//			// check if the sink nodes are compatible
	//			std::string ops_key_1 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].result.time_step,
	//				FP_operations[operations_iter1].result.sink_node->getType(),
	//				FP_operations[operations_iter1].result.sink_node->getIntegration()->getName(),
	//				FP_operations[operations_iter1].result.sink_node->getActivation()->getName());
	//			std::string ops_key_2 = makeForwardPropogationOperationsKey(FP_operations[operations_iter2].result.time_step,
	//				FP_operations[operations_iter2].result.sink_node->getType(),
	//				FP_operations[operations_iter2].result.sink_node->getIntegration()->getName(),
	//				FP_operations[operations_iter2].result.sink_node->getActivation()->getName());
	//			if (ops_key_1 != ops_key_2) continue;

	//			// check for shared weights
	//			std::set<std::string> argument_weights, argument_weights_1, argument_weights_2;
	//			for (const auto& argument : FP_operations[operations_iter1].arguments) {
	//				argument_weights.insert(argument.weight->getName());
	//				argument_weights_1.insert(argument.weight->getName());
	//			}
	//			for (const auto& argument : FP_operations[operations_iter2].arguments) {
	//				argument_weights.insert(argument.weight->getName());
	//				argument_weights_2.insert(argument.weight->getName());
	//			}
	//			if (argument_weights.size() != argument_weights_1.size() || argument_weights.size() != argument_weights_2.size()) continue;

	//			// update the maps
	//			identified_sink_nodes.insert(FP_operations[operations_iter1].result.sink_node->getName());
	//			auto found = Conv_layers.emplace(FP_operations[operations_iter1].result.sink_node->getName(), std::vector<int>({ operations_iter1 }));
	//			Conv_layers.at(FP_operations[operations_iter1].result.sink_node->getName()).push_back(operations_iter2);
	//		}
	//	}
	//	return Conv_layers;
	//}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getTensorOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes, const bool& fast_check)
	{
		std::map<std::string, std::vector<int>> FC_layers;
		for (size_t operations_iter1 = 0; operations_iter1 < FP_operations.size(); ++operations_iter1) {
			std::string sink_node_key1 = FP_operations[operations_iter1].result.sink_node->getName() + "/" + std::to_string(operations_iter1);
			if (identified_sink_nodes.count(sink_node_key1)) continue; // Skip identified sink nodes

			// Check for compatibility
			for (size_t operations_iter2 = operations_iter1 + 1; operations_iter2 < FP_operations.size(); ++operations_iter2) {
				std::string sink_node_key2 = FP_operations[operations_iter2].result.sink_node->getName() + "/" + std::to_string(operations_iter2);
				if (identified_sink_nodes.count(sink_node_key2)) continue; // Skip identified sink nodes

				// check if the sink nodes are compatible
				std::string sink_ops_key_1 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].result.time_step,
					FP_operations[operations_iter1].result.sink_node->getType(),
					FP_operations[operations_iter1].result.sink_node->getIntegration()->getName(),
					FP_operations[operations_iter1].result.sink_node->getActivation()->getName(),
					FP_operations[operations_iter1].result.sink_node->getLayerName(),
					FP_operations[operations_iter1].result.sink_node->getTensorIndex().first,
					FP_operations[operations_iter1].arguments[0].weight->getLayerName());
				std::string sink_ops_key_2 = makeForwardPropogationOperationsKey(FP_operations[operations_iter2].result.time_step,
					FP_operations[operations_iter2].result.sink_node->getType(),
					FP_operations[operations_iter2].result.sink_node->getIntegration()->getName(),
					FP_operations[operations_iter2].result.sink_node->getActivation()->getName(),
					FP_operations[operations_iter2].result.sink_node->getLayerName(),
					FP_operations[operations_iter2].result.sink_node->getTensorIndex().first,
					FP_operations[operations_iter2].arguments[0].weight->getLayerName());
				if (sink_ops_key_1 != sink_ops_key_2) continue;

				// check if the source nodes are compatible
				std::set<std::string> argument1_nodes, argument2_nodes;
				for (const auto& argument : FP_operations[operations_iter1].arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.source_node->getType(),
						argument.source_node->getIntegration()->getName(),
						argument.source_node->getActivation()->getName(),
						argument.source_node->getLayerName(),
						argument.source_node->getTensorIndex().first,
						argument.weight->getLayerName());
					argument1_nodes.insert(ops_key);
				}
				for (const auto& argument : FP_operations[operations_iter2].arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.source_node->getType(),
						argument.source_node->getIntegration()->getName(),
						argument.source_node->getActivation()->getName(),
						argument.source_node->getLayerName(),
						argument.source_node->getTensorIndex().first,
						argument.weight->getLayerName());
					argument2_nodes.insert(ops_key);
				}
				if (argument1_nodes != argument2_nodes ) continue;

				// Check compatibility with future operations
				if (!fast_check) {
					std::set<std::string> sinkAsSourceNode_1, sinkAsSourceNode_2, sourceAsSourceNode_1, sourceAsSourceNode_2,
						sinkAsSinkNode_1, sinkAsSinkNode_2, sinkToSinkNode_1, sinkToSinkNode_2;
					for (size_t operations_iter3 = operations_iter1 + 1; operations_iter3 < FP_operations.size(); ++operations_iter3) {
						std::string sink_node_key3 = FP_operations[operations_iter3].result.sink_node->getName() + "/" + std::to_string(operations_iter3);
						if (identified_sink_nodes.count(sink_node_key3) || operations_iter3 == operations_iter2) continue; // Skip current and identified sink nodes
						std::string sink_ops_key_3 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].result.time_step,
							FP_operations[operations_iter3].result.sink_node->getType(),
							FP_operations[operations_iter3].result.sink_node->getIntegration()->getName(),
							FP_operations[operations_iter3].result.sink_node->getActivation()->getName(),
							FP_operations[operations_iter3].result.sink_node->getLayerName(),
							FP_operations[operations_iter3].result.sink_node->getTensorIndex().first,
							FP_operations[operations_iter3].arguments[0].weight->getLayerName());

						for (auto& argument : FP_operations[operations_iter3].arguments) {
							// Check if sink nodes will be compatible as future source nodes
							std::string source_ops_key_1 = makeForwardPropogationOperationsKey(argument.time_step,
								argument.source_node->getType(),
								argument.source_node->getIntegration()->getName(),
								argument.source_node->getActivation()->getName(),
								argument.source_node->getLayerName(),
								argument.source_node->getTensorIndex().first,
								argument.weight->getLayerName());
							if (source_ops_key_1 == sink_ops_key_1
								&& argument.source_node->getName() == FP_operations[operations_iter1].result.sink_node->getName()
								) {
								sinkAsSourceNode_1.insert(FP_operations[operations_iter3].result.sink_node->getName());
							}
							if (source_ops_key_1 == sink_ops_key_2
								&& argument.source_node->getName() == FP_operations[operations_iter2].result.sink_node->getName()
								) {
								sinkAsSourceNode_2.insert(FP_operations[operations_iter3].result.sink_node->getName());
							}

							// Check if the sink nodes target sink layer is compatible
							if (sink_ops_key_3 != sink_ops_key_1
								&& argument.source_node->getName() == FP_operations[operations_iter1].result.sink_node->getName()
								) {
								sinkToSinkNode_1.insert(FP_operations[operations_iter3].result.sink_node->getName());
							}
							if (sink_ops_key_3 != sink_ops_key_2
								&& argument.source_node->getName() == FP_operations[operations_iter2].result.sink_node->getName()
								) {
								sinkToSinkNode_2.insert(FP_operations[operations_iter3].result.sink_node->getName());
							}

							// Check if the source nodes will be compatible as future source nodes
							for (const auto& argument1 : FP_operations[operations_iter1].arguments) {
								std::string ops_key = makeForwardPropogationOperationsKey(argument1.time_step,
									argument1.source_node->getType(),
									argument1.source_node->getIntegration()->getName(),
									argument1.source_node->getActivation()->getName(),
									argument1.source_node->getLayerName(),
									argument1.source_node->getTensorIndex().first,
									argument1.weight->getLayerName());
								if (source_ops_key_1 == ops_key
									&& argument.source_node->getName() == argument1.source_node->getName()
									) {
									sourceAsSourceNode_1.insert(FP_operations[operations_iter3].result.sink_node->getName());
								}
								// Also check if the sink nodes will be compatible with future source nodes
								if (source_ops_key_1 != ops_key //&& sink_ops_key_3 == sink_ops_key_1
									&& FP_operations[operations_iter3].result.sink_node->getName() == FP_operations[operations_iter1].result.sink_node->getName()
									) {
									sinkAsSinkNode_1.insert(argument.source_node->getName());
								}
							}
							for (const auto& argument2 : FP_operations[operations_iter2].arguments) {
								std::string ops_key = makeForwardPropogationOperationsKey(argument2.time_step,
									argument2.source_node->getType(),
									argument2.source_node->getIntegration()->getName(),
									argument2.source_node->getActivation()->getName(),
									argument2.source_node->getLayerName(),
									argument2.source_node->getTensorIndex().first,
									argument2.weight->getLayerName());
								if (source_ops_key_1 == ops_key
									&& argument.source_node->getName() == argument2.source_node->getName()
									) {
									sourceAsSourceNode_2.insert(FP_operations[operations_iter3].result.sink_node->getName());
								}
								// Also check if the sink nodes will be compatible with future source nodes
								if (source_ops_key_1 != ops_key //&& sink_ops_key_3 == sink_ops_key_2
									&& FP_operations[operations_iter3].result.sink_node->getName() == FP_operations[operations_iter2].result.sink_node->getName()
									) {
									sinkAsSinkNode_2.insert(argument.source_node->getName());
								}
							}
						}
					}
					if (sinkAsSourceNode_1 != sinkAsSourceNode_2 || sourceAsSourceNode_1 != sourceAsSourceNode_2
						|| sinkAsSinkNode_1 != sinkAsSinkNode_2 || sinkToSinkNode_1 != sinkToSinkNode_2) continue;
				}

				// update the maps
				identified_sink_nodes.insert(sink_node_key1);
				identified_sink_nodes.insert(sink_node_key2);
				std::vector<int> first_operation = { (int)operations_iter1 };
				auto found = FC_layers.emplace(sink_node_key1, first_operation);
				FC_layers.at(sink_node_key1).push_back(operations_iter2);
			}

			// Check if compatible operations were found, if not add as is
			if (identified_sink_nodes.count(sink_node_key1) == 0) {
				identified_sink_nodes.insert(sink_node_key1);
				std::vector<int> first_operation = { (int)operations_iter1 };
				auto found = FC_layers.emplace(sink_node_key1, first_operation);
			}
		}
		return FC_layers;
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::getForwardPropogationLayerTensorDimensions(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes, std::vector<std::vector<std::pair<int, int>>>& weight_indices, 
		std::vector<std::map<std::string, std::vector<std::pair<int, int>>>>& shared_weight_indices, std::vector<std::vector<TensorT>>& weight_values,
		std::vector<bool>& make_source_tensors, std::vector<bool>& make_sink_tensors, std::vector<bool>& make_weight_tensors) {
		// track the layer_tensor positions for the source and sink nodes
		// as well as the weight_tensor positions
		int sink_layer_pos = layer_tensors_.size();
		int source_layer_pos = layer_tensors_.size() + 1;
		int weight_pos = weight_tensors_.size();

		for (const auto& operations : operations_map) {
			// determine the tensor sizes
			int sink_layer_size = 0;
			int source_layer_size = 0;
			std::vector<std::pair<int, int>> weight_index;
			std::map<std::string, std::vector<std::pair<int, int>>> shared_weight_index;
			std::vector<TensorT> weight_value;
			bool make_sink_tensor = false;
			bool make_source_tensor = false;
			bool make_weight_tensor = false;

			// inernal variables to track changes in source/sink layer positions
			bool updated_source_layer_pos = false;

			for (const int& ops_index : operations.second) {
				// index sink node tensors (if it does not yet exist)
				int sink_layer_index;
				bool increment_sink_layer_size = false;
				if (FP_operations[ops_index].result.sink_node->getTensorIndex().first == -1) {
					FP_operations[ops_index].result.sink_node->setTensorIndex(std::make_pair(sink_layer_pos, sink_layer_size));
					sink_layer_index = sink_layer_size;
					make_sink_tensor = true;
					increment_sink_layer_size = true;
				}
				else {
					sink_layer_index = FP_operations[ops_index].result.sink_node->getTensorIndex().second;
				}

				if (!updated_source_layer_pos && !make_sink_tensor) {
					source_layer_pos = sink_layer_pos;
					updated_source_layer_pos = true;
				}

				// index source node tensor (if it does not yet exist)
				for (const OperationArguments<TensorT>& argument : FP_operations[ops_index].arguments) {
					int source_layer_index;
					bool increment_source_layer_size = false;
					if (argument.source_node->getTensorIndex().first == -1) {
						argument.source_node->setTensorIndex(std::make_pair(source_layer_pos, source_layer_size));
						source_layer_index = source_layer_size;
						make_source_tensor = true;
						increment_source_layer_size = true;
					}
					else {
						source_layer_index = argument.source_node->getTensorIndex().second;
					}

					// index weight tensors
					if (argument.weight->getTensorIndex().size() == 0) {
						argument.weight->addTensorIndex(std::make_tuple(weight_pos, source_layer_index, sink_layer_index));
						weight_index.push_back(std::make_pair(source_layer_index, sink_layer_index));
						if (argument.weight->getInitWeight()) {
							TensorT tmp = argument.weight->getWeightInitOp()->operator()();
							weight_value.push_back(tmp);
							argument.weight->setWeight(tmp);
							argument.weight->setInitWeight(false); // ensures that from now on the weight will not be re-initialized
						}
						else {
							weight_value.push_back(argument.weight->getWeight());
						}
						make_weight_tensor = true;
					}
					else {
						argument.weight->addTensorIndex(std::make_tuple(weight_pos, source_layer_index, sink_layer_index));
						weight_index.push_back(std::make_pair(source_layer_index, sink_layer_index));
						weight_value.push_back(argument.weight->getWeight());
						make_weight_tensor = true;  // even if the weights are shared, we should still make a new weight tensor
						std::vector<std::pair<int, int>> tmp = { std::make_pair(source_layer_index, sink_layer_index) };
						auto found = shared_weight_index.emplace(argument.weight->getName(), tmp);
						if (!found.second) {
							// add the new shared weight index
							shared_weight_index.at(argument.weight->getName()).push_back(std::make_pair(source_layer_index, sink_layer_index));
						}
						else {
							// add the first shared weight index
							int weight_pos_0 = std::get<0>(argument.weight->getTensorIndex()[0]);
							if (weight_pos_0 != weight_pos) {
								char error_char[512];
								sprintf(error_char, "The weight is shared across multiple tensors.  This is currently not supported.");
								std::string error(error_char);
								throw std::runtime_error(error_char);
								// if this fails, then the weight is shared with another layer.
								// the current weight sharing implementation cannot handle such cases.
							}
							int source_layer_index_0 = std::get<1>(argument.weight->getTensorIndex()[0]);
							int sink_layer_index_0 = std::get<2>(argument.weight->getTensorIndex()[0]);
							shared_weight_index.at(argument.weight->getName()).push_back(std::make_pair(source_layer_index_0, sink_layer_index_0));
						}
					}
					if (increment_source_layer_size) ++source_layer_size;
				}
				if (increment_sink_layer_size) ++sink_layer_size; //?
			}

			// determine the actual source and sink layer sizes
			std::set<int> source_nodes, sink_nodes;
			for (const std::pair<int, int>& p : weight_index) {
				source_nodes.insert(p.first);
				sink_nodes.insert(p.second);
			}

			//// check the source and sink layer indices
      //// NOTE: this check should no longer be needed...
			//for (const std::pair<int, int>& p : weight_index) {
			//	if (p.first >= source_nodes.size() || p.second >= sink_nodes.size()) {
			//		throw std::out_of_range("Weight index is greater than the layer size.");
			//		// Error is caused by incorrectly allocating nodes and weights to different tensors
			//	}
			//}

			// store the tensor sizes
			//sink_layer_sizes.push_back(sink_layer_size); // This is not accurate because we are actually tracking the next sink_layer position...
			//source_layer_sizes.push_back(source_layer_size); // This is not accurate because we are actually tracking the next source_layer position...
			//sink_layer_sizes.push_back(sink_nodes.size());  // This is also not accurate because the sink node size may be less than the maximum sink index
			//source_layer_sizes.push_back(source_nodes.size());  // This is also not accurate because the source node size may be less than the maximum source index
			sink_layer_sizes.push_back(*std::max_element(sink_nodes.begin(), sink_nodes.end()) + 1);
			source_layer_sizes.push_back(*std::max_element(source_nodes.begin(), source_nodes.end()) + 1);
			make_source_tensors.push_back(make_source_tensor);
			make_sink_tensors.push_back(make_sink_tensor);
			make_weight_tensors.push_back(make_weight_tensor);
			weight_indices.push_back(weight_index);
			weight_values.push_back(weight_value);
			shared_weight_indices.push_back(shared_weight_index);

      // Check that source layer size is not less than the # of source nodes
      if (source_layer_sizes.back() < source_nodes.size()) {
        char error_char[512];
        sprintf(error_char, "Attempting to join multiple source nodes into a single layer that were previously split into seperate layers.");
        std::string error(error_char);
        throw std::runtime_error(error_char);
      }

			// update the layer positions
			if (make_sink_tensor && make_source_tensor) {
				sink_layer_pos += 2;
				source_layer_pos = sink_layer_pos + 1;
			}
			else if (make_sink_tensor || make_source_tensor) {
				sink_layer_pos += 1;
				source_layer_pos = sink_layer_pos + 1;
			}

			// update the weight positions
			if (make_weight_tensor) {
				weight_pos += 1;
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	std::string ModelInterpreter<TensorT, DeviceT>::makeForwardPropogationOperationsKey(
		const int & time_step, const NodeType& node_type, const std::string & node_integration, const std::string & node_activation, 
		const std::string& node_layer_name, const int& node_layer_index, const std::string& weight_layer_name)
	{
		// [TODO: may not need to add in node type
		//std::string ops_key = std::to_string(time_step) + "/" + std::to_string(node_type) + "/" + node_integration + "/" + node_activation;
		std::string ops_key = std::to_string(time_step) + "/" + node_integration + "/" + node_activation + "/" + node_layer_name + "/" + weight_layer_name;// +"/" + std::to_string(layer_index);
		return ops_key;
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getForwardPropogationOperations(Model<TensorT>& model, const int& batch_size, const int& memory_size, 
		const bool& train, const bool& fast_check, const bool& find_cycles, const bool& preserve_OoO)
	{
		// register the batch and memory sizes with the model
		// [TODO: add tests]
		model.setBatchAndMemorySizes(batch_size, memory_size);

		// buffer the memory size
		const int memory_size_buffered = memory_size + 1;

		// Get the forward operation steps
		if (tensor_ops_steps_.size() == 0) {

			// compile the model into a list of operations
			int iter = 0;
			std::vector<OperationList<TensorT>> FP_operations_expanded;
			if (preserve_OoO) 
				getFPOpsOoO_(model, find_cycles, FP_operations_expanded, iter); // TODO: remove `find_cycles`
			else
				getFPOpsGraph_(model, FP_operations_expanded, iter);

			// identify tensor operation motifs in the list of operations
			std::set<std::string> identified_sink_nodes;
			//std::map<std::string, std::vector<int>> custom_ops = getCustomOperations(FP_operations_expanded, identified_sink_nodes);
			std::map<std::string, std::vector<int>> SC_ops = GetSinglyConnectedOperations(FP_operations_expanded, identified_sink_nodes);

      // TODO:  add check to accept or reject SC the optimization based on the following criteria:
      // - Given a series of FanIn or or FanOut operations where the size of the Fan layer is N 
      //   and the number of unique target nodes in the series is M
      //   acceptance criteria is M * 2 < N
      //   e.g., SC_ops.size() * 2 < SC_ops.front().size() where all SC_ops target the same sink layer
      // - if (!SC_ops_acceptance) SC_ops.clear(); identified_sink_nodes.clear();
      //   assuming the only identified sink nodes are those from `GetSinglyConnectedOperations`

			//std::map<std::string, std::vector<int>> Conv_ops = getConvOperations(FP_operations_expanded, identified_sink_nodes);
			std::map<std::string, std::vector<int>> tensor_ops = getTensorOperations(FP_operations_expanded, identified_sink_nodes, fast_check);

			std::vector<std::map<std::string, std::vector<int>>> tensor_ops_steps;
			tensor_ops_steps.resize(iter);
      for (auto& tensor_op : SC_ops) {         
        FP_operations_expanded[tensor_op.second[0]].result.sink_node->setIntegration(
          std::shared_ptr<IntegrationOp<TensorT>>(new ProdSCOp<TensorT>())); // TODO/ASSUMPTION: update the integration to ProdSCOp since
                                                                             // ProdOp integration types are the only allowed SC motifes for now...
        tensor_ops_steps[FP_operations_expanded[tensor_op.second[0]].operation_index].emplace(tensor_op.first, tensor_op.second);
      }
			for (auto& tensor_op : tensor_ops) {
				tensor_ops_steps[FP_operations_expanded[tensor_op.second[0]].operation_index].emplace(tensor_op.first, tensor_op.second);
			}

			// Save the list of operations for fast model check-pointing
			tensor_ops_steps_ = tensor_ops_steps;
			FP_operations_ = FP_operations_expanded;

			// Allocate memory for tensors
			for (auto& tensor_ops_step: tensor_ops_steps) {
				//if (custom_ops.size() != 0) {
				//	std::vector<int> source_layer_sizes, sink_layer_sizes;
				//	std::vector<std::vector<TensorT>> weight_values;
				//	std::vector<std::vector<std::pair<int, int>>> weight_indices;
				//	std::vector<std::map<std::string, std::vector<std::pair<int, int>>>> shared_weight_indices;
				//	std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				//	getForwardPropogationLayerTensorDimensions(tensor_ops_step, custom_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);
				//	allocateForwardPropogationLayerTensors(tensor_ops_step, custom_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size_buffered, train);
				//}
				if (tensor_ops_step.size() != 0) {
					std::vector<int> source_layer_sizes, sink_layer_sizes;
					std::vector<std::vector<TensorT>> weight_values;
					std::vector<std::vector<std::pair<int, int>>> weight_indices;
					std::vector<std::map<std::string, std::vector<std::pair<int, int>>>> shared_weight_indices;
					std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
					getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops_step, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);
					allocateForwardPropogationLayerTensors(FP_operations_expanded, tensor_ops_step, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size_buffered, train);
				}
			}
		}
		// Work from the cache
		else {
			// Clear the tensor indices
			// NOTE: could be avoided by instead using the model directly
			//			to keep track of the tensor indices during `getForwardPropogationLayerDimensions`
			for (auto& FP_operation : FP_operations_) {
				FP_operation.result.sink_node->setTensorIndex(std::make_pair(-1, -1));
				for (auto& argument : FP_operation.arguments) {
					argument.source_node->setTensorIndex(std::make_pair(-1, -1));
					argument.weight->clearTensorIndex();
				}
			}
			for (auto& tensor_ops_step : tensor_ops_steps_) {
				//if (custom_ops.size() != 0) {
				//	std::vector<int> source_layer_sizes, sink_layer_sizes;
				//	std::vector<std::vector<TensorT>> weight_values;
				//	std::vector<std::vector<std::pair<int, int>>> weight_indices;
				//	std::vector<std::map<std::string, std::vector<std::pair<int, int>>>> shared_weight_indices;
				//	std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				//	getForwardPropogationLayerTensorDimensions(tensor_ops_step, custom_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);
				//	allocateForwardPropogationLayerTensors(tensor_ops_step, custom_ops, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size_buffered, train);
				//}
				if (tensor_ops_step.size() != 0) {
					std::vector<int> source_layer_sizes, sink_layer_sizes;
					std::vector<std::vector<TensorT>> weight_values;
					std::vector<std::vector<std::pair<int, int>>> weight_indices;
					std::vector<std::map<std::string, std::vector<std::pair<int, int>>>> shared_weight_indices;
					std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
					getForwardPropogationLayerTensorDimensions(FP_operations_, tensor_ops_step, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);
					allocateForwardPropogationLayerTensors(FP_operations_, tensor_ops_step, source_layer_sizes, sink_layer_sizes, weight_indices, shared_weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size_buffered, train);
				}
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::getFPOpsOoO_(Model<TensorT>& model, const bool& find_cycles, std::vector<OperationList<TensorT>>& FP_operations_expanded, int& iter)
	{
		FP_operations_expanded.clear();
		iter = 0;

		// STEP 1: Preliminaries...
		// initialize the node statuses to determine the FP propogation steps
		for (auto& nodes_map : model.nodes_) {
			if (nodes_map.second->getType() == NodeType::input || nodes_map.second->getType() == NodeType::bias)
				nodes_map.second->setStatus(NodeStatus::activated);
			else
				nodes_map.second->setStatus(NodeStatus::initialized);
		}

		// STEP 2: Get a list of unoptimized operations for FP
		const int max_iters = 1e6;
		std::vector<OperationList<TensorT>> FP_operations;
		for (; iter < max_iters; ++iter)
		{
      std::map<std::string, int> FP_operations_map;
      std::vector<OperationList<TensorT>> FP_operations_list;
			//if (find_cycles) { // TBD: if an optimization can be made to reduce the `NextActiveLayer` method to the minimal number of link iterations
      // get the next hidden layer
      getNextInactiveLayer(model, FP_operations_map, FP_operations_list);

			// get cycles
			std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
			std::vector<OperationList<TensorT>> FP_operations_list_cycles = FP_operations_list;
			std::vector<std::string> sink_nodes_cycles;
			getNextInactiveLayerCycles(model, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_cycles);

			// Remove all nodes involved in "cycles" that have arguments
			// involving source to sink node pairs not identified as cycles
			pruneInactiveLayerCycles(model, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_cycles);
			//}
      //else {
      //  // get the next hidden layer
      //  getNextInactiveLayerWOBiases(model, FP_operations_map, FP_operations_list);

      //  // get biases
      //  std::vector<std::string> sink_nodes_with_biases;
      //  getNextInactiveLayerBiases(model, FP_operations_map, FP_operations_list, sink_nodes_with_biases);
      //}

			// check if all nodes have been activated
			if (FP_operations_list.size() == 0) {
				break;
			}

			// activate sink nodes and update the Operations index
			for (auto& FP_operation : FP_operations_list) {
				FP_operation.result.sink_node->setStatus(NodeStatus::activated);
				FP_operation.operation_index = iter;
				FP_operations.push_back(FP_operation);
			}
		}

		// STEP 3: organize the operations into Tensor layers for hardware acceleration
		// Pre-emptively Expand the set of operations
		//expandForwardPropogationOperations(FP_operations, FP_operations_expanded); // Slower and not needed...
		expandAllForwardPropogationOperations(FP_operations, FP_operations_expanded);
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::getFPOpsGraph_(Model<TensorT>& model, std::vector<OperationList<TensorT>>& FP_operations_expanded, int & iter)
	{
		FP_operations_expanded.clear();

		// get all operations in the graph
		for (auto& link_map : model.links_)
		{
			// arguments
			// NOTE: each link is given it's own OperationList to avoid the eventual split downstream
			OperationArguments<TensorT> arguments;
			arguments.source_node = model.nodes_.at(link_map.second->getSourceNodeName());
			arguments.weight = model.weights_.at(link_map.second->getWeightName());
			arguments.time_step = 1;
			arguments.link_name = link_map.first;

			// results
			OperationList<TensorT> operation_list;
			OperationResult<TensorT> result;
			result.sink_node = model.nodes_.at(link_map.second->getSinkNodeName());
			result.time_step = 0;
			operation_list.result = result;
			operation_list.arguments.push_back(arguments);
			operation_list.operation_index = 0;
			FP_operations_expanded.push_back(operation_list);
		}

		iter = 1;
	}
	
	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::addLayerTensor(NodeTensorData<TensorT, DeviceT>& layer)
	{
		std::shared_ptr<NodeTensorData<TensorT, DeviceT>> layer_ptr(std::move(&layer));
		layer_tensors_.push_back(layer_ptr);
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::clearLayerTensors()
	{
		layer_tensors_.clear();
	}

	template<typename TensorT, typename DeviceT>
	inline std::shared_ptr<NodeTensorData<TensorT, DeviceT>> ModelInterpreter<TensorT, DeviceT>::getLayerTensor(const int & layer_index)
	{
		try { return layer_tensors_.at(layer_index); }
		catch (const std::exception& e) {
			std::cout << "Layer index " << layer_index << " does not exist" << std::endl;
			return std::shared_ptr<NodeTensorData<TensorT, DeviceT>>();
		}		
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::addWeightTensor(WeightTensorData<TensorT, DeviceT>& weight)
	{
		std::shared_ptr<WeightTensorData<TensorT, DeviceT>> weight_ptr(std::move(&weight));
		weight_tensors_.push_back(weight_ptr);
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::clearWeightTensors()
	{
		weight_tensors_.clear();
	}

	template<typename TensorT, typename DeviceT>
	inline std::shared_ptr<WeightTensorData<TensorT, DeviceT>> ModelInterpreter<TensorT, DeviceT>::getWeightTensor(const int & weight_index)
	{
		return weight_tensors_.at(weight_index);
	}

	template<typename TensorT, typename DeviceT>
	inline std::shared_ptr<ModelErrorData<TensorT, DeviceT>> ModelInterpreter<TensorT, DeviceT>::getModelError()
	{
		return model_error_;
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::addOperationSteps(const std::vector<OperationTensorStep<TensorT, DeviceT>>& operation_steps) {
		operation_steps_.push_back(operation_steps);
	}

	template<typename TensorT, typename DeviceT>
	inline std::vector<OperationTensorStep<TensorT, DeviceT>> ModelInterpreter<TensorT, DeviceT>::getOperationSteps(const int& operation_index) {
		return operation_steps_.at(operation_index);
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::clearOperationSteps()
	{
		operation_steps_.clear();
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::FPTT(const int& time_steps)
	{
		// check time_steps vs memory_size
		int max_steps = time_steps;
		if (time_steps >= layer_tensors_[0]->getMemorySize())
		{
			std::cout << "Time_steps will be scaled back to the memory_size - 1." << std::endl;
			max_steps = layer_tensors_[0]->getMemorySize() - 1;
		}

		for (int time_step = 0; time_step < max_steps; ++time_step)		{
			const int time_step_cur = max_steps - 1 - time_step;
			executeForwardPropogationOperations(time_step_cur);
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::CETT(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& values, const std::vector<std::string>& node_names, 
		LossFunctionOp<TensorT>* loss_function, LossFunctionGradOp<TensorT>* loss_function_grad, const int & time_steps)
	{
		// check time_steps vs memory_size
		// [NOTE: was changed form memory_size to memory_size - 1]
		int max_steps = time_steps;
		if (time_steps >= layer_tensors_[0]->getMemorySize())
		{
			std::cout << "Time_steps will be scaled back to the memory_size - 1." << std::endl;
			max_steps = layer_tensors_[0]->getMemorySize() - 1;
		}

		if (values.dimension(1) - 1 > layer_tensors_[0]->getMemorySize())
			std::cout << "The sequence for CETT needs to be the memory_size - 1!" << std::endl;

		// extract out the layer id
		const int layer_id = model.nodes_.at(node_names[0])->getTensorIndex().first;
		if (layer_id < 0) {
			char error_char[512];
			sprintf(error_char, "The output layer does not exist.");
			std::string error(error_char);
			throw std::runtime_error(error_char);
		}
		if (getLayerTensor(layer_id)->getLayerSize() != node_names.size()) {
			char error_char[512];
			sprintf(error_char, "The number of output nodes does not match the output layer tensor size.");
			std::string error(error_char);
			throw std::runtime_error(error_char);
		}

		// convert the loss function
		LossFunctionTensorOp<TensorT, DeviceT>* loss_function_tensor = nullptr;
		LossFunctionOpToLossFunctionTensorOp<TensorT, DeviceT> loss_conv;
		loss_conv(loss_function, loss_function_tensor, std::vector<TensorT>() = {});
		LossFunctionGradTensorOp<TensorT, DeviceT>* loss_function_grad_tensor = nullptr;
		LossFunctionGradOpToLossFunctionGradTensorOp<TensorT, DeviceT> loss_grad_conv;
		loss_grad_conv(loss_function_grad, loss_function_grad_tensor, std::vector<TensorT>() = {});

		// NOTE: the output are stored [Tmax, Tmax - 1, ..., T=0, T=-1] where T=-1 is added automatically
		//	     so the expected values should also be stored [Tmax, Tmax - 1, ..., T=0, T=-1]
		for (int time_step = 0; time_step < max_steps; ++time_step)
		{
			// calculate the error for each batch of memory
			Eigen::Tensor<TensorT, 2> expected = values.chip(time_step, 1);
		  executeModelErrorOperations(expected, layer_id, loss_function_tensor, loss_function_grad_tensor, time_step);
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::TBPTT(const int& time_steps)
	{
		// check time_steps vs memory_size
		int max_steps = time_steps;
		if (time_steps >= layer_tensors_[0]->getMemorySize())
		{
			std::cout << "Time_steps will be scaled back to the memory_size - 1." << std::endl;
			max_steps = layer_tensors_[0]->getMemorySize() - 1;
		}
		for (int time_step = 0; time_step < max_steps; ++time_step) {

			// calculate the error for each batch of memory
			executeBackwardPropogationOperations(time_step);
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::updateWeights()
	{
		executeWeightErrorOperations();
		executeWeightUpdateOperations();
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::setModelResources(const ModelResources & model_resources)
	{
		model_resources_ = model_resources;
	}

	template<typename TensorT, typename DeviceT>
	inline ModelResources ModelInterpreter<TensorT, DeviceT>::getModelResources()
	{
		return model_resources_;
	}
	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::clear_cache()
	{
		operation_steps_.clear();
		layer_tensors_.clear();
		weight_tensors_.clear();
		model_error_.reset();
		FP_operations_.clear();
		tensor_ops_steps_.clear();
	}
	template<typename TensorT, typename DeviceT>
	inline std::vector<std::map<std::string, std::vector<int>>> ModelInterpreter<TensorT, DeviceT>::getTensorOpsSteps() const {
		return tensor_ops_steps_;
	}
}
#endif //SMARTPEAK_MODELINTERPRETER_H