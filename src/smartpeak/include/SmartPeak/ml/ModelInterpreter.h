/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETER_H
#define SMARTPEAK_MODELINTERPRETER_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS

// .h
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/NodeMatrixData.h>
#include <SmartPeak/ml/WeightMatrixData.h>
#include <SmartPeak/ml/IntegrationFunction3.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <set>

// .cpp
#include <SmartPeak/ml/ModelErrorData.h>
#include <SmartPeak/ml/ModelKernal.h>

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
	};

	template<typename TensorT>
	struct OperationArguments
	{
		std::shared_ptr<Node<TensorT>> source_node;
		std::shared_ptr<Weight<TensorT>> weight;
		std::string link_name;
		int time_step = 0;
	};

	template<typename TensorT>
	struct OperationList
	{
		OperationResult<TensorT> result;
		std::vector<OperationArguments<TensorT>> arguments;
	};

	/*
	Structures required for layer operations
	*/
	template<typename TensorT, typename DeviceT>
	struct OperationLayer
	{
		std::shared_ptr<NodeMatrixData<TensorT>> sink_layer;
		int time_step = 0;
		std::shared_ptr<LayerIntegrationOp<TensorT, DeviceT>> sink_integration_function;
		std::shared_ptr<LayerIntegrationErrorOp<TensorT, DeviceT>> sink_integration_error;
		std::shared_ptr<LayerIntegrationWeightGradOp<TensorT, DeviceT>> sink_integration_weight_grad_error;
	};

	template<typename TensorT>
	struct OperationWeight
	{
		std::shared_ptr<WeightMatrixData<TensorT>> weight;
	};

	/*
	Class used for layer operations
	*/
	template<typename TensorT, typename DeviceT>
	class OperationTensorStep
	{
		OperationLayer<TensorT, DeviceT> sink_layer;
		OperationLayer<TensorT, DeviceT> source_layer;
		OperationWeight<TensorT> weight;
	};

	template<typename TensorT>
	class OperationTensorStepDefaultDevice : OperationTensorStep<TensorT, Eigen::DefaultDevice>
	{
		OperationLayer<TensorT, Eigen::DefaultDevice> sink_layer;
		OperationLayer<TensorT, Eigen::DefaultDevice> source_layer;
		OperationWeight<TensorT> weight;
	};

	template<typename TensorT>
	class OperationTensorStepCpu : OperationTensorStep<TensorT, Eigen::ThreadPool>
	{
		OperationLayer<TensorT, Eigen::ThreadPool> sink_layer;
		OperationLayer<TensorT, Eigen::ThreadPool> source_layer;
		OperationWeight<TensorT> weight;
	};

	template<typename TensorT>
	class OperationTensorStepGpu : OperationTensorStep<TensorT, Eigen::GpuDevice>
	{
		OperationLayer<TensorT, Eigen::GpuDevice> sink_layer;
		OperationLayer<TensorT, Eigen::GpuDevice> source_layer;
		OperationWeight<TensorT> weight;
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
		~ModelInterpreter() = default; ///< Default destructor

		inline bool operator==(const ModelInterpreter& other) const
		{
			return
				std::tie(
					operations_cache_
				) == std::tie(
					other.operations_cache_
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
			operations_cache_ = other.operations_cache_;
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
			@param[in] status_update
			@param[in] value_type ("output", "derivative", "error", or "dt")
		*/
		void mapValuesToLayers(
			const Eigen::Tensor<TensorT, 3>& values,
			const std::vector<std::string>& node_names,
			const NodeStatus& status_update,
			const std::string& value_type);

		/**
			@brief A prelude to a forward propogation step. Returns a vector of links
				and associated nodes that satisfy the following conditions:
				1. all sink output values are unknown (i.e. inactive),
				2. all source node output values are known (i.e. active).
				3. all nodes need not be the same type

			@param[out] FP_operations_map Key/Value pair of sink node name to FP_operations index
			@param[out] FP_operations
		*/
		void getNextInactiveLayer(
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
		void getNextInactiveLayerBiases(
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
		void getNextInactiveLayerCycles(
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
		void pruneInactiveLayerCycles(
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

		/**
			@brief Re-organizes the identified layers into tensors and attempts to optimizes
				the layer operations to maximize hardware acceleration.

			Layer operations will be partitioned into predefined tensor integration motifs
			- Tensor integration motifs: FC/SC/Conv/FanIn/FanOut
			- Node integration types: Sum/Prod/Max/Mean/Var/Count
			- Custom

			Criteria for FC
			- all arguments for sinks are equal

			Criteria for SC
			- unique argument per sinks

			Criteria for Conv/pool
			- shared weights with FanIn

			Criteria for FanIn and FanOut
			- FanIn: 1 sink, multiple sources
			- FanOut: 1 source, multiple sinks

			Critera for Custom
			- Module with optimized computation (e.g., softmax, attention, etc.,)

			[TODO: add tests!]

			@param[in, out] FP_operations
		*/
		void optimizeForwardPropogationLayers(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& FC_ops,
			const std::map<std::string, std::vector<int>>& SC_ops,
			const std::map<std::string, std::vector<int>>& Conv_ops,
			const std::map<std::string, std::vector<int>>& custom_ops,
			const std::map<std::string, std::vector<int>>& fanOut_ops,
			const std::map<std::string, std::vector<int>>& fanIn_ops);

		/**
			@brief Identify layer operations

			[TODO: add tests]

			@param[in] FP_operations
			@param[in] identified_sink_nodes Set of identified sink nodes

			@returns map of identified operations consisting of the identifying sink node name or module name
				for the operation and a list of indices corresponding to the operations in FP_operations
		*/
		std::map<std::string, std::vector<int>> getCustomOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		std::map<std::string, std::vector<int>> getFullyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		std::map<std::string, std::vector<int>> GetSinglyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		std::map<std::string, std::vector<int>> getConvOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		std::map<std::string, std::vector<int>> getFanOutOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);
		std::map<std::string, std::vector<int>> getFanInOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);

		/**
		@brief Allocate Node and Weight tensor memory for all model operations.
			Node and weight tensors indices are registered.

		@param[in] FP_operations
		@param[in] ...
		*/
		void getForwardPropogationLayerTensorDimensions(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes, 
			std::vector<bool>& make_source_tensor, std::vector<bool>& make_sink_tensor, std::vector<bool>& make_weight_tensor);

		static std::string makeForwardPropogationOperationsKey(const int& time_step, const NodeType& node_type,
			const std::string& node_integration, const std::string& node_activation);

		void getForwardPropogationOperations();

		/**
		@brief Allocate Node and Weight tensor memory for all model operations.
			Weight solver params tensors are created using the first weight in the layer.
			Weight matrix is initialized using the first weight in the layer

		@param[in] FP_operations
		@param[in] ...
		*/
		virtual void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes,
			std::vector<bool>& make_source_tensor, std::vector<bool>& make_sink_tensor, std::vector<bool>& make_weight_tensor) = 0;
		virtual void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeModelErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		
		void clearCache(); ///< clear the FP and BP caches

	private:
		std::vector<std::vector<OperationTensorStep<TensorT>>> operations_cache_;
	};

	template<typename TensorT, typename DeviceT>
	ModelInterpreter<TensorT>::ModelInterpreter(const ModelInterpreter<TensorT>& other)
	{
		operations_cache_ = other.operations_cache_;
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT>::getNextInactiveLayer(
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations)
	{

		// get all links where the source node is active and the sink node is inactive
		// except for biases
		for (auto& link_map : links_)
		{
			if (nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias &&
				nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = nodes_.at(link_map.second->getSourceNodeName());
				arguments.weight = weights_.at(link_map.second->getWeightName());
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
					result.sink_node = nodes_.at(link_map.second->getSinkNodeName());
					operation_list.result = result;
					operation_list.arguments.push_back(arguments);
					FP_operations.push_back(operation_list);
				}
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT>::getNextInactiveLayerBiases(
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations,
		std::vector<std::string>& sink_nodes_with_biases)
	{

		// get all the biases for the sink nodes
		for (auto& link_map : links_)
		{
			std::string ops_key = link_map.second->getSinkNodeName();
			if (
				// does not allow for cycles
				nodes_.at(link_map.second->getSourceNodeName())->getType() == NodeType::bias &&
				nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				// required regardless if cycles are or are not allowed
				nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
				FP_operations_map.count(ops_key) != 0 // sink node has already been identified
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = nodes_.at(link_map.second->getSourceNodeName());
				arguments.weight = weights_.at(link_map.second->getWeightName());
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
	void ModelInterpreter<TensorT>::getNextInactiveLayerCycles(
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations,
		std::vector<std::string>& sink_nodes_with_cycles)
	{

		// get cyclic source nodes
		for (auto& link_map : links_)
		{
			std::string ops_key = link_map.second->getSinkNodeName();
			if (
				nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::initialized &&
				// required regardless if cycles are or are not allowed
				nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
				FP_operations_map.count(ops_key) != 0 // sink node has already been identified
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = nodes_.at(link_map.second->getSourceNodeName());
				arguments.weight = weights_.at(link_map.second->getWeightName());

				arguments.time_step = 1;
				arguments.link_name = link_map.first;
				FP_operations[FP_operations_map.at(ops_key)].arguments.push_back(arguments);
				sink_nodes_with_cycles.push_back(ops_key);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT>::pruneInactiveLayerCycles(std::map<std::string, int>& FP_operations_map, std::map<std::string, int>& FP_operations_map_cycles, std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_cycles, std::vector<std::string>& sink_nodes_with_cycles)
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
					for (const auto& cyclic_pair : cyclic_pairs_) {
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
	inline void ModelInterpreter<TensorT>::expandForwardPropogationOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		std::vector<OperationList<TensorT>> FP_operations_1, FP_operations_2;
		expandForwardPropogationOperationsBySourceNodeKey(FP_operations, FP_operations_1); // Pass 1:		 
		expandForwardPropogationOperationsByWeightKey(FP_operations_1, FP_operations_2); // Pass 2:		
		expandForwardPropogationOperationsByCachedNodes(FP_operations_2, FP_operations_expanded); // Pass 3: 
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT>::expandForwardPropogationOperationsBySourceNodeKey(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		FP_operations_expanded.clear();
		for (const OperationList<TensorT>& FP_operation : FP_operations) {
			// check that all arguments have the same time-step/activation/node_integration
			std::set<std::string> unique_node_types;
			for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
				std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
					argument.sink_node->getType(),
					argument.sink_node->getIntegration()->getName(),
					argument.sink_node->getActivation()->getName());
				unique_node_types.insert(ops_key);
			}
			for (const std::string& node_types : unique_node_types) {
				OperationList<TensorT> operations_list;
				operations_list.result = FP_operation.result;
				for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.sink_node->getType(),
						argument.sink_node->getIntegration()->getName(),
						argument.sink_node->getActivation()->getName());
					if (node_types == ops_key) {
						operations_list.arguments.push_back(argument);
					}
				}
				FP_operations_expanded.push_back(operations_list);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT>::expandForwardPropogationOperationsByWeightKey(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		FP_operations_expanded.clear();
		for (const OperationList<TensorT>& FP_operation : FP_operations_1) {
			// check that all links have the same solver/weight_init operator
			std::set<std::string> unique_weight_types;
			for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
				// Does not account for different solver parameters and weight init op parameters!
				std::string ops_key = argument.weight->getSolverOp()->getName() + "/" + argument.weight->getWeightInitOp()->getName();
				unique_weight_types.insert(ops_key);
			}
			for (const std::string& weight_types : unique_weight_types) {
				OperationList<TensorT> operations_list;
				operations_list.result = FP_operation.result;
				for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
					std::string ops_key = argument.weight->getSolverOp()->getName() + "/" + argument.weight->getWeightInitOp()->getName();
					if (weight_types == ops_key) {
						operations_list.arguments.push_back(argument);
					}
				}
				FP_operations_expanded.push_back(operations_list);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT>::expandForwardPropogationOperationsByCachedNodes(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
	{
		FP_operations_expanded.clear();
		for (const OperationList<TensorT>& FP_operation : FP_operations_2) {
			// check that all nodes are either cached or not yet cached into a layer
			OperationList<TensorT> operations_list_cached, operations_list;
			operations_list.result = FP_operation.result;
			operations_list_cached.result = FP_operation.result;
			for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
				if (argument.source_node->getLayerID().first == -1) {
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT>::getCustomOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		std::set<std::string> supported_custom_module_names = { "SoftMax" }; // [TODO: add support for ModuleType]
		std::map<std::string, std::vector<int>> custom_layers;
		for (size_t operations_iter = 0; operations_iter < FP_operations.size(); ++operations_iter) {
			if (identified_sink_nodes.count(FP_operations[operations_iter].result.sink_node->getName())) continue; // Skip identified sink nodes
			if (supported_custom_module_names.count(FP_operations[operations_iter].result.sink_node->getModuleName())) { // [TODO: replace with comparison after adding support for module types]
				std::string sink_node_key = FP_operations[operations_iter1].result.sink_node->getName() + std::string(operations_iter1);
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT>::getFullyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		std::map<std::string, std::vector<int>> FC_layers;
		for (size_t operations_iter1 = 0; operations_iter1 < FP_operations.size(); ++operations_iter1) {
			if (identified_sink_nodes.count(FP_operations[operations_iter1].result.sink_node->getName())) continue; // Skip identified sink nodes
			for (size_t operations_iter2 = operations_iter1 + 1; operations_iter2 < FP_operations.size(); ++operations_iter2) {
				if (identified_sink_nodes.count(FP_operations[operations_iter2].result.sink_node->getName())) continue; // Skip identified sink nodes

				// check if the sink nodes are compatible
				std::string ops_key_1 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].result.time_step,
					FP_operations[operations_iter1].result.sink_node->getType(),
					FP_operations[operations_iter1].result.sink_node->getIntegration()->getName(),
					FP_operations[operations_iter1].result.sink_node->getActivation()->getName());
				std::string ops_key_2 = makeForwardPropogationOperationsKey(FP_operations[operations_iter2].result.time_step,
					FP_operations[operations_iter2].result.sink_node->getType(),
					FP_operations[operations_iter2].result.sink_node->getIntegration()->getName(),
					FP_operations[operations_iter2].result.sink_node->getActivation()->getName());
				if (ops_key_1 != ops_key_2) continue;

				// check if the node names are all the same and compatible
				std::set<std::string> argument_nodes;
				for (const auto& argument : FP_operations[operations_iter1].arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.sink_node->getType(),
						argument.sink_node->getIntegration()->getName(),
						argument.sink_node->getActivation()->getName());
					std::string ops_key_id = argument.sink_node->getNodeName() + "/" + ops_key;
					argument_nodes.insert(ops_key_id);
				}
				for (const auto& argument : FP_operations[operations_iter2].arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.sink_node->getType(),
						argument.sink_node->getIntegration()->getName(),
						argument.sink_node->getActivation()->getName());
					std::string ops_key_id = argument.sink_node->getNodeName() + "/" + ops_key;
					argument_nodes.insert(ops_key_id);
				}
				if (argument_nodes.size() != FP_operations[operations_iter1].arguments.size() || argument_nodes.size() != FP_operations[operations_iter2].arguments.size()) continue;

				// update the maps
				std::string sink_node_key = FP_operations[operations_iter1].result.sink_node->getName() + std::string(operations_iter1);
				identified_sink_nodes.insert(sink_node_key);
				auto found = FC_layers.emplace(sink_node_key, std::vector<int>({ operations_iter1 }));
				FC_layers.at(sink_node_key).push_back(operations_iter2);
			}
		}
		return FC_layers;
	}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT>::GetSinglyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		std::map<std::string, std::vector<int>> SC_layers
			for (size_t operations_iter1 = 0; operations_iter1 < FP_operations.size(); ++operations_iter1) {
				if (identified_sink_nodes.count(FP_operations[operations_iter1].result.sink_node->getName())) continue; // Skip identified sink nodes
				if (FP_operations[operations_iter1].arguments.size() != 1) continue; // Not singly connected
				for (size_t operations_iter2 = operations_iter1 + 1; operations_iter2 < FP_operations.size(); ++operations_iter2) {
					if (identified_sink_nodes.count(FP_operations[operations_iter2].result.sink_node->getName())) continue; // Skip identified sink nodes
					if (FP_operations[operations_iter2].arguments.size() != 1) continue; // Not singly connected

					// check if the sink nodes are compatible
					std::string ops_key_1 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].result.time_step,
						FP_operations[operations_iter1].result.sink_node->getType(),
						FP_operations[operations_iter1].result.sink_node->getIntegration()->getName(),
						FP_operations[operations_iter1].result.sink_node->getActivation()->getName());
					std::string ops_key_2 = makeForwardPropogationOperationsKey(FP_operations[operations_iter2].result.time_step,
						FP_operations[operations_iter2].result.sink_node->getType(),
						FP_operations[operations_iter2].result.sink_node->getIntegration()->getName(),
						FP_operations[operations_iter2].result.sink_node->getActivation()->getName());
					if (ops_key_1 != ops_key_2) continue;

					// check if the source nodes are compatible
					ops_key_1 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].arguments[0].time_step,
						FP_operations[operations_iter1].arguments[0].source_node->getType(),
						FP_operations[operations_iter1].arguments[0].source_node->getIntegration()->getName(),
						FP_operations[operations_iter1].arguments[0].source_node->getActivation()->getName());
					ops_key_2 = makeForwardPropogationOperationsKey(FP_operations[operations_iter2].arguments[0].time_step,
						FP_operations[operations_iter2].arguments[0].source_node->getType(),
						FP_operations[operations_iter2].arguments[0].source_node->getIntegration()->getName(),
						FP_operations[operations_iter2].arguments[0].source_node->getActivation()->getName());
					if (ops_key_1 != ops_key_2) continue;

					// update the maps
					std::string sink_node_key = FP_operations[operations_iter1].result.sink_node->getName() + std::string(operations_iter1);
					identified_sink_nodes.insert(sink_node_key);
					auto found = SC_layers.emplace(sink_node_key, std::vector<int>({ operations_iter1 }));
					SC_layers.at(sink_node_key).push_back(operations_iter2);
				}
			}
		return SC_layers;
	}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT>::getConvOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		std::map<std::string, std::vector<int>> Conv_layers;
		// getConvOperations (special case of multiple FanIn with shared weights)
		for (size_t operations_iter1 = 0; operations_iter1 < FP_operations.size(); ++operations_iter1) {
			if (identified_sink_nodes.count(FP_operations[operations_iter1].result.sink_node->getName())) continue; // Skip identified sink nodes
			for (size_t operations_iter2 = operations_iter1 + 1; operations_iter2 < FP_operations.size(); ++operations_iter2) {
				if (identified_sink_nodes.count(FP_operations[operations_iter2].result.sink_node->getName())) continue; // Skip identified sink nodes

				// check if the sink nodes are compatible
				std::string ops_key_1 = makeForwardPropogationOperationsKey(FP_operations[operations_iter1].result.time_step,
					FP_operations[operations_iter1].result.sink_node->getType(),
					FP_operations[operations_iter1].result.sink_node->getIntegration()->getName(),
					FP_operations[operations_iter1].result.sink_node->getActivation()->getName());
				std::string ops_key_2 = makeForwardPropogationOperationsKey(FP_operations[operations_iter2].result.time_step,
					FP_operations[operations_iter2].result.sink_node->getType(),
					FP_operations[operations_iter2].result.sink_node->getIntegration()->getName(),
					FP_operations[operations_iter2].result.sink_node->getActivation()->getName());
				if (ops_key_1 != ops_key_2) continue;

				// check for shared weights
				std::set<std::string> argument_weights, argument_weights_1, argument_weights_2;
				for (const auto& argument : FP_operations[operations_iter1].arguments) {
					argument_weights.insert(argument.weight->getName());
					argument_weights_1.insert(argument.weight->getName());
				}
				for (const auto& argument : FP_operations[operations_iter2].arguments) {
					argument_weights.insert(argument.weight->getName());
					argument_weights_2.insert(argument.weight->getName());
				}
				if (argument_weights.size() != argument_weights_1.size() || argument_weights.size() != argument_weights_2.size()) continue;

				// update the maps
				identified_sink_nodes.insert(FP_operations[operations_iter1].result.sink_node->getName());
				auto found = Conv_layers.emplace(FP_operations[operations_iter1].result.sink_node->getName(), std::vector<int>({ operations_iter1 }));
				Conv_layers.at(FP_operations[operations_iter1].result.sink_node->getName()).push_back(operations_iter2);
			}
		}
		return Conv_layers;
	}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT>::getFanOutOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		return std::map<std::string, std::vector<int>>();
	}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT>::getFanInOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		// Default of what is left...
		return std::map<std::string, std::vector<int>>();
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT>::getForwardPropogationLayerTensorDimensions(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes,
		std::vector<bool>& make_source_tensors, std::vector<bool>& make_sink_tensors, std::vector<bool>& make_weight_tensors) {
		for (const auto& operations : FC_ops) {
			// determine the tensor sizes
			const int layer_index = operations_cache_.size();
			int sink_layer_size = 0;
			int source_layer_size = 0;
			bool make_sink_tensor = true;
			bool make_source_tensor = true;
			bool make_weight_tensor = true;

			for (const int& ops_index : operations.second) {
				// allocate sink node tensors (if it does not yet exist)
				if (FP_operations[ops_index].result.sink_node->getLayerId().first == -1) {
					FP_operations[ops_index].result.sink_node->getLayerId().first = layer_index;
					FP_operations[ops_index].result.sink_node->getLayerId().second = sink_layer_size;
				}
				else
					make_sink_tensor = false;

				// allocate source node tensor (if it does not yet exist)
				for (const OperationArgument& argument : FP_operations[ops_index].arguments) {
					if (argument.source_node->getLayerId().first == -1) {
						argument.source_node->getLayerId().first = layer_index;
						argument.source_node->getLayerId().second = source_layer_size;
					}
					else
						make_source_tensor = false;

					// allocate weight tensors
					if (std::get<0>(argument.weight->getLayerId()) == -1) {
						std::get<0>(argument.weight->getLayerId()) = layer_index;
						std::get<1>(argument.weight->getLayerId()) = source_layer_size;
						std::get<2>(argument.weight->getLayerId()) = sink_layer_size;
					}
					else
						make_weight_tensor = false;

					++source_layer_size; //?
				}
				++sink_layer_size; //?
			}
			sink_layer_sizes.push_back(sink_layer_size);
			source_layer_sizes.push_back(source_layer_size);
			make_source_tensors.push_back(make_source_tensor);
			make_sink_tensors.push_back(make_sink_tensor);
			make_weight_tensors.push_back(make_weight_tensor);
		}
	}

	template<typename TensorT, typename DeviceT>
	std::string ModelInterpreter<TensorT>::makeForwardPropogationOperationsKey(const int & time_step, const NodeType& node_type, const std::string & node_integration, const std::string & node_activation)
	{
		// [TODO: make tests; this appears to break the forward propogation algorithm because it does not match the cyclic node name
		std::string ops_key = std::to_string(time_step) + "/" + std::to_string(node_type) + "/" + node_integration + "/" + node_activation;
		return ops_key;
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT>::getForwardPropogationOperations()
	{
		const int max_iters = 1e6;
		for (int iter = 0; iter < max_iters; ++iter)
		{
			// STEP 1: get an unoptimized set of operations for FP
			// get the next hidden layer
			std::map<std::string, int> FP_operations_map;
			std::vector<OperationList<TensorT>> FP_operations_list;
			getNextInactiveLayer(FP_operations_map, FP_operations_list);

			// get biases,
			std::vector<std::string> sink_nodes_with_biases;
			getNextInactiveLayerBiases(FP_operations_map, FP_operations_list, sink_nodes_with_biases);

			// get cycles
			std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
			std::vector<OperationList<TensorT>> FP_operations_list_cycles = FP_operations_list;
			std::vector<std::string> sink_nodes_cycles;
			getNextInactiveLayerCycles(FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_cycles);

			// Remove all nodes involved in "cycles" that have arguments
			// involving source to sink node pairs not identified as cycles
			pruneInactiveLayerCycles(FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_cycles);

			// check if all nodes have been activated
			if (FP_operations_list.size() == 0)
			{
				break;
			}

			// STEP 2: optimized the operations set for hardware acceleration
			// re-organize into tensors
			std::vector<OperationList<TensorT>> FP_operations_expanded;
			expandForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

			// identify tensor operation motifs
			std::set<std::string> identified_sink_nodes;
			std::map<std::string, std::vector<int>> custom_ops = getCustomOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> FC_ops = getFullyConnectedOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> SC_ops = GetSinglyConnectedOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> Conv_ops = getConvOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> FIn_ops = getFanOutOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> FOut_ops = getFanInOperations(FP_operations, identified_sink_nodes);

			// allocate memory for tensors
			if (custom_ops.size() != 0)
				allocateForwardPropogationLayerTensors(FP_operations, custom_ops);
			if (FC_ops.size() != 0)
				allocateForwardPropogationLayerTensors(FP_operations, FC_ops);
			if (SC_ops.size() != 0)
				allocateForwardPropogationLayerTensors(FP_operations, SC_ops);
			if (Conv_ops.size() != 0)
				allocateForwardPropogationLayerTensors(FP_operations, Conv_ops);
			if (FIn_ops.size() != 0)
				allocateForwardPropogationLayerTensors(FP_operations, FIn_ops);
			if (FOut_ops.size() != 0)
				allocateForwardPropogationLayerTensors(FP_operations, FOut_ops);

			// activate sink nodes
			for (auto& FP_operation : FP_operations_list)
				FP_operation.result.sink_node->setStatus(NodeStatus::activated);
		}
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT>::clearCache()
	{
		operations_cache_.clear();
	}


	template<typename TensorT>
	class ModelInterpreterDefaultDevice : public ModelInterpreter<TensorT, Eigen::DefaultDevice>
	{
	public:
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes,
			std::vector<bool>& make_source_tensor, std::vector<bool>& make_sink_tensor, std::vector<bool>& make_weight_tensor);
		void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeModelErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
	};

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes,
		std::vector<bool>& make_source_tensors, std::vector<bool>& make_sink_tensors, std::vector<bool>& make_weight_tensors)
	{
		std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>> operation_step_list;
		for (int i = 0; i < source_layer_sizes.size(); ++i) {

			// make the tensors
			OperationTensorStep<TensorT, Eigen::DefaultDevice> operation_step;
			batch_memory_size = getBatchAndMemorySizes();

			// make the source layer tensor
			NodeMatrixDataCpu<TensorT> source_node_data;
			if (make_source_tensor) {
				source_node_data.setBatchSize(batch_memory_size.first);
				source_node_data.setMemorySize(batch_memory_size.second);
				source_node_data.setLayerSize(source_layer_size);
				// [TODO: how best to set input, output, derivative, error, dt?]
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
			}

			operation_step.source_layer.reset(&source_node_data);
			operation_step.source_time_step = FP_operations[operations.second[0]].arguments[0].time_step;
			// [TODO: set the integration functions]

			// make the sink layer tensor
			NodeMatrixDataCpu<TensorT> sink_node_data;
			if (make_sink_tensor) {
				sink_node_data.setBatchSize(batch_memory_size.first);
				sink_node_data.setMemorySize(batch_memory_size.second);
				sink_node_data.setLayerSize(sink_layer_size);
				// [TODO: how best to set input, output, derivative, error, dt?]
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
			}

			operation_step.sink_layer.reset(&sink_node_data);
			operation_step.sink_time_step = FP_operations[operations.second[0]].result.time_step;
			// [TODO: set the integration functions]

			// make the weight tensor
			// [TODO: there are differences between FC, SC, FanIn, FanOut, and Conv that need to be accounted for!]

			WeightMatrixDataCpu<TensorT> weight_data;
			if (make_weight_tensor) {
				weight_data.setLayer1Size(source_layer_size);
				weight_data.setLayer2Size(sink_layer_size);
				weight_data.setNSolverParams(operation_step.source_time_step = FP_operations[operations.second[0]].arguments[0].weight->getSolverOp()->getNParameters());
				// [TODO: how to initialize the weights? use the first weight init function?]
				// [TODO: how to initialize the solver_params? use the first solver op?]
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
			}

			operation_step.weight.reset(&weight_data);

			operation_step_list.push_back(operation_step);
		}
		// add the tensors to the cache
		operations_cache_.push_back(operation_step_list);
	}

	template<typename TensorT>
	void ModelInterpreterDefaultDevice<TensorT>::executeForwardPropogationOperations(const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		for (auto& operations_list : operations_cache_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (auto& operation : operations_list) {

				bool success = model_kernal.executeForwardPropogation(
					operation.source_layer->,
					d_source_outputs,
					h_weights,
					d_weights,
					h_sink_input,
					d_sink_input,
					integration_function,
					batch_size,
					memory_size,
					source_layer_size,
					sink_layer_size,
					source_time_steps,
					sink_time_step,
					device,
					true,
					true);

				bool success = model_kernal.executeNodeActivation(
					h_node_input,
					d_node_input,
					h_node_output,
					d_node_output,
					h_node_dt,
					d_node_dt,
					activation_function,
					batch_size,
					memory_size,
					layer_size,
					node_time_step,
					device,
					true,
					true);
			}
		}
	}

#if COMPILE_WITH_CUDA
	template<typename TensorT>
	class ModelInterpreterGpu : public ModelInterpreter<TensorT, Eigen::GpuDevice>
	{
	public:
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes,
			std::vector<bool>& make_source_tensor, std::vector<bool>& make_sink_tensor, std::vector<bool>& make_weight_tensor);
		void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeModelErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
	};

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes,
		std::vector<bool>& make_source_tensors, std::vector<bool>& make_sink_tensors, std::vector<bool>& make_weight_tensors)
	{
		std::vector<OperationTensorStep<TensorT, Eigen::GpuDevice>> operation_step_list;
		for (int i = 0; i < source_layer_sizes.size(); ++i) {

			// make the tensors
			OperationTensorStep<TensorT, Eigen::GpuDevice> operation_step;
			batch_memory_size = getBatchAndMemorySizes();

			// make the source layer tensor
			NodeMatrixDataGpu<TensorT> source_node_data;
			if (make_source_tensor) {
				source_node_data.setBatchSize(batch_memory_size.first);
				source_node_data.setMemorySize(batch_memory_size.second);
				source_node_data.setLayerSize(source_layer_size);
				// [TODO: how best to set input, output, derivative, error, dt?]
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
			}

			operation_step.source_layer.reset(&source_node_data);
			operation_step.source_time_step = FP_operations[operations.second[0]].arguments[0].time_step;
			// [TODO: set the integration functions]

			// make the sink layer tensor
			NodeMatrixDataGpu<TensorT> sink_node_data;
			if (make_sink_tensor) {
				sink_node_data.setBatchSize(batch_memory_size.first);
				sink_node_data.setMemorySize(batch_memory_size.second);
				sink_node_data.setLayerSize(sink_layer_size);
				// [TODO: how best to set input, output, derivative, error, dt?]
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
			}

			operation_step.sink_layer.reset(&sink_node_data);
			operation_step.sink_time_step = FP_operations[operations.second[0]].result.time_step;
			// [TODO: set the integration functions]

			// make the weight tensor
			// [TODO: there are differences between FC, SC, FanIn, FanOut, and Conv that need to be accounted for!]

			WeightMatrixDataGpu<TensorT> weight_data;
			if (make_weight_tensor) {
				weight_data.setLayer1Size(source_layer_size);
				weight_data.setLayer2Size(sink_layer_size);
				weight_data.setNSolverParams(operation_step.source_time_step = FP_operations[operations.second[0]].arguments[0].weight->getSolverOp()->getNParameters());
				// [TODO: how to initialize the weights? use the first weight init function?]
				// [TODO: how to initialize the solver_params? use the first solver op?]
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
			}

			operation_step.weight.reset(&weight_data);

			operation_step_list.push_back(operation_step);
		}
		// add the tensors to the cache
		operations_cache_.push_back(operation_step_list);
	}

	template<typename TensorT>
	void ModelInterpreterGpu<TensorT>::executeForwardPropogationOperations(const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		int FP_operations_cnt = 0;
		for (auto& operations_list : operations_cache_) {

			// Set up the device, streams, and kernals
			ModelKernalGpu<TensorT> model_kernal;
			const int device_id = 0;
			assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?
			std::vector<cudaStream_t> streams;
			std::vector<Eigen::GpuStreamDevice> stream_devices;
			std::vector<Eigen::GpuDevice> devices;
			for (size_t i = 0; i < operations_list.size(); ++i) {
				cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
				assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
				streams.push_back(stream);
				Eigen::GpuStreamDevice stream_device(&stream, 0);
				stream_devices.push_back(stream_device);
				Eigen::GpuDevice device(&stream_device);
				devices.push_back(device);
			}

			// execute the forward propogation steps
			for (auto& operation : operations_list) {
				// 

				// activate the net input
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_list.size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}
#endif

}
#endif //SMARTPEAK_MODELINTERPRETER_H