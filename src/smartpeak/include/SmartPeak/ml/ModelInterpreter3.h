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
#include <SmartPeak/ml/Model3.h>
#include <SmartPeak/ml/LossFunctionTensor.h>
#include <SmartPeak/ml/NodeMatrixData.h>
#include <SmartPeak/ml/WeightMatrixData.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>
#include <SmartPeak/ml/ActivationFunctionWrapper.h>
#include <SmartPeak/ml/SolverTensor.h>
#include <SmartPeak/ml/WeightInit.h>

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
	class OperationLayer
	{
	public:
		std::shared_ptr<NodeMatrixData<TensorT>> tensor = nullptr;
		int time_step = 0;
		std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> integration = nullptr;
		std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> integration_error = nullptr;
		std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> integration_weight_grad = nullptr;
		std::shared_ptr<ActivationOpWrapper<TensorT, DeviceT>> activation = nullptr;
		std::shared_ptr<ActivationOpWrapper<TensorT, DeviceT>> activation_grad = nullptr;
	};

	template<typename TensorT, typename DeviceT>
	class OperationWeight
	{
	public:
		std::shared_ptr<WeightMatrixData<TensorT>> tensor = nullptr;
		std::shared_ptr<WeightInitOp<TensorT>> weight_init = nullptr;
		std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> solver = nullptr;
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
	};

	template<typename TensorT>
	class OperationTensorStepDefaultDevice : public OperationTensorStep<TensorT, Eigen::DefaultDevice>
	{
	public:
		OperationLayer<TensorT, Eigen::DefaultDevice> sink_layer;
		OperationLayer<TensorT, Eigen::DefaultDevice> source_layer;
		OperationWeight<TensorT, Eigen::DefaultDevice> weight;
	};

	template<typename TensorT>
	class OperationTensorStepCpu : public OperationTensorStep<TensorT, Eigen::ThreadPool>
	{
	public:
		OperationLayer<TensorT, Eigen::ThreadPool> sink_layer;
		OperationLayer<TensorT, Eigen::ThreadPool> source_layer;
		OperationWeight<TensorT, Eigen::ThreadPool> weight;
	};

	template<typename TensorT>
	class OperationTensorStepGpu : public OperationTensorStep<TensorT, Eigen::GpuDevice>
	{
	public:
		OperationLayer<TensorT, Eigen::GpuDevice> sink_layer;
		OperationLayer<TensorT, Eigen::GpuDevice> source_layer;
		OperationWeight<TensorT, Eigen::GpuDevice> weight;
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
			Model<TensorT>& model,
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
		void getNextInactiveLayer(Model<TensorT>& model,
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

		void getForwardPropogationOperations(Model<TensorT>& model, const int& batch_size, const int& memory_size);

		/**
		@brief Allocate Node and Weight tensor memory for all model operations.
			Weight solver params tensors are created using the first weight in the layer.
			Weight matrix is initialized using the first weight in the layer

		@param[in] FP_operations
		@param[in] ...
		*/
		virtual void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size) = 0;
		virtual void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id, LossFunctionTensorOp<TensorT, DeviceT>* loss_function, LossFunctionGradTensorOp<TensorT, DeviceT>* loss_function_grad, const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		
		virtual void clearCache(); ///< clear the operations caches
	};

	template<typename TensorT, typename DeviceT>
	ModelInterpreter<TensorT, DeviceT>::ModelInterpreter(const ModelInterpreter<TensorT, DeviceT>& other)
	{
		operations_cache_ = other.operations_cache_;
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::mapValuesToLayers(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& values, const std::vector<std::string>& node_names, const NodeStatus & status_update, const std::string & value_type)
	{
		// check dimension mismatches
		if (node_names.size() != values.dimension(2))
		{
			printf("The number of input features %d and the number of nodes %d do not match.\n", (int)values.dimension(2), node_names.size());
			return;
		}
		// assumes the tensor operations have been cached
		else if (operations_cache_[0][0].source_layer.tensor->getBatchSize() != values.dimension(0))
		{
			printf("The number of input samples %d and the batch size %d does not match.\n", (int)values.dimension(0), (int)operations_cache_[0][0].source_layer.tensor->getBatchSize());
			return;
		}
		else if (operations_cache_[0][0].source_layer.tensor->getMemorySize() != values.dimension(1))
		{
			printf("The number of input time steps %d and the memory size %d does not match.\n", (int)values.dimension(1), (int)operations_cache_[0][0].source_layer.tensor->getMemorySize());
			return;
		}

		for (std::vector<OperationTensorStep<TensorT, DeviceT>>& operations_list : operations_cache_) {
			for (OperationTensorStep<TensorT, DeviceT>& operation : operations_list) {
				int layer_size = 0;
				// check that all nodes are in the layer
				// check that the # of nodes matches the tensor layer size
				// copy over the values
				if (value_type == "output")
					operation.source_layer.tensor->setOutput(values);
				else if (value_type == "error")
					operation.source_layer.tensor->setError(values);
				else if (value_type == "derivative")
					operation.source_layer.tensor->setDerivative(values);
				else if (value_type == "dt")
					operation.source_layer.tensor->setDt(values);
			}
		}
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getNextInactiveLayer(Model<TensorT>& model,
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations)
	{

		// get all links where the source node is active and the sink node is inactive
		// except for biases
		for (auto& link_map : model.getLinksMap())
		{
			if (model.getNodesMap().at(link_map.second->getSourceNodeName())->getType() != NodeType::bias &&
				model.getNodesMap().at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				model.getNodesMap().at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = model.getNodesMap().at(link_map.second->getSourceNodeName());
				arguments.weight = model.getWeightsMap().at(link_map.second->getWeightName());
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
					result.sink_node = model.getNodesMap().at(link_map.second->getSinkNodeName());
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
		for (auto& link_map : model.getLinksMap())
		{
			std::string ops_key = link_map.second->getSinkNodeName();
			if (
				// does not allow for cycles
				model.getNodesMap().at(link_map.second->getSourceNodeName())->getType() == NodeType::bias &&
				model.getNodesMap().at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				// required regardless if cycles are or are not allowed
				model.getNodesMap().at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
				FP_operations_map.count(ops_key) != 0 // sink node has already been identified
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = model.getNodesMap().at(link_map.second->getSourceNodeName());
				arguments.weight = model.getWeightsMap().at(link_map.second->getWeightName());
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
		for (auto& link_map : model.getLinksMap())
		{
			std::string ops_key = link_map.second->getSinkNodeName();
			if (
				model.getNodesMap().at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::initialized &&
				// required regardless if cycles are or are not allowed
				model.getNodesMap().at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::initialized &&
				FP_operations_map.count(ops_key) != 0 // sink node has already been identified
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = model.getNodesMap().at(link_map.second->getSourceNodeName());
				arguments.weight = model.getWeightsMap().at(link_map.second->getWeightName());

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
	inline void ModelInterpreter<TensorT, DeviceT>::expandForwardPropogationOperationsByWeightKey(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
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
	inline void ModelInterpreter<TensorT, DeviceT>::expandForwardPropogationOperationsByCachedNodes(const std::vector<OperationList<TensorT>>& FP_operations, std::vector<OperationList<TensorT>>& FP_operations_expanded)
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getCustomOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getFullyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::GetSinglyConnectedOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getConvOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getFanOutOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		return std::map<std::string, std::vector<int>>();
	}

	template<typename TensorT, typename DeviceT>
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getFanInOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
	{
		// Default of what is left...
		return std::map<std::string, std::vector<int>>();
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::getForwardPropogationLayerTensorDimensions(const std::vector<OperationList<TensorT>>& FP_operations,
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
	std::string ModelInterpreter<TensorT, DeviceT>::makeForwardPropogationOperationsKey(const int & time_step, const NodeType& node_type, const std::string & node_integration, const std::string & node_activation)
	{
		// [TODO: make tests; this appears to break the forward propogation algorithm because it does not match the cyclic node name
		std::string ops_key = std::to_string(time_step) + "/" + std::to_string(node_type) + "/" + node_integration + "/" + node_activation;
		return ops_key;
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getForwardPropogationOperations(Model<TensorT>& model, const int& batch_size, const int& memory_size)
	{
		// initialize the input nodes to active
		for (auto& input_node : model.getInputNodes()) {
			input_node->setStatus(NodeStatus::activated);
		}

		const int max_iters = 1e6;
		for (int iter = 0; iter < max_iters; ++iter)
		{
			// STEP 1: get an unoptimized set of operations for FP
			// get the next hidden layer
			std::map<std::string, int> FP_operations_map;
			std::vector<OperationList<TensorT>> FP_operations_list;
			getNextInactiveLayer(model, FP_operations_map, FP_operations_list);

			// get biases,
			std::vector<std::string> sink_nodes_with_biases;
			getNextInactiveLayerBiases(model, FP_operations_map, FP_operations_list, sink_nodes_with_biases);

			// get cycles
			std::map<std::string, int> FP_operations_map_cycles = FP_operations_map;
			std::vector<OperationList<TensorT>> FP_operations_list_cycles = FP_operations_list;
			std::vector<std::string> sink_nodes_cycles;
			getNextInactiveLayerCycles(model, FP_operations_map_cycles, FP_operations_list_cycles, sink_nodes_cycles);

			// Remove all nodes involved in "cycles" that have arguments
			// involving source to sink node pairs not identified as cycles
			pruneInactiveLayerCycles(model, FP_operations_map, FP_operations_map_cycles, FP_operations_list, FP_operations_list_cycles, sink_nodes_cycles);

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
			std::map<std::string, std::vector<int>> SC_ops = getSinglyConnectedOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> Conv_ops = getConvOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> FIn_ops = getFanOutOperations(FP_operations, identified_sink_nodes);
			std::map<std::string, std::vector<int>> FOut_ops = getFanInOperations(FP_operations, identified_sink_nodes);

			// allocate memory for tensors
			if (custom_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations, custom_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations, custom_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size);
			}
			if (FC_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations, custom_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations, FC_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size);
			}
			if (SC_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations, custom_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations, SC_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size);
			}
			if (Conv_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations, custom_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations, Conv_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size);
			}
			if (FIn_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations, custom_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations, FIn_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size);
			}
			if (FOut_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations, custom_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations, FOut_ops, source_layer_sizes, sink_layer_sizes, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size);
			}

			// activate sink nodes
			for (auto& FP_operation : FP_operations_list)
				FP_operation.result.sink_node->setStatus(NodeStatus::activated);
		}
	}

	template<typename TensorT>
	class ModelInterpreterDefaultDevice : public ModelInterpreter<TensorT, Eigen::DefaultDevice>
	{
	public:
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size);
		void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id, LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_function_grad, const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void clearCache();
	protected:
		std::vector<std::vector<OperationTensorStepDefaultDevice<TensorT>>> operations_cache_;
		ModelErrorDataCpu<TensorT> model_error_;
	};

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes,
		const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
		const int& batch_size, const int& memory_size)
	{
		std::vector<OperationTensorStepDefaultDevice<TensorT>> operation_step_list;
		int iter = 0;
		for (const auto& operations : operations_map) {

			// make the tensors
			OperationTensorStepDefaultDevice<TensorT> operation_step;

			// make the source layer tensor
			NodeMatrixDataCpu<TensorT> source_node_data;
			if (make_source_tensors[iter]) {
				source_node_data.setBatchSize(batch_size);
				source_node_data.setMemorySize(memory_size);
				source_node_data.setLayerSize(source_layer_sizes[iter]);
				// [TODO: how best to set input, output, derivative, error, dt?]
				// [TODO: implement initNodeData?]
				operation_step.source_layer.tensor.reset(&source_node_data);
				operation_step.source_layer.time_step = FP_operations[operations.second[0]].arguments[0].time_step;
				// [TODO: how to copy out the node integration and activation classes 
				// - integration functions can be parameters of the class based on the motif
				// - look up dictionary for activation and activation_grad functions?]
				//operation_step.source_layer.activation = FP_operations[operations.second[0]].arguments[0].source_node->getActivationShared();
				//operation_step.source_layer.activation_grad = FP_operations[operations.second[0]].arguments[0].source_node->getActivationGradShared();
				//operation_step.source_layer.integration = FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationShared();
				//operation_step.source_layer.integration_error = FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationErrorShared();
				//operation_step.source_layer.integration_weight_grad = FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationWeightGradShared();
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
				// [TODO: update the time step]
			}
			
			// make the sink layer tensor
			NodeMatrixDataCpu<TensorT> sink_node_data;
			if (make_sink_tensors[iter]) {
				sink_node_data.setBatchSize(batch_size);
				sink_node_data.setMemorySize(memory_size);
				sink_node_data.setLayerSize(sink_layer_sizes[iter]);
				// [TODO: how best to set input, output, derivative, error, dt?]
				operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
				operation_step.sink_layer.tensor.reset(&sink_node_data);
				// [TODO: set the integration functions]
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
				// [TODO: update the time step]
			}


			// make the weight tensor
			// [TODO: there are differences between FC, SC, FanIn, FanOut, and Conv that need to be accounted for!]

			WeightMatrixDataCpu<TensorT> weight_data;
			if (make_weight_tensors[iter]) {
				weight_data.setLayer1Size(source_layer_sizes[iter]);
				weight_data.setLayer2Size(sink_layer_sizes[iter]);
				weight_data.setNSolverParams(FP_operations[operations.second[0]].arguments[0].weight->getSolverOp()->getNParameters());
				Eigen::Tensor<TensorT, 2> weight_tensor(source_layer_sizes[iter], sink_layer_sizes[iter]);
				// [TODO: implemented initWeight (init values and solver params)
				//weight_tensor.unaryExpr(*(FP_operations[operations.second[0]].arguments[0].weight->getWeightInitOp()));
				// [TODO: how to initialize the solver_params? use the first solver op?]
				operation_step.weight.tensor.reset(&weight_data);
			}
			else {
				// [TODO: copy out the sink_node_data if it already exists]
			}


			operation_step_list.push_back(operation_step);

			++iter;
		}
		// add the tensors to the cache
		operations_cache_.push_back(operation_step_list);
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeForwardPropogationOperations(const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStepDefaultDevice<TensorT>>& operations_list : operations_cache_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStepDefaultDevice<TensorT>& operation : operations_list) {
				model_kernal.executeForwardPropogation(
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.sink_layer.integration.get(),
					operation.source_layer.tensor->getBatchSize(),
					operation.source_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.source_layer.time_step + time_step,
					operation.sink_layer.time_step + time_step,
					device, sync_HToD, sync_DToH);

				model_kernal.executeNodeActivation(
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.sink_layer.tensor->getHOutputPointer().get(),
					operation.sink_layer.tensor->getDOutputPointer().get(),
					operation.sink_layer.tensor->getHDtPointer().get(),
					operation.sink_layer.tensor->getDDtPointer().get(),
					operation.sink_layer.activation.get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeBackwardPropogationOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (size_t i = operations_cache_.size() - 1; i >= 0; --i) { //iterate backwards
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStepDefaultDevice<TensorT>& operation : operations_cache_[i]) { //reverse source/sink

				model_kernal.executeNodeDerivative(
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.source_layer.tensor->getHDerivativePointer().get(),
					operation.source_layer.tensor->getDDerivativePointer().get(),
					operation.source_layer.activation_grad.get(),
					operation.source_layer.tensor->getBatchSize(),
					operation.source_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.source_layer.time_step + time_step,
					device, sync_HToD, sync_DToH);

				model_kernal.executeBackwardPropogation(
					operation.sink_layer.tensor->getHErrorPointer().get(),
					operation.sink_layer.tensor->getDErrorPointer().get(),
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.source_layer.tensor->getHErrorPointer().get(),
					operation.source_layer.tensor->getDErrorPointer().get(),
					operation.source_layer.tensor->getHDerivativePointer().get(),
					operation.source_layer.tensor->getDDerivativePointer().get(),
					operation.source_layer.tensor->getLayerSize(), // [TODO: replace with N]
					operation.source_layer.integration_error.get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					operation.source_layer.time_step + time_step,
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id,	LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function,	LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_function_grad, const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		ModelKernalDefaultDevice<TensorT> model_kernal;
		Eigen::DefaultDevice device;
		OperationTensorStepDefaultDevice<TensorT> operation = operations_cache_[layer_id.first][layer_id.second];
		model_kernal.executeModelErrors(
			expected,
			operation.sink_layer.tensor->getHOutputPointer().get(),
			operation.sink_layer.tensor->getDOutputPointer().get(),
			model_error_.getHErrorPointer().get(),
			model_error_.getDErrorPointer().get(),
			operation.sink_layer.tensor->getHErrorPointer().get(),
			operation.sink_layer.tensor->getDErrorPointer().get(),
			loss_function,
			loss_function_grad,
			operation.sink_layer.tensor->getBatchSize(),
			operation.sink_layer.tensor->getMemorySize(),
			operation.sink_layer.tensor->getLayerSize(),
			time_step,
			device, sync_HToD, sync_DToH);
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeWeightErrorOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStepDefaultDevice<TensorT>>& operations_list : operations_cache_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStepDefaultDevice<TensorT>& operation : operations_list) {

				model_kernal.executeWeightErrors(
					operation.sink_layer.tensor->getHErrorPointer().get(),
					operation.sink_layer.tensor->getDErrorPointer().get(),
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.source_layer.tensor->getHInputPointer().get(),
					operation.source_layer.tensor->getDInputPointer().get(),
					operation.source_layer.tensor->getLayerSize(), // [TODO: change to N]
					operation.sink_layer.integration_weight_grad.get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.weight.tensor->getHErrorPointer().get(),
					operation.weight.tensor->getDErrorPointer().get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeWeightUpdateOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStepDefaultDevice<TensorT>>& operations_list : operations_cache_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStepDefaultDevice<TensorT>& operation : operations_list) {

				model_kernal.executeWeightUpdate(
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.weight.tensor->getHSolverParamsPointer().get(),
					operation.weight.tensor->getDSolverParamsPointer().get(),
					operation.weight.tensor->getHErrorPointer().get(),
					operation.weight.tensor->getDErrorPointer().get(),
					operation.weight.solver.get(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					device, sync_HToD, sync_DToH);
			}
		}
	}

	template<typename TensorT>
	void ModelInterpreterDefaultDevice<TensorT>::clearCache()
	{
		operations_cache_.clear();
	}

#if COMPILE_WITH_CUDA
	template<typename TensorT>
	class ModelInterpreterGpu : public ModelInterpreter<TensorT, Eigen::GpuDevice>
	{
	public:
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size);
		void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id, LossFunctionTensorOp<TensorT, Eigen::GpuDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::GpuDevice>* loss_function_grad, const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void clearCache();
	protected:
		std::vector<std::vector<OperationTensorStepGpu<TensorT>>> operations_cache_;
		ModelErrorDataGpu<TensorT> model_error_;
	};

	template<typename TensorT>
	void ModelInterpreterGpu<TensorT>::executeForwardPropogationOperations(const int& time_step, bool sync_HToD, bool sync_DToH)
	{
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
			int device_iter = 0;
			for (OperationTensorStepGpu<TensorT>& operation : operations_list) {
				model_kernal.executeForwardPropogation(
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.sink_layer.integration.get(),
					operation.source_layer.tensor->getBatchSize(),
					operation.source_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.source_layer.time_step + time_step,
					operation.sink_layer.time_step + time_step,
					devices[device_iter], sync_HToD, sync_DToH);

				model_kernal.executeNodeActivation(
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.sink_layer.tensor->getHOutputPointer().get(),
					operation.sink_layer.tensor->getDOutputPointer().get(),
					operation.sink_layer.tensor->getHDtPointer().get(),
					operation.sink_layer.tensor->getDDtPointer().get(),
					operation.sink_layer.activation.get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					devices[device_iter], sync_HToD, sync_DToH);
				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_list.size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeBackwardPropogationOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (size_t iter = operations_cache_.size() - 1; iter >= 0; --iter) { //iterate backwards

			// Set up the device, streams, and kernals
			ModelKernalGpu<TensorT> model_kernal;
			const int device_id = 0;
			assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?
			std::vector<cudaStream_t> streams;
			std::vector<Eigen::GpuStreamDevice> stream_devices;
			std::vector<Eigen::GpuDevice> devices;
			for (size_t i = 0; i < operations_cache_[iter].size(); ++i) {
				cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
				assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
				streams.push_back(stream);
				Eigen::GpuStreamDevice stream_device(&stream, 0);
				stream_devices.push_back(stream_device);
				Eigen::GpuDevice device(&stream_device);
				devices.push_back(device);
			}

			// execute the forward propogation steps
			int device_iter = 0;
			for (OperationTensorStepGpu<TensorT>& operation : operations_cache_[iter]) { //reverse source/sink

				model_kernal.executeNodeDerivative(
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.source_layer.tensor->getHDerivativePointer().get(),
					operation.source_layer.tensor->getDDerivativePointer().get(),
					operation.source_layer.activation_grad.get(),
					operation.source_layer.tensor->getBatchSize(),
					operation.source_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.source_layer.time_step + time_step,
					devices[device_iter], sync_HToD, sync_DToH);

				model_kernal.executeBackwardPropogation(
					operation.sink_layer.tensor->getHErrorPointer().get(),
					operation.sink_layer.tensor->getDErrorPointer().get(),
					operation.sink_layer.tensor->getHInputPointer().get(),
					operation.sink_layer.tensor->getDInputPointer().get(),
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.source_layer.tensor->getHErrorPointer().get(),
					operation.source_layer.tensor->getDErrorPointer().get(),
					operation.source_layer.tensor->getHDerivativePointer().get(),
					operation.source_layer.tensor->getDDerivativePointer().get(),
					operation.source_layer.tensor->getLayerSize(), // [TODO: replace with N]
					operation.source_layer.integration_error.get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.sink_layer.tensor->getLayerSize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.time_step + time_step,
					operation.source_layer.time_step + time_step,
					device[device_iter], sync_HToD, sync_DToH);

				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_cache_[iter].size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id, LossFunctionTensorOp<TensorT, Eigen::GpuDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::GpuDevice>* loss_function_grad, const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		// More performant if all model error calculations were passed at the same time
		ModelKernalGpu<TensorT> model_kernal;
		cudaStream_t stream; // The stream will be destroyed by GpuStreamDevice once the function goes out of scope!
		assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
		streams.push_back(stream);
		Eigen::GpuStreamDevice stream_device(&stream, 0);
		stream_devices.push_back(stream_device);
		Eigen::GpuDevice device(&stream_device);

		OperationTensorStepDefaultDevice<TensorT> operation = operations_cache_[layer_id.first][layer_id.second];
		model_kernal.executeModelErrors(
			expected,
			operation.sink_layer.tensor->getHOutputPointer().get(),
			operation.sink_layer.tensor->getDOutputPointer().get(),
			model_error_.getHErrorPointer().get(),
			model_error_.getDErrorPointer().get(),
			operation.sink_layer.tensor->getHErrorPointer().get(),
			operation.sink_layer.tensor->getDErrorPointer().get(),
			loss_function,
			loss_function_grad,
			operation.sink_layer.tensor->getBatchSize(),
			operation.sink_layer.tensor->getMemorySize(),
			operation.sink_layer.tensor->getLayerSize(),
			time_step,
			device, sync_HToD, sync_DToH);

		assert(cudaStreamSynchronize(streams) == cudaSuccess);
		assert(cudaStreamDestroy(streams) == cudaSuccess);
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeWeightErrorOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStepDefaultDevice<TensorT>>& operations_list : operations_cache_) {

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
			int device_iter = 0;
			for (OperationTensorStepGpu<TensorT>& operation : operations_list) {

				model_kernal.executeWeightErrors(
					operation.sink_layer.tensor->getHErrorPointer().get(),
					operation.sink_layer.tensor->getDErrorPointer().get(),
					operation.source_layer.tensor->getHOutputPointer().get(),
					operation.source_layer.tensor->getDOutputPointer().get(),
					operation.source_layer.tensor->getHInputPointer().get(),
					operation.source_layer.tensor->getDInputPointer().get(),
					operation.source_layer.tensor->getLayerSize(), // [TODO: change to N]
					operation.sink_layer.integration_weight_grad.get(),
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.weight.tensor->getHErrorPointer().get(),
					operation.weight.tensor->getDErrorPointer().get(),
					operation.sink_layer.tensor->getBatchSize(),
					operation.sink_layer.tensor->getMemorySize(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					device[device_iter], sync_HToD, sync_DToH);
				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_list.size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::executeWeightUpdateOperations(const int & time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStepDefaultDevice<TensorT>>& operations_list : operations_cache_) {

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
			int device_iter = 0;
			for (OperationTensorStepGpu<TensorT>& operation : operations_list) {

				model_kernal.executeWeightUpdate(
					operation.weight.tensor->getHWeightPointer().get(),
					operation.weight.tensor->getDWeightPointer().get(),
					operation.weight.tensor->getHSolverParamsPointer().get(),
					operation.weight.tensor->getDSolverParamsPointer().get(),
					operation.weight.tensor->getHErrorPointer().get(),
					operation.weight.tensor->getDErrorPointer().get(),
					operation.weight.solver.get(),
					operation.source_layer.tensor->getLayerSize(),
					operation.sink_layer.tensor->getLayerSize(),
					device[device_iter], sync_HToD, sync_DToH);
				++device_iter;
			}

			// sync and destroy the streams
			for (size_t i = 0; i < operations_list.size(); ++i) {
				assert(cudaStreamSynchronize(streams[i]) == cudaSuccess);
				assert(cudaStreamDestroy(streams[i]) == cudaSuccess);
			}
		}
	}

	template<typename TensorT>
	void ModelInterpreterGpu<TensorT>::clearCache()
	{
		operations_cache_.clear();
	}
#endif

}
#endif //SMARTPEAK_MODELINTERPRETER_H