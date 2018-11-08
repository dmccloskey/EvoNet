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
#include <SmartPeak/ml/NodeTensorData.h>
#include <SmartPeak/ml/WeightTensorData.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <SmartPeak/ml/SolverTensor.h>
#include <SmartPeak/ml/LossFunctionTensor.h>
#include <SmartPeak/ml/OpToTensorOp.h>

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
		std::shared_ptr<NodeTensorData<TensorT>> tensor = nullptr;
		int time_step = 0;
		std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> integration = nullptr;
		std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> integration_error = nullptr;
		std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> integration_weight_grad = nullptr;
		std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> activation = nullptr;
		std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> activation_grad = nullptr;
	};

	template<typename TensorT, typename DeviceT>
	class OperationWeight
	{
	public:
		std::shared_ptr<WeightTensorData<TensorT>> tensor = nullptr;
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

	///*
	//Class specializations needed to operate with specific data specializations
	//*/
	//template<typename TensorT>
	//class OperationTensorStepDefaultDevice : public OperationTensorStep<TensorT, Eigen::DefaultDevice>
	//{
	//public:
	//	OperationLayer<TensorT, Eigen::DefaultDevice> sink_layer;
	//	OperationLayer<TensorT, Eigen::DefaultDevice> source_layer;
	//	OperationWeight<TensorT, Eigen::DefaultDevice> weight;
	//};

	//template<typename TensorT>
	//class OperationTensorStepCpu : public OperationTensorStep<TensorT, Eigen::ThreadPool>
	//{
	//public:
	//	OperationLayer<TensorT, Eigen::ThreadPool> sink_layer;
	//	OperationLayer<TensorT, Eigen::ThreadPool> source_layer;
	//	OperationWeight<TensorT, Eigen::ThreadPool> weight;
	//};

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
				) == std::tie(
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
		std::map<std::string, std::vector<int>> getTensorOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes);

		/**
		@brief Allocate Node and Weight tensor memory for all model operations.
			Node and weight tensors indices are registered.

		@param[in] FP_operations
		@param[in] ...
		*/
		void getForwardPropogationLayerTensorDimensions(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes, std::vector<std::vector<std::pair<int, int>>>& weight_indices, std::vector<std::vector<TensorT>>& weight_values,
			std::vector<bool>& make_source_tensor, std::vector<bool>& make_sink_tensor, std::vector<bool>& make_weight_tensor);

		static std::string makeForwardPropogationOperationsKey(const int& time_step, const NodeType& node_type,
			const std::string& node_integration, const std::string& node_activation);

		void getForwardPropogationOperations(Model<TensorT>& model, const int& batch_size, const int& memory_size, const bool& train);

		/**
		@brief Allocate Node and Weight tensor memory for all model operations.
			Weight solver params tensors are created using the first weight in the layer.
			Weight matrix is initialized using the first weight in the layer

		@param[in] FP_operations
		@param[in] ...
		*/
		virtual void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size, const bool& train) = 0;
		virtual void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id, LossFunctionTensorOp<TensorT, DeviceT>* loss_function, LossFunctionGradTensorOp<TensorT, DeviceT>* loss_function_grad, const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		virtual void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false) = 0;
		
		void addLayerTensor(NodeTensorData<TensorT>& layer); ///< add a layer to the cache
		void clearLayerTensors(); ///< clear all layers from the cache
		std::shared_ptr<NodeTensorData<TensorT>> getLayerTensor(const int& layer_index); ///< get a layer from the cache

		void addWeightTensor(WeightTensorData<TensorT>& weight); ///< add a weight to the cache
		void clearWeightTensors(); ///< clear all weights from the cache
		std::shared_ptr<WeightTensorData<TensorT>> getWeightTensor(const int& weight_index); ///< get a weight from the cache

		virtual void allocateModelErrorTensor(const int& batch_size, const int& memory_size) = 0; ///< set the model error
		std::shared_ptr<ModelErrorData<TensorT>> getModelError(); ///< get the model error

		void addOperationSteps(const std::vector<OperationTensorStep<TensorT, DeviceT>>& operation_steps);
		std::vector<OperationTensorStep<TensorT, DeviceT>> getOperationSteps(const int& operation_index);
		void clearOperationSteps(); ///< clear the operations caches

	protected:
		std::vector<std::vector<OperationTensorStep<TensorT, DeviceT>>> operation_steps_;
		std::vector<std::shared_ptr<NodeTensorData<TensorT>>> layer_tensors_;
		std::vector<std::shared_ptr<WeightTensorData<TensorT>>> weight_tensors_;
		std::shared_ptr<ModelErrorData<TensorT>> model_error_;
	};

	template<typename TensorT, typename DeviceT>
	ModelInterpreter<TensorT, DeviceT>::ModelInterpreter(const ModelInterpreter<TensorT, DeviceT>& other)
	{
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::mapValuesToLayers(Model<TensorT>& model, const Eigen::Tensor<TensorT, 3>& values, const std::vector<std::string>& node_names, const std::string & value_type)
	{
		// check dimension mismatches
		if (node_names.size() != values.dimension(2))
		{
			printf("The number of input features %d and the number of nodes %d do not match.\n", (int)values.dimension(2), node_names.size());
			return;
		}
		// assumes the tensors have been cached
		else if (layer_tensors_[0]->getBatchSize() != values.dimension(0))
		{
			printf("The number of input samples %d and the batch size %d does not match.\n", (int)values.dimension(0), (int)layer_tensors_[0]->getBatchSize());
			return;
		}
		else if (layer_tensors_[0]->getMemorySize() != values.dimension(1))
		{
			printf("The number of input time steps %d and the memory size %d does not match.\n", (int)values.dimension(1), (int)layer_tensors_[0]->getMemorySize());
			return;
		}

		for (int i = 0; i < node_names.size(); ++i){
			auto node = model.getNodesMap().at(node_names[i]);
			// copy over the values
			if (value_type == "output")
				getLayerTensor(node->getTensorIndex().first)->getOutput().chip(node->getTensorIndex().second, 2) = values.chip(i, 2);
			else if (value_type == "error")
				getLayerTensor(node->getTensorIndex().first)->getError().chip(node->getTensorIndex().second, 2) = values.chip(i, 2);
			else if (value_type == "derivative")
				getLayerTensor(node->getTensorIndex().first)->getDerivative().chip(node->getTensorIndex().second, 2) = values.chip(i, 2);
			else if (value_type == "dt")
				getLayerTensor(node->getTensorIndex().first)->getDt().chip(node->getTensorIndex().second, 2) = values.chip(i, 2);
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
					argument.source_node->getType(),
					argument.source_node->getIntegration()->getName(),
					argument.source_node->getActivation()->getName());
				unique_node_types.insert(ops_key);
			}
			for (const std::string& node_types : unique_node_types) {
				OperationList<TensorT> operations_list;
				operations_list.result = FP_operation.result;
				for (const OperationArguments<TensorT>& argument : FP_operation.arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.source_node->getType(),
						argument.source_node->getIntegration()->getName(),
						argument.source_node->getActivation()->getName());
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
		for (const OperationList<TensorT>& FP_operation : FP_operations) {
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
		for (const OperationList<TensorT>& FP_operation : FP_operations) {
			// check that all nodes are either cached or not yet cached into a layer
			OperationList<TensorT> operations_list_cached, operations_list;
			operations_list.result = FP_operation.result;
			operations_list_cached.result = FP_operation.result;
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
	inline std::map<std::string, std::vector<int>> ModelInterpreter<TensorT, DeviceT>::getTensorOperations(const std::vector<OperationList<TensorT>>& FP_operations, std::set<std::string>& identified_sink_nodes)
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

				// check if the source nodes are compatible
				std::set<std::string> argument_nodes;
				for (const auto& argument : FP_operations[operations_iter1].arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.source_node->getType(),
						argument.source_node->getIntegration()->getName(),
						argument.source_node->getActivation()->getName());
					argument_nodes.insert(ops_key);
				}
				for (const auto& argument : FP_operations[operations_iter2].arguments) {
					std::string ops_key = makeForwardPropogationOperationsKey(argument.time_step,
						argument.source_node->getType(),
						argument.source_node->getIntegration()->getName(),
						argument.source_node->getActivation()->getName());
					argument_nodes.insert(ops_key);
				}
				if (argument_nodes.size() > 1) continue;

				// update the maps
				std::string sink_node_key = FP_operations[operations_iter1].result.sink_node->getName() + "/" + std::to_string(operations_iter1);
				identified_sink_nodes.insert(sink_node_key);
				std::vector<int> first_operation = { (int)operations_iter1 };
				auto found = FC_layers.emplace(sink_node_key, first_operation);
				FC_layers.at(sink_node_key).push_back(operations_iter2);
			}
		}
		return FC_layers;
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::getForwardPropogationLayerTensorDimensions(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		std::vector<int>& source_layer_sizes, std::vector<int>& sink_layer_sizes, std::vector<std::vector<std::pair<int, int>>>& weight_indices, std::vector<std::vector<TensorT>>& weight_values,
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
						TensorT tmp = argument.weight->getWeightInitOp()->operator()();
						argument.weight->setWeight_(tmp);
						weight_value.push_back(tmp);
						make_weight_tensor = true;
					}
					else {
						argument.weight->addTensorIndex(std::make_tuple(weight_pos, source_layer_index, sink_layer_index));
						weight_index.push_back(std::make_pair(source_layer_index, sink_layer_index));
						weight_value.push_back(argument.weight->getWeight_());
					}
					if (increment_source_layer_size) ++source_layer_size;
				}
				if (increment_sink_layer_size) ++sink_layer_size; //?
			}

			// determine the actual source and sink layer sizes
			std::set<int> source_nodes, sink_nodes;
			for (const std::pair<int, int> p : weight_index) {
				source_nodes.insert(p.first);
				sink_nodes.insert(p.second);
			}

			// store the tensor sizes
			//sink_layer_sizes.push_back(sink_layer_size); // This is not accurate because we are actually tracking the next sink_layer position...
			//source_layer_sizes.push_back(source_layer_size); // This is not accurate because we are actually tracking the next source_layer position...
			sink_layer_sizes.push_back(sink_nodes.size());
			source_layer_sizes.push_back(source_nodes.size());
			make_source_tensors.push_back(make_source_tensor);
			make_sink_tensors.push_back(make_sink_tensor);
			make_weight_tensors.push_back(make_weight_tensor);
			weight_indices.push_back(weight_index);
			weight_values.push_back(weight_value);

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
	std::string ModelInterpreter<TensorT, DeviceT>::makeForwardPropogationOperationsKey(const int & time_step, const NodeType& node_type, const std::string & node_integration, const std::string & node_activation)
	{
		// [TODO: may not need to add in node type
		//std::string ops_key = std::to_string(time_step) + "/" + std::to_string(node_type) + "/" + node_integration + "/" + node_activation;
		std::string ops_key = std::to_string(time_step) + "/" + node_integration + "/" + node_activation;
		return ops_key;
	}

	template<typename TensorT, typename DeviceT>
	void ModelInterpreter<TensorT, DeviceT>::getForwardPropogationOperations(Model<TensorT>& model, const int& batch_size, const int& memory_size, const bool& train)
	{
		// initialize the node statuses to determine the FP propogation steps
		// [NOTE: is this needed?]
		//// initialize the input nodes to active (if not activated already)
		//for (auto& input_node : model.getInputNodes()) {
		//	input_node->setStatus(NodeStatus::activated);
		//}
		// [OR]
		for (auto& nodes_map : model.getNodesMap()) {
			if (nodes_map.second->getType() == NodeType::input || nodes_map.second->getType() == NodeType::bias)
				nodes_map.second->setStatus(NodeStatus::activated);
			else
				nodes_map.second->setStatus(NodeStatus::initialized);
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
			// re-organize into tensors with compatible source nodes, sink nodes, and weights
			std::vector<OperationList<TensorT>> FP_operations_expanded;
			expandForwardPropogationOperations(FP_operations_list, FP_operations_expanded);

			// identify tensor operation motifs
			std::set<std::string> identified_sink_nodes;
			std::map<std::string, std::vector<int>> custom_ops = getCustomOperations(FP_operations_expanded, identified_sink_nodes);
			//std::map<std::string, std::vector<int>> FC_ops = getFullyConnectedOperations(FP_operations_expanded, identified_sink_nodes);
			//std::map<std::string, std::vector<int>> SC_ops = getSinglyConnectedOperations(FP_operations_expanded, identified_sink_nodes);
			//std::map<std::string, std::vector<int>> Conv_ops = getConvOperations(FP_operations_expanded, identified_sink_nodes);
			//std::map<std::string, std::vector<int>> FIn_ops = getFanOutOperations(FP_operations_expanded, identified_sink_nodes);
			//std::map<std::string, std::vector<int>> FOut_ops = getFanInOperations(FP_operations_expanded, identified_sink_nodes);
			std::map<std::string, std::vector<int>> tensor_ops = getTensorOperations(FP_operations_expanded, identified_sink_nodes);

			// allocate memory for tensors
			if (custom_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<std::vector<TensorT>> weight_values;
				std::vector<std::vector<std::pair<int, int>>> weight_indices;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations_expanded, custom_ops, source_layer_sizes, sink_layer_sizes, weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations_expanded, custom_ops, source_layer_sizes, sink_layer_sizes, weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size, train);
			}
			if (tensor_ops.size() != 0) {
				std::vector<int> source_layer_sizes, sink_layer_sizes;
				std::vector<std::vector<TensorT>> weight_values;
				std::vector<std::vector<std::pair<int, int>>> weight_indices;
				std::vector<bool> make_source_tensors, make_sink_tensors, make_weight_tensors;
				getForwardPropogationLayerTensorDimensions(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors);
				allocateForwardPropogationLayerTensors(FP_operations_expanded, tensor_ops, source_layer_sizes, sink_layer_sizes, weight_indices, weight_values, make_source_tensors, make_sink_tensors, make_weight_tensors, batch_size, memory_size, train);
			}

			// activate sink nodes
			for (auto& FP_operation : FP_operations_list)
				FP_operation.result.sink_node->setStatus(NodeStatus::activated);
		}
	}
	
	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::addLayerTensor(NodeTensorData<TensorT>& layer)
	{
		std::shared_ptr<NodeTensorData<TensorT>> layer_ptr(std::move(&layer));
		layer_tensors_.push_back(layer_ptr);
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::clearLayerTensors()
	{
		layer_tensors_.clear();
	}

	template<typename TensorT, typename DeviceT>
	inline std::shared_ptr<NodeTensorData<TensorT>> ModelInterpreter<TensorT, DeviceT>::getLayerTensor(const int & layer_index)
	{
		return layer_tensors_.at(layer_index);
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::addWeightTensor(WeightTensorData<TensorT>& weight)
	{
		std::shared_ptr<WeightTensorData<TensorT>> weight_ptr(&weight);
		weight_tensors_.push_back(weight_ptr);
	}

	template<typename TensorT, typename DeviceT>
	inline void ModelInterpreter<TensorT, DeviceT>::clearWeightTensors()
	{
		weight_tensors_.clear();
	}

	template<typename TensorT, typename DeviceT>
	inline std::shared_ptr<WeightTensorData<TensorT>> ModelInterpreter<TensorT, DeviceT>::getWeightTensor(const int & weight_index)
	{
		return weight_tensors_.at(weight_index);
	}

	template<typename TensorT, typename DeviceT>
	inline std::shared_ptr<ModelErrorData<TensorT>> ModelInterpreter<TensorT, DeviceT>::getModelError()
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

	template<typename TensorT>
	class ModelInterpreterDefaultDevice : public ModelInterpreter<TensorT, Eigen::DefaultDevice>
	{
	public:
		using ModelInterpreter::ModelInterpreter;
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size, const bool& train);
		void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id, LossFunctionTensorOp<TensorT, Eigen::DefaultDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::DefaultDevice>* loss_function_grad, const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void allocateModelErrorTensor(const int& batch_size, const int& memory_size);

	//	void addOperationSteps(const std::vector<OperationTensorStepDefaultDevice<TensorT>>& operation_steps);
	//	std::vector<OperationTensorStepDefaultDevice<TensorT>> getOperationSteps(const int& operation_index);
	//	void clearOperationSteps();
	//protected:
	//	std::vector<std::vector<OperationTensorStepDefaultDevice<TensorT>>> operation_steps_;
	//	std::vector<std::shared_ptr<NodeTensorDataCpu<TensorT>>> layer_tensors_;
	//	std::vector<std::shared_ptr<WeightTensorDataCpu<TensorT>>> weight_tensors_;
	//	std::shared_ptr<ModelErrorDataCpu<TensorT>> model_error_;
	};

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
		const std::map<std::string, std::vector<int>>& operations_map,
		const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
		const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
		const int& batch_size, const int& memory_size, const bool& train)
	{
		std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>> operation_step_list;
		
		ActivationOpToActivationTensorOp<TensorT, Eigen::DefaultDevice> activation_conv;
		SolverOpToSolverTensorOp<TensorT, Eigen::DefaultDevice> solver_conv;
		IntegrationOpToIntegrationTensorOp<TensorT, Eigen::DefaultDevice> integration_conv;
		IntegrationErrorOpToIntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice> integration_error_conv;
		IntegrationWeightGradOpToIntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice> integration_weight_grad_conv;
		int iter = 0;
		for (const auto& operations : operations_map) {

			// make the tensors
			OperationTensorStep<TensorT, Eigen::DefaultDevice> operation_step;

			// [NOTE: order matters!  sink layer should come before the source layer to keep with
			//  the ordering generated in getForwardPropogationTensorDimensions.]
			std::shared_ptr<NodeTensorData<TensorT>> sink_node_data(new NodeTensorDataCpu<TensorT>());
			{ // make the sink layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::DefaultDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice>* integration_weight_grad = nullptr;
				if (make_sink_tensors[iter]) {
					sink_node_data->initNodeTensorData(batch_size, memory_size, sink_layer_sizes[iter], FP_operations[operations.second[0]].result.sink_node->getType(), train);
					layer_tensors_.push_back(sink_node_data);
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.sink_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.sink_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].result.sink_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.sink_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.sink_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.sink_layer.integration_weight_grad.reset(integration_weight_grad);
					operation_step.sink_layer.tensor = layer_tensors_[FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first];
				}
				else {
					operation_step.sink_layer.tensor = layer_tensors_[FP_operations[operations.second[0]].result.sink_node->getTensorIndex().first];
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.sink_layer.activation.reset(std::move(activation));
					activation_conv(FP_operations[operations.second[0]].result.sink_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.sink_layer.activation_grad.reset(std::move(activation_grad));
					integration_conv(FP_operations[operations.second[0]].result.sink_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.sink_layer.integration.reset(std::move(integration));
					integration_error_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.sink_layer.integration_error.reset(std::move(integration_error));
					integration_weight_grad_conv(FP_operations[operations.second[0]].result.sink_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.sink_layer.integration_weight_grad.reset(std::move(integration_weight_grad));
					operation_step.sink_layer.time_step = FP_operations[operations.second[0]].result.time_step;
				}
				//delete activation;
				//delete activation_grad;
				//delete integration;
				//delete integration_error;
				//delete integration_weight_grad;
			}
			
			std::shared_ptr<NodeTensorData<TensorT>> source_node_data(new NodeTensorDataCpu<TensorT>());
			{ // make the source layer tensor and add it to the cache and operation step
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation = nullptr;
				ActivationTensorOp<TensorT, Eigen::DefaultDevice>* activation_grad = nullptr;
				IntegrationTensorOp<TensorT, Eigen::DefaultDevice>* integration = nullptr;
				IntegrationErrorTensorOp<TensorT, Eigen::DefaultDevice>* integration_error = nullptr;
				IntegrationWeightGradTensorOp<TensorT, Eigen::DefaultDevice>* integration_weight_grad = nullptr;
				if (make_source_tensors[iter]) {
					source_node_data->initNodeTensorData(batch_size, memory_size, source_layer_sizes[iter], FP_operations[operations.second[0]].arguments[0].source_node->getType(), train);
					operation_step.source_layer.time_step = FP_operations[operations.second[0]].arguments[0].time_step;
					layer_tensors_.push_back(source_node_data);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.source_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.source_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.source_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.source_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.source_layer.integration_weight_grad.reset(integration_weight_grad);
					operation_step.source_layer.tensor = getLayerTensor(FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first);
				}
				else {
					operation_step.source_layer.tensor = getLayerTensor(FP_operations[operations.second[0]].arguments[0].source_node->getTensorIndex().first);
					operation_step.source_layer.time_step = FP_operations[operations.second[0]].arguments[0].time_step;
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivation(), activation, std::vector<TensorT>());
					operation_step.source_layer.activation.reset(activation);
					activation_conv(FP_operations[operations.second[0]].arguments[0].source_node->getActivationGrad(), activation_grad, std::vector<TensorT>());
					operation_step.source_layer.activation_grad.reset(activation_grad);
					integration_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegration(), integration, std::vector<TensorT>());
					operation_step.source_layer.integration.reset(integration);
					integration_error_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationError(), integration_error, std::vector<TensorT>());
					operation_step.source_layer.integration_error.reset(integration_error);
					integration_weight_grad_conv(FP_operations[operations.second[0]].arguments[0].source_node->getIntegrationWeightGrad(), integration_weight_grad, std::vector<TensorT>());
					operation_step.source_layer.integration_weight_grad.reset(integration_weight_grad);
				}
				//delete activation;
				//delete activation_grad;
				//delete integration;
				//delete integration_error;
				//delete integration_weight_grad;
			}

			// make the weight tensor and add it to the cache and operation step
			std::shared_ptr<WeightTensorData<TensorT>> weight_data(new WeightTensorDataCpu<TensorT>());
			if (make_weight_tensors[iter]) {
				SolverTensorOp<TensorT, Eigen::DefaultDevice>* solver = nullptr;
				std::vector<TensorT> solver_params;
				solver_conv(FP_operations[operations.second[0]].arguments[0].weight->getSolverOp(), solver, solver_params);
				weight_data->initWeightTensorData(source_layer_sizes[iter], sink_layer_sizes[iter], weight_indices[iter], weight_values[iter], train,
					solver_params);
				weight_tensors_.push_back(weight_data);
				operation_step.weight.tensor = weight_tensors_.at(std::get<0>(FP_operations[operations.second[0]].arguments[0].weight->getTensorIndex()[0]));
				operation_step.weight.solver.reset(solver);
				//delete solver;
			}
			else {
				std::cout << "Weight tensor is not being created...Check!" << std::endl;
			}

			operation_step_list.push_back(operation_step);
			++iter;
		}
		// add the operations to the cache
		operation_steps_.push_back(operation_step_list);
	}

	template<typename TensorT>
	inline void ModelInterpreterDefaultDevice<TensorT>::executeForwardPropogationOperations(const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : operation_steps_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operations_list) {
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
		for (size_t i = operation_steps_.size() - 1; i >= 0; --i) { //iterate backwards
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operation_steps_[i]) { //reverse source/sink

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
		OperationTensorStep<TensorT, Eigen::DefaultDevice> operation = operation_steps_[layer_id.first][layer_id.second];
		model_kernal.executeModelErrors(
			expected,
			operation.sink_layer.tensor->getHOutputPointer().get(),
			operation.sink_layer.tensor->getDOutputPointer().get(),
			model_error_->getHErrorPointer().get(),
			model_error_->getDErrorPointer().get(),
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
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : operation_steps_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operations_list) {

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
		for (std::vector<OperationTensorStep<TensorT, Eigen::DefaultDevice>>& operations_list : operation_steps_) {
			ModelKernalDefaultDevice<TensorT> model_kernal;
			Eigen::DefaultDevice device;

			// execute the forward propogation steps
			for (OperationTensorStep<TensorT, Eigen::DefaultDevice>& operation : operations_list) {

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
	inline void ModelInterpreterDefaultDevice<TensorT>::allocateModelErrorTensor(const int& batch_size, const int& memory_size) {
		std::shared_ptr<ModelErrorData<TensorT>> model_error_data(new ModelErrorDataCpu<TensorT>());
		model_error_data->initModelErrorData(batch_size, memory_size);
		model_error_ = model_error_data;
	}

	//template<typename TensorT>
	//inline void ModelInterpreterDefaultDevice<TensorT>::addLayerTensor(NodeTensorDataCpu<TensorT>& layer)
	//{
	//	std::shared_ptr<NodeTensorData<TensorT>> layer_ptr(std::move(&layer));
	//	layer_tensors_.push_back(layer_ptr);
	//}

	//template<typename TensorT>
	//inline void ModelInterpreterDefaultDevice<TensorT>::clearLayerTensors()
	//{
	//	layer_tensors_.clear();
	//}

	//template<typename TensorT>
	//inline std::shared_ptr<NodeTensorDataCpu<TensorT>> ModelInterpreterDefaultDevice<TensorT>::getLayerTensor(const int & layer_index)
	//{
	//	return layer_tensors_.at(layer_index);
	//}

	//template<typename TensorT>
	//inline void ModelInterpreterDefaultDevice<TensorT>::addWeightTensor(WeightTensorDataCpu<TensorT>& weight)
	//{
	//	std::shared_ptr<WeightTensorDataCpu<TensorT>> weight_ptr(&weight);
	//	weight_tensors_.push_back(weight_ptr);
	//}

	//template<typename TensorT>
	//inline void ModelInterpreterDefaultDevice<TensorT>::clearWeightTensors()
	//{
	//	weight_tensors_.clear();
	//}

	//template<typename TensorT>
	//inline std::shared_ptr<WeightTensorDataCpu<TensorT>> ModelInterpreterDefaultDevice<TensorT>::getWeightTensor(const int & weight_index)
	//{
	//	return weight_tensors_[weight_index];
	//}

	//template<typename TensorT>
	//inline std::shared_ptr<ModelErrorDataCpu<TensorT>> ModelInterpreterDefaultDevice<TensorT>::getModelError()
	//{
	//	return model_error_;
	//}

	//template<typename TensorT>
	//inline void ModelInterpreterDefaultDevice<TensorT>::addOperationSteps(const std::vector<OperationTensorStepDefaultDevice<TensorT>>& operation_steps) {
	//	operation_steps_.push_back(operation_steps);
	//}

	//template<typename TensorT>
	//inline std::vector<OperationTensorStepDefaultDevice<TensorT>> ModelInterpreterDefaultDevice<TensorT>::getOperationSteps(const int& operation_index) {
	//	return operation_steps_.at(operation_index);
	//}

	//template<typename TensorT>
	//inline void ModelInterpreterDefaultDevice<TensorT>::clearOperationSteps()
	//{
	//	operation_steps_.clear();
	//}

#if COMPILE_WITH_CUDA
	template<typename TensorT>
	class ModelInterpreterGpu : public ModelInterpreter<TensorT, Eigen::GpuDevice>
	{
	public:
		void allocateForwardPropogationLayerTensors(const std::vector<OperationList<TensorT>>& FP_operations,
			const std::map<std::string, std::vector<int>>& operations_map,
			const std::vector<int>& source_layer_sizes, const std::vector<int>& sink_layer_sizes, const std::vector<std::vector<std::pair<int, int>>> weight_indices, const std::vector<std::vector<TensorT>>& weight_values,
			const std::vector<bool>& make_source_tensors, const std::vector<bool>& make_sink_tensors, const std::vector<bool>& make_weight_tensors,
			const int& batch_size, const int& memory_size, const bool& train);
		void executeForwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeModelErrorOperations(Eigen::Tensor<TensorT, 2>& expected, const std::pair<int, int>& layer_id, LossFunctionTensorOp<TensorT, Eigen::GpuDevice>* loss_function, LossFunctionGradTensorOp<TensorT, Eigen::GpuDevice>* loss_function_grad, const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeBackwardPropogationOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightErrorOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void executeWeightUpdateOperations(const int& time_step, bool sync_HToD = false, bool sync_DToH = false);
		void allocateModelErrorTensor(const int& batch_size, const int& memory_size);
		void addOperationSteps(const std::vector<OperationTensorStepGpu<TensorT>>& operation_steps);
		void clearOperationSteps();
	protected:
		std::vector<std::vector<OperationTensorStepGpu<TensorT>>> operation_steps_;
		ModelErrorDataGpu<TensorT> model_error_;
	};

	template<typename TensorT>
	void ModelInterpreterGpu<TensorT>::executeForwardPropogationOperations(const int& time_step, bool sync_HToD, bool sync_DToH)
	{
		for (auto& operations_list : operation_steps_) {

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
		for (size_t iter = operation_steps_.size() - 1; iter >= 0; --iter) { //iterate backwards

			// Set up the device, streams, and kernals
			ModelKernalGpu<TensorT> model_kernal;
			const int device_id = 0;
			assert(cudaSetDevice(device_id) == cudaSuccess); // is this needed?
			std::vector<cudaStream_t> streams;
			std::vector<Eigen::GpuStreamDevice> stream_devices;
			std::vector<Eigen::GpuDevice> devices;
			for (size_t i = 0; i < operation_steps_[iter].size(); ++i) {
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
			for (OperationTensorStepGpu<TensorT>& operation : operation_steps_[iter]) { //reverse source/sink

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
			for (size_t i = 0; i < operation_steps_[iter].size(); ++i) {
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

		OperationTensorStepDefaultDevice<TensorT> operation = operation_steps_[layer_id.first][layer_id.second];
		model_kernal.executeModelErrors(
			expected,
			operation.sink_layer.tensor->getHOutputPointer().get(),
			operation.sink_layer.tensor->getDOutputPointer().get(),
			model_error_->getHErrorPointer().get(),
			model_error_->getDErrorPointer().get(),
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
		for (std::vector<OperationTensorStepDefaultDevice<TensorT>>& operations_list : operation_steps_) {

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
		for (std::vector<OperationTensorStepDefaultDevice<TensorT>>& operations_list : operation_steps_) {

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
	inline void ModelInterpreterGpu<TensorT>::addOperationSteps(const std::vector<OperationTensorStepGpu<TensorT>>& operation_steps) {
		operations_steps_.push_back(operation_steps);
	}

	template<typename TensorT>
	inline void ModelInterpreterGpu<TensorT>::allocateModelErrorTensor(const int& batch_size, const int& memory_size) {
		std::shared_ptr<ModelErrorData<TensorT>> model_error_data(new ModelErrorDataGpu<TensorT>());
		model_error_data->initModelErrorData(batch_size, memory_size);
		model_error_ = model_error_data;
	}

	template<typename TensorT>
	void ModelInterpreterGpu<TensorT>::clearOperationSteps()
	{
		operation_steps_.clear();
	}
#endif

}
#endif //SMARTPEAK_MODELINTERPRETER_H