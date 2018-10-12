/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

// .h
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/LossFunction.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <tuple>
#include <list>

// .cpp
#include <SmartPeak/ml/SharedFunctions.h>
#include <SmartPeak/graph/CircuitFinder.h>

#include <iostream>
#include <algorithm>
#include <thread>
#include <future>
#include <mutex>

static std::mutex calculateNetNodeInput_mutex;
static std::mutex calculateNodeInput_mutex;
static std::mutex calculateNetNodeError_mutex;
static std::mutex calculateNodeError_mutex;
static std::mutex calculateModelError_mutex;
static std::mutex calculateOutputNodeError_mutex;

namespace SmartPeak
{

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

  /**
    @brief Directed Network Model

    Assumptions about the model structure:
    1. Inputs can only be sources
    2. Outputs can only be sinks (will break back propogation algorithm)
  */
	template<typename TensorT>
  class Model
  {
public:
    Model() = default; ///< Default constructor
    Model(const Model& other); ///< Copy constructor that does not create a shared memory address between model nodes/links/weights
    Model(const int& id); ///< Explicit constructor  
    ~Model() = default; ///< Default destructor

    inline bool operator==(const Model& other) const
    {
      return
        std::tie(
          id_,
          name_,
          links_,
          nodes_,
          weights_
        ) == std::tie(
          other.id_,
          other.name_,
          other.links_,
          other.nodes_,
          other.weights_
        )
      ;
    }

    inline bool operator!=(const Model& other) const
    {
      return !(*this == other);
    }


		/**
		@brief Copy assignment operator that creates a new model with different memory addresses
		*/
    inline Model& operator=(const Model& other)
    {
      id_ = other.id_;
      name_ = other.name_;
      links_ = other.links_;
      nodes_ = other.nodes_;
      weights_ = other.weights_;
      error_ = other.error_;
      loss_function_ = other.loss_function_;
			loss_function_grad_ = other.loss_function_grad_;
			cyclic_pairs_ = other.cyclic_pairs_;
      return *this;
    }

    /**
      @brief Initialize all link weights
    */ 
    void initWeights();

		/**
			@brief Initialize all link weights dropout probability

			[TODO: add tests]
			[TODO: implement sampling from a Gaussian distribution during interence]
		*/
		void initWeightsDropProbability(bool train = false);

    /**
      @brief Initialize all node output to zero.
        The node statuses are then changed to NodeStatus::deactivated

      @param[in] batch_size Batch size of the output, error, and derivative node vectors
      @param[in] memory_size Memory size of the output, error, and derivative node vectors

			[TODO: implement sampling from a Gaussian distribution during interence]
    */ 
    void initNodes(const int& batch_size, const int& memory_size, bool train = false);

		/**
		@brief Initialize model errors to zero.

		@param[in] batch_size Batch size of the output, error, and derivative node vectors
		@param[in] memory_size Memory size of the output, error, and derivative node vectors
		*/
		void initError(const int& batch_size, const int& memory_size);

		/**
		@brief Infer the batch_size and memory_size.

		@return a pair of batch_size and memory_size
		*/
		std::pair<int, int> getBatchAndMemorySizes() const;

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
    void mapValuesToNodes(
      const Eigen::Tensor<TensorT, 3>& values,
      const std::vector<std::string>& node_names,
      const NodeStatus& status_update,
      const std::string& value_type);

    /**
      @brief Assigns output or error values to the nodes at a specific
        place in memory.
        The node statuses are then changed accordingly (i.e.,
        status_update of "activated" will update the output values
        of the node and status_update of "corrected" will update
        the error values of the node.

      dimensions of batch size by nodes

      @param[in] values Values to assign to the node
      @param[in] memory_step The memory step to add values to 
      @param[in] node_names 
      @param[in] status_update
      @param[in] value_type String of "output", "derivative", or "error"
    */ 
    void mapValuesToNodes(
      const Eigen::Tensor<TensorT, 2>& values,
      const int& memory_step,
      const std::vector<std::string>& node_names,
      const NodeStatus& status_update,
      const std::string& value_type);

    /**
      @brief Assigns output or error values to a single node at a specific
        place in memory.

      [TODO: replace chip with index assignment
        w/ chip: 1/1 Test #16: PopulationTrainer_test ...........   Passed  766.58 sec
        w/o chip: 1/1 Test #16: PopulationTrainer_test ...........   Passed   54.16 sec
      ]

      dimensions of batch size

      @param[in] values Values to assign to the node
      @param[in] memory_step The memory step to add values to 
      @param[in] node_name
      @param[in] status_update
      @param[in] value_type String of "output", "derivative", or "error"
    */ 
    void mapValuesToNode(
      const Eigen::Tensor<TensorT, 1>& values,
      const int& memory_step,
      const std::string& node_name,
      const NodeStatus& status_update,
      const std::string& value_type);

    /**
      @brief Assigns output or error values to all nodes at a specific
        place in memory.
        The node statuses are also updated according to status_update.

      dimensions of batch size by nodes

      @param[in] values Values to assign to the node
      @param[in] memory_step The memory step to add values to 
      @param[in] status_update
      @param[in] value_type String of "output", "derivative", or "error"
    */       
    void mapValuesToNodes(
      const Eigen::Tensor<TensorT, 1>& values,
      const int& memory_step,
      const NodeStatus& status_update,
      const std::string& value_type);
 
    /**
      @brief A prelude to a forward propogation step. Returns a vector of links
        and associated nodes that satisfy the following conditions:
        1. all sink output values are unknown (i.e. inactive),
        2. all source node output values are known (i.e. active).
        3. all nodes need not be the same type

			@param[out] FP_operations_map Key/Value pair of sink node name to FP_peroations index
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
    @brief Allocate tensor memory for all forward
      propogation tensors.

    Note that nodes need not be the same type.

    @param[in] time_step Time step to activate.
    */
    void allocateForwardPropogationLayerTensors(const int& time_step);
 
    /**
    @brief A prelude to a forward propogation step. Computes the net
      input into all nodes composing the next layer:
      1. all sink output values are unknown (i.e. inactive),
      2. all source node output values are known (i.e. active).

    Note that nodes need not be the same type.

    @param[in] FP_operations
    @param[in] time_step Time step to activate.

    [OPTIMIZATION:
      pass memory to tensors so that when the tensors compute the matrices
      the underlying node values are automatically updated?]
    [PARALLEL: allow for parallelization of iteration of sink nodes]
    [THREADPOOL/CUDA: move to seperate file for cpu/cuda compilation]
    */ 
    void forwardPropogateLayerNetInput(
      std::vector<OperationList<TensorT>>& FP_operations,
      const int& time_step, int n_threads = 1);

    static bool calculateNodeInput_(
			OperationResult<TensorT>* result,
      OperationArguments<TensorT>* arguments,
      const int& batch_size,
      const int& memory_size,
      const int& time_step
    );
    static bool calculateNetNodeInput_(
      OperationList<TensorT>* operations,
      const int& batch_size,
      const int& memory_size,
      const int& time_step,
      int n_threads = 1
    );
 
    /**
    @brief Foward propogation of the network model.
      All node outputs and derivatives are calculating
      starting from the input nodes.  Each node status is
      changed from "initialized" to "activated" when the
      outputs and derivatives are calculated.
    
    [TODO: add tests for caching]

    @param[in] time_step Time step to forward propogate.
    @param[in] cache_FP_steps Whether to save the FP steps
      for faster iteration next epoch.
    @param[in] use_cache Whether to use the cached FP steps.
    */ 
    void forwardPropogate(const int& time_step, bool cache_FP_steps = false, bool use_cache = false,
      int n_threads = 1);
				
		static std::string makeFPOpsKey(const std::string& node_name, const int& time_step,
			const std::string& node_integration, const std::string& node_activation);
		void getFPOperations();
		void convertFPOpsToTensorOps();
		void executeFPOperations(const int& time_step);
 
    /**
    @brief Foward propogation through time (FPTT) of the network model.
      All node outputs and derivatives are calculating
      starting from the input nodes.  Each node status is
      changed from "initialized" to "activated" when the
      outputs and derivatives are calculated.  This is repeated
      for n_time steps without weight updates.
    
    NOTE: The implementation assumes that the output values for
      all input and biases have already been set.  

    @param[in] time_steps The number of time_steps forward to 
      continuously calculate node outputs and node derivatives.
    @param[in] values Input values at each time step where
      dim0: batch_size, dim1: time_step, and dim2: nodes.
    @param[in] node_names
    @param[in] dt Node time resolution 
    */ 
    void FPTT(const int& time_steps, 
      const Eigen::Tensor<TensorT, 3>& values,
      const std::vector<std::string> node_names,
      const Eigen::Tensor<TensorT, 2>& dt, 
      bool cache_FP_steps = false, 
      bool use_cache = false,
      int n_threads = 1);
 
    /**
    @brief Calculates the error of the model with respect to
      the expected values

    @param[in] values Expected node output values
    @param[in] node_names Output nodes
    */ 
    void calculateError(const Eigen::Tensor<TensorT, 2>& values, const std::vector<std::string>& node_names,
			const int& time_step, int n_threads = 1);

		/**
		@brief Calculates the error of the model for a given node
		*/
		static Eigen::Tensor<TensorT, 1> calculateModelError_(
			Node<TensorT>* output_node,
			const Eigen::Tensor<TensorT, 1>& expected,
			LossFunctionOp<TensorT>* loss_function,
			const int& batch_size,
			const int& time_step
			);

		/**
		@brief Calculates the error of the output node
		*/
		static bool calculateOutputNodeError_(
			Node<TensorT>* output_node,
			const Eigen::Tensor<TensorT, 1>& expected,
			LossFunctionGradOp<TensorT>* loss_function_grad,
			const int& time_step
			);
 
    /**
    @brief Calculates the error of the model through time (CETT)
      with respect to the expected values

    @param[in] values Expected node output values
			(dim0: batch_size, dim1: memory_size, dim2: output nodes)
			where t=n to t=0
    @param[in] node_names Output nodes
    */ 
    void CETT(const Eigen::Tensor<TensorT, 3>& values, const std::vector<std::string>& node_names, const int& time_steps, int n_threads = 1);
 
    /**
    @brief A prelude to a back propogation step.  Returns a vector of links
      and associated nodes that satisfy the following conditions:
      1. all sink error values are unknown (i.e. active),
      2. all source error values are known (i.e. corrected).
      3. all nodes need not be the same type

    @param[out] BP_operatations_map Key/Value pair of source nodes (sink nodes in BP) to index in BP_operations list
		@param[out] BP_operations Operations list for Back Propogation
    @param[out] sink_nodes
    */ 
    void getNextUncorrectedLayer(
      std::map<std::string, int>& BP_operations_map,
      std::vector<OperationList<TensorT>>& BP_operations,
      std::vector<std::string>& source_nodes);

		void getNextUncorrectedLayerBiases(
			std::map<std::string, int>& BP_operations_map, 
			std::vector<OperationList<TensorT>>& BP_operations,
			std::vector<std::string>& source_nodes, 
			std::vector<std::string>& sink_nodes_with_biases);
 
    /**
    @brief A continuation of a back propogation step.  Returns a vector of links
      and associated nodes that satisfy the following conditions:
      1. all sink error values are known (i.e. corrected),
      2. all source error values are known (i.e. corrected).
      3. all nodes need not be the same type

		@param[out] BP_operatations_map Key/Value pair of source nodes (sink nodes in BP) to index in BP_operations list
		@param[out] BP_operations Operations list for Back Propogation
    @param[out] source_nodes
    @param[out] source_nodes_with_cycles
    */ 
    void getNextUncorrectedLayerCycles(
      std::map<std::string, int>& BP_operations_map,
      std::vector<OperationList<TensorT>>& BP_operations,
      std::vector<std::string>& source_nodes,
      std::vector<std::string>& source_nodes_with_cycles);
 
    /**
    @brief A back propogation step. Computes the net
      error into all nodes composing the next layer:
      1. all sink error values are unknown (i.e. active),
      2. all source error values are known (i.e. corrected).

    Note that nodes need not be the same type.

		@param[out] BP_operations Operations list for Back Propogation
		@param[in] step to forward propogate.

    [OPTIMIZATION:
    pass memory to tensors so that when the tensors compute the matrices
    the underlying node values are automatically updated]
    */ 
    void backPropogateLayerError(
      std::vector<OperationList<TensorT>>& BP_operations,
      const int& time_step, int n_threads = 1);

    static bool calculateNodeError_(
			OperationResult<TensorT>* operations,
      OperationArguments<TensorT>* arguments,
      const int& batch_size,
      const int& memory_size,
      const int& time_step
    );
    static bool calculateNetNodeError_(
      OperationList<TensorT>* operations,
      const int& batch_size,
      const int& memory_size,
      const int& time_step,
      int n_threads = 1
    );
 
    /**
    @brief Back propogation of the network model.
      All node errors are calculating starting from the output nodes.  
      Each node status is changed from "activated" to "corrected" when the
      outputs and derivatives are calculated.

    [TODO: add tests for caching]

    @param[in] time_step Time step to forward propogate.
    @param[in] cache_BP_steps Whether to save the BP steps
      for faster iteration next epoch.
    @param[in] use_cache Whether to use the cached BP steps.

    @returns Vector of cyclic sink node IDs
    */ 
    void backPropogate(const int& time_step, bool cache_BP_steps = false, bool use_cache = false, int n_threads = 1);  
 
    /**
    @brief Truncated Back Propogation Through Time (TBPTT) of the network model.
      All node errors are calculating starting from the output nodes.  
      Each node status is changed from "activated" to "corrected" when the
      outputs and derivatives are calculated.

    @param[in] time_steps The number of time_steps backwards to 
      unfold the network model.
    */ 
    void TBPTT(const int& time_steps, bool cache_FP_steps = false, bool use_cache = false, int n_threads = 1);  
 
    /**
    @brief Recurrent Real Time Learning (RTRL) of the network model.
      All node errors are calculating starting from the output nodes.  
      Each node status is changed from "activated" to "corrected" when the
      outputs and derivatives are calculated.

    @param[in] time_steps The number of time_steps backwards to 
      unfold the network model.
    */ 
    void RTRL(const int& time_steps, int n_threads = 1);  
 
    /**
    @brief Update the weights
      
    */ 
		void updateWeights(const int& time_steps, std::vector<std::string> weight_names = {});
 
    /**
    @brief Reset the node statuses back to inactivated
      
    */ 
    void reInitializeNodeStatuses();

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< name setter
    std::string getName() const; ///< name getter

    void setError(const Eigen::Tensor<TensorT, 2>& error); ///< error setter
    Eigen::Tensor<TensorT, 2> getError() const; ///< error getter

    void setLossFunction(const std::shared_ptr<LossFunctionOp<TensorT>>& loss_function); ///< loss_function setter
    LossFunctionOp<TensorT>* getLossFunction() const; ///< loss_function getter

		void setLossFunctionGrad(const std::shared_ptr<LossFunctionGradOp<TensorT>>& loss_function); ///< loss_function grad setter
		LossFunctionGradOp<TensorT>* getLossFunctionGrad() const; ///< loss_function grad getter

		std::vector<std::shared_ptr<Node<TensorT>>> getInputNodes(); ///< input_node getter
		std::vector<std::shared_ptr<Node<TensorT>>> getOutputNodes(); ///< output_node getter
		std::vector<std::string> getOutputNodeNames() const;
 
    /**
      @brief Add new links to the model.

      @param[in] links Links to add to the model
    */ 
    void addLinks(const std::vector<Link>& links);
    Link getLink(const std::string& link_name) const; ///< link getter
    std::vector<Link> getLinks() const;  ///< links getter
 
    /**
      @brief Remove existing links from the model.

      @param[in] Link_names Links to remove from the model
    */ 
    void removeLinks(const std::vector<std::string>& link_names);
 
    /**
      @brief Add new nodes to the model.

      @param[in] nodes Nodes to add to the model
    */ 
    void addNodes(const std::vector<Node<TensorT>>& nodes);
    Node<TensorT> getNode(const std::string& node_name) const; ///< node getter
    std::vector<Node<TensorT>> getNodes() const; ///< nodes getter
		std::map<std::string, std::shared_ptr<Node<TensorT>>> getNodesMap();  ///< return a modifiable version of weights
		std::map<std::string, std::vector<std::string>> getModuleNodeNameMap() const; ///< return a map of modules to a vector of node names [TODO: test!]

    /**
      @brief Remove existing nodes from the model.

      @param[in] node_names Nodes to remove from the model
    */ 
    void removeNodes(const std::vector<std::string>& node_names);
 
    /**
      @brief Add new weights to the model.

      @param[in] weights Weights to add to the model
    */ 
    void addWeights(const std::vector<Weight<TensorT>>& weights);
    Weight<TensorT> getWeight(const std::string& weight_name) const; ///< weight getter
    std::vector<Weight<TensorT>> getWeights() const;  ///< weights getter
		std::map<std::string, std::shared_ptr<Weight<TensorT>>> getWeightsMap();  ///< return a modifiable version of weights_		
 
    /**
      @brief Remove existing weights from the model.

      @param[in] weight_names Weights to remove from the model
    */ 
    void removeWeights(const std::vector<std::string>& weight_names);
 
    /**
      @brief Removes nodes from the model that no longer
        have an associated link.

      @returns True if nodes were removed, False otherwise
    */ 
    bool pruneNodes();
 
    /**
      @brief Removes links from the model that no longer
        have associated nodes.

      @returns True if links were removed, False otherwise
    */ 
    bool pruneLinks();    
 
    /**
      @brief Removes weights from the model that no longer
        have associated links.

      @returns True if weights were removed, False otherwise
    */ 
    bool pruneWeights(); 
 
    /**
      @brief Removes dangling links, weights, and nodes 
        recursively until there are no more dangling
        model components or the number of user specified
        iterations has been reached.

      @param[in] iterations The number of recursive iterations to prune
    */ 
    void pruneModel(int iterations = 1e3); 
 
    /**
      @brief Check to ensure that the nodes are in the model

      @param[in] node_names 
    */ 
    bool checkNodeNames(const std::vector<std::string> node_names); 
 
    /**
      @brief Check to ensure that the links are in the model

      @param[in] link_names 
    */ 
    bool checkLinkNames(const std::vector<std::string> link_names); 
 
    /**
      @brief Check to ensure that the weights are in the model

      @param[in] weight_names 
    */ 
    bool checkWeightNames(const std::vector<std::string> weight_names);

		/**
		@brief Check that the path from input to output is not broken

		[DEPRECATED: params no longer needed]
		@param[in] input_nodes
		@param[out] output_nodes
		*/
		bool checkCompleteInputToOutput(int n_threads = 1);

		/**
		@brief Check model link node and weight names

		[TODO: add tests...]

		@param[out] nodes_not_found
		@param[out] weights_not_found
		*/
		bool checkLinksNodeAndWeightNames(
			std::vector<std::string>& nodes_not_found,
			std::vector<std::string>& weights_not_found);

		/**
		@brief Remove hidden nodes that have either only 1 source and no sink connection
			or 1 sink and no source connection
		*/
		bool removeIsolatedNodes();

    void clearCache(); ///< clear the FP and BP caches

		/**
		@brief Convert model to adjacency list
		TODO: Implement tests

		@param[out] node_id_map Map of node id to node name
		@param[out] node_cnt the number of vertices in the adjacency list

		@returns An adjacency list representation of a graph
		*/
		std::list<int>* convertToAdjacencyList(std::map<int, std::string>& node_id_map, int& node_cnt);
		void findCycles();

		std::vector<std::pair<std::string, std::string>> getCyclicPairs();

private:
    int id_; ///< Model ID
    std::string name_; ///< Model Name
    std::map<std::string, std::shared_ptr<Link>> links_; ///< Model links
    std::map<std::string, std::shared_ptr<Node<TensorT>>> nodes_; ///< Model nodes
    std::map<std::string, std::shared_ptr<Weight<TensorT>>> weights_; ///< Model nodes
    Eigen::Tensor<TensorT, 2> error_; ///< Model error
    std::shared_ptr<LossFunctionOp<TensorT>> loss_function_; ///< Model loss function
		std::shared_ptr<LossFunctionGradOp<TensorT>> loss_function_grad_; ///< Model loss function
		std::vector<std::pair<std::string, std::string>> cyclic_pairs_;
		std::vector<std::shared_ptr<Node<TensorT>>> input_nodes_;
		std::vector<std::shared_ptr<Node<TensorT>>> output_nodes_;

    // Internal structures to allow for efficient multi-threading
    // and off-loading of computation from host to devices
    std::vector<std::vector<OperationList<TensorT>>> FP_operations_cache_;
		std::vector<std::pair<int, int>> FP_operations_dimensions_;  // vector of source/sink node sizes
    std::vector<std::vector<OperationList<TensorT>>> BP_operations_cache_;
  };
	template<typename TensorT>
	Model<TensorT>::Model(const Model<TensorT>& other)
	{
		id_ = other.id_;
		name_ = other.name_;
		addLinks(other.getLinks());
		addNodes(other.getNodes());
		addWeights(other.getWeights());
		error_ = other.error_;
		loss_function_ = other.loss_function_;
		loss_function_grad_ = other.loss_function_grad_;
		cyclic_pairs_ = other.cyclic_pairs_;
	}

	template<typename TensorT>
	Model<TensorT>::Model(const int& id) :
		id_(id)
	{
	}

	template<typename TensorT>
	void Model<TensorT>::setId(const int& id)
	{
		id_ = id;
	}
	template<typename TensorT>
	int Model<TensorT>::getId() const
	{
		return id_;
	}

	template<typename TensorT>
	void Model<TensorT>::setName(const std::string& name)
	{
		name_ = name;
	}
	template<typename TensorT>
	std::string Model<TensorT>::getName() const
	{
		return name_;
	}

	template<typename TensorT>
	void Model<TensorT>::setError(const Eigen::Tensor<TensorT, 2>& error)
	{
		error_ = error;
	}
	template<typename TensorT>
	Eigen::Tensor<TensorT, 2> Model<TensorT>::getError() const
	{
		return error_;
	}

	template<typename TensorT>
	void Model<TensorT>::setLossFunction(const std::shared_ptr<LossFunctionOp<TensorT>>& loss_function)
	{
		loss_function_.reset();
		loss_function_ = std::move(loss_function);
	}
	template<typename TensorT>
	LossFunctionOp<TensorT>* Model<TensorT>::getLossFunction() const
	{
		return loss_function_.get();
	}

	template<typename TensorT>
	void Model<TensorT>::setLossFunctionGrad(const std::shared_ptr<LossFunctionGradOp<TensorT>>& loss_function)
	{
		loss_function_grad_.reset();
		loss_function_grad_ = std::move(loss_function);
	}

	template<typename TensorT>
	LossFunctionGradOp<TensorT>* Model<TensorT>::getLossFunctionGrad() const
	{
		return loss_function_grad_.get();
	}

	template<typename TensorT>
	std::vector<std::shared_ptr<Node<TensorT>>> Model<TensorT>::getInputNodes()
	{
		return input_nodes_;
	}

	template<typename TensorT>
	std::vector<std::shared_ptr<Node<TensorT>>> Model<TensorT>::getOutputNodes()
	{
		return output_nodes_;
	}
	template<typename TensorT>
	std::vector<std::string> Model<TensorT>::getOutputNodeNames() const
	{
		std::vector<std::string> nodes;
		for (const auto& node : output_nodes_)
		{
			nodes.push_back(node->getName());
		}
		return nodes;
	}

	template<typename TensorT>
	void Model<TensorT>::addNodes(const std::vector<Node<TensorT>>& nodes)
	{
		for (const Node<TensorT>& node : nodes)
		{
			std::shared_ptr<Node<TensorT>> node_ptr;
			node_ptr.reset(new Node<TensorT>(node));
			auto found = nodes_.emplace(node.getName(), node_ptr);
			if (!found.second)
			{
				// TODO: move to debug log
				std::cout << "Node name " << node.getName() << " already exists!" << std::endl;
			}
			else {
				if (node.getType() == NodeType::input) {
					std::shared_ptr<Node<TensorT>> node_ptr_cpy = node_ptr;
					input_nodes_.push_back(node_ptr_cpy);
				}
				else if (node.getType() == NodeType::output) {
					std::shared_ptr<Node<TensorT>> node_ptr_cpy = node_ptr;
					output_nodes_.push_back(node_ptr);
				}
			}
		}
	}

	template<typename TensorT>
	Node<TensorT> Model<TensorT>::getNode(const std::string& node_name) const
	{
		if (!nodes_.empty() && nodes_.count(node_name) != 0)
		{
			return *nodes_.at(node_name);
		}
		else
		{
			// TODO: move to debug log
			std::cout << "Node name " << node_name << " not found!" << std::endl;
		}
	}

	template<typename TensorT>
	std::vector<Node<TensorT>> Model<TensorT>::getNodes() const
	{
		std::vector<Node<TensorT>> nodes;
		for (const auto& node : nodes_)
		{
			nodes.push_back(*node.second);
		}
		return nodes;
	}

	template<typename TensorT>
	std::map<std::string, std::shared_ptr<Node<TensorT>>> Model<TensorT>::getNodesMap()
	{
		return nodes_;
	}

	template<typename TensorT>
	std::map<std::string, std::vector<std::string>> Model<TensorT>::getModuleNodeNameMap() const
	{
		std::map<std::string, std::vector<std::string>> module_to_node_names;
		for (const auto& node_map : nodes_) {
			std::vector<std::string> node_names = { node_map.first };
			auto found = module_to_node_names.emplace(node_map.second->getModuleName(), node_names);
			if (!found.second) {
				module_to_node_names.at(node_map.second->getModuleName()).push_back(node_map.first);
			}
		}
		return module_to_node_names;
	}

	template<typename TensorT>
	void Model<TensorT>::removeNodes(const std::vector<std::string>& node_names)
	{
		for (const std::string& node_name : node_names)
		{
			// check for duplicate nodes (by id)
			if (nodes_.count(node_name) != 0)
			{
				nodes_.erase(node_name);
			}
		}
		// pruneLinks(); // Allow for dangling links
	}

	template<typename TensorT>
	void Model<TensorT>::addWeights(const std::vector<Weight<TensorT>>& weights)
	{
		for (const Weight<TensorT>& weight : weights)
		{
			std::shared_ptr<Weight<TensorT>> weight_ptr;
			weight_ptr.reset(new Weight<TensorT>(weight));
			auto found = weights_.emplace(weight.getName(), weight_ptr);
			if (!found.second)
			{
				// TODO: move to debug log
				std::cout << "Weight name " << weight.getName() << " already exists!" << std::endl;
			}
		}
	}

	template<typename TensorT>
	Weight<TensorT> Model<TensorT>::getWeight(const std::string& weight_name) const
	{
		if (!weights_.empty() && weights_.count(weight_name) != 0)
		{
			//return *std::move(weights_.at(weight_name));
			return *weights_.at(weight_name);
		}
		else
		{
			// TODO: move to debug log
			std::cout << "Weight name " << weight_name << " not found!" << std::endl;
		}
	}

	template<typename TensorT>
	std::vector<Weight<TensorT>> Model<TensorT>::getWeights() const
	{
		std::vector<Weight<TensorT>> weights;
		for (const auto& weight : weights_)
		{
			weights.push_back(*weight.second);
		}
		return weights;
	}

	template<typename TensorT>
	std::map<std::string, std::shared_ptr<Weight<TensorT>>> Model<TensorT>::getWeightsMap()
	{
		return weights_;
	}

	template<typename TensorT>
	void Model<TensorT>::removeWeights(const std::vector<std::string>& weight_names)
	{
		for (std::string const& weight_name : weight_names)
		{
			// check for duplicate weights (by id)
			if (weights_.count(weight_name) != 0)
			{
				weights_.erase(weight_name);
			}
		}
		pruneLinks();
	}

	template<typename TensorT>
	void Model<TensorT>::addLinks(const std::vector<Link>& links)
	{
		for (const Link& link : links)
		{
			std::shared_ptr<Link> link_ptr;
			link_ptr.reset(new Link(link));
			auto found = links_.emplace(link.getName(), link_ptr);
			if (!found.second)
			{
				// TODO: move to debug log
				std::cout << "Link name " << link.getName() << " already exists!" << std::endl;
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::removeLinks(const std::vector<std::string>& link_names)
	{
		for (const std::string& link_name : link_names)
		{
			// check for duplicate links (by id)
			if (links_.count(link_name) != 0)
			{
				links_.erase(link_name);
			}
		}
		// pruneNodes(); // Allow dangling nodes to exist
		//pruneWeights();  // testing
	}

	template<typename TensorT>
	Link Model<TensorT>::getLink(const std::string& link_name) const
	{
		if (!links_.empty() && links_.count(link_name) != 0)
		{
			return *links_.at(link_name);
		}
		else
		{
			// TODO: move to debug log
			std::cout << "Link name " << link_name << " not found!" << std::endl;
		}
	}

	template<typename TensorT>
	std::vector<Link> Model<TensorT>::getLinks() const
	{
		std::vector<Link> links;
		for (const auto& link : links_)
		{
			links.push_back(*link.second);
		}
		return links;
	}

	template<typename TensorT>
	bool Model<TensorT>::pruneNodes()
	{
		std::vector<std::string> node_names;
		if (nodes_.empty()) { return false; }
		for (const auto& node : nodes_)
		{
			bool found = false;
			// if (links_.empty()) { found = true; }
			for (const auto& link : links_)
			{
				if (node.second->getName() == link.second->getSourceNodeName() ||
					node.second->getName() == link.second->getSinkNodeName())
				{
					found = true;
					break;
				}
			}
			if (!found)
			{
				node_names.push_back(node.first);
			}
		}
		if (node_names.size() != 0)
		{
			removeNodes(node_names);
			return true;
		}
		else
			return false;
	}

	template<typename TensorT>
	bool Model<TensorT>::pruneWeights()
	{
		std::vector<std::string> weight_names;
		if (weights_.empty()) { return false; }
		for (const auto& weight : weights_)
		{
			bool found = false;
			// if (links_.empty()) { found = true; }
			for (const auto& link : links_)
			{
				if (weight.second->getName() == link.second->getWeightName())
				{
					found = true;
					break;
				}
			}
			if (!found)
			{
				weight_names.push_back(weight.first);
			}
		}
		if (weight_names.size() != 0)
		{
			removeWeights(weight_names);
			return true;
		}
		else
			return false;
	}

	template<typename TensorT>
	bool Model<TensorT>::pruneLinks()
	{
		std::vector<std::string> link_names;
		if (links_.empty()) { return false; }
		for (const auto& link : links_)
		{
			bool source_node_found = false;
			bool sink_node_found = false;
			// if (nodes_.empty())
			// {
			//   source_node_found = true;
			//   sink_node_found = true;
			// }
			for (const auto& node : nodes_)
			{
				if (node.second->getName() == link.second->getSourceNodeName())
					source_node_found = true;
				if (node.second->getName() == link.second->getSinkNodeName())
					sink_node_found = true;
				if (source_node_found && sink_node_found)
					break;
			}
			bool weight_found = false;
			// if (weights_.empty()) { weight_found = true; }
			for (const auto& weight : weights_)
			{
				if (weight.second->getName() == link.second->getWeightName())
				{
					weight_found = true;
					break;
				}
			}
			if (!source_node_found || !sink_node_found)
			{
				link_names.push_back(link.first);
			}
		}
		if (link_names.size() != 0)
		{
			removeLinks(link_names);
			return true;
		}
		else
			return false;
	}

	template<typename TensorT>
	void Model<TensorT>::pruneModel(int iterations)
	{
		try
		{
			int cnt = 0;
			while (pruneLinks() || pruneWeights() || pruneNodes())
			{
				if (cnt >= iterations) { break; }
				// std::cout<<"Pruning model iteration: "<<cnt<<std::endl;
				cnt += 1;
			}
		}
		catch (std::exception& e)
		{
			printf("Exception: %s", e.what());
		}
	}

	template<typename TensorT>
	void Model<TensorT>::initNodes(const int& batch_size, const int& memory_size, bool train)
	{
		for (auto& node_map : nodes_)
		{
			node_map.second->initNode(batch_size, memory_size + 1, train); // +1 to ensure we stay within the allocated bounds of the tensor during F and BPTT
		}
	}

	template<typename TensorT>
	void Model<TensorT>::initError(const int & batch_size, const int & memory_size)
	{
		Eigen::Tensor<TensorT, 2> init_values(batch_size, memory_size);
		init_values.setConstant(0.0f);
		setError(init_values);
	}

	template<typename TensorT>
	std::pair<int, int> Model<TensorT>::getBatchAndMemorySizes() const
	{
		int batch_size = 0;
		int memory_size = 0;
		for (const auto& node : nodes_) {
			batch_size = node.second->getBatchSize();
			memory_size = node.second->getMemorySize();
			break;
		}
		return std::make_pair(batch_size, memory_size);
	}

	template<typename TensorT>
	void Model<TensorT>::initWeights()
	{
		for (auto& weight_map : weights_)
		{
			weight_map.second->initWeight();
		}
	}

	template<typename TensorT>
	void Model<TensorT>::initWeightsDropProbability(bool train)
	{
		if (train)
			for (auto& weight_map : weights_)
				weight_map.second->setDropProbability(weight_map.second->getDropProbability());
		else
			for (auto& weight_map : weights_)
				weight_map.second->setDrop(1.0f);
	}

	template<typename TensorT>
	void Model<TensorT>::mapValuesToNodes(
		const Eigen::Tensor<TensorT, 1>& values,
		const int& memory_step,
		const NodeStatus& status_update,
		const std::string& value_type)
	{

		// copy over the input values
		for (auto& node_map : nodes_)
		{
			for (int j = 0; j < values.dimension(0); ++j)
			{
				if (value_type == "output")
				{
					node_map.second->getOutput()(j, memory_step) = values(j);
				}
				else if (value_type == "error")
				{
					node_map.second->getError()(j, memory_step) = values(j);
				}
				else if (value_type == "dt")
				{
					node_map.second->getDt()(j, memory_step) = values(j);
				}
				if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
					node_map.second->setStatus(status_update);
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::mapValuesToNodes(
		const Eigen::Tensor<TensorT, 2>& values,
		const int& memory_step,
		const std::vector<std::string>& node_names,
		const NodeStatus& status_update,
		const std::string& value_type)
	{
		// check dimension mismatches
		if (node_names.size() != values.dimension(1))
		{
			std::cout << "The number of input features and the number of nodes do not match." << std::endl;
			return;
		}
		// assumes the node exists
		else if (nodes_.at(node_names[0])->getOutput().dimension(0) != values.dimension(0))
		{
			std::cout << "The number of input samples and the node batch size does not match." << std::endl;
			return;
		}
		// assumes the node exists
		else if (nodes_.at(node_names[0])->getOutput().dimension(1) <= memory_step)
		{
			std::cout << "The memory_step is greater than the memory_size." << std::endl;
			return;
		}

		// // infer the memory size from the node output size
		// const int memory_size = nodes_.at(node_names[0])->getOutput().dimension(1);

		// copy over the input values
		for (int i = 0; i < node_names.size(); ++i)
		{
			for (int j = 0; j < values.dimension(0); ++j)
			{
				if (value_type == "output")
				{
					nodes_.at(node_names[i])->getOutput()(j, memory_step) = values(j, i);
				}
				else if (value_type == "error")
				{
					nodes_.at(node_names[i])->getError()(j, memory_step) = values(j, i);
				}
				else if (value_type == "dt")
				{
					nodes_.at(node_names[i])->getDt()(j, memory_step) = values(j, i);
				}
				if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
					nodes_.at(node_names[i])->setStatus(status_update);
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::mapValuesToNode(
		const Eigen::Tensor<TensorT, 1>& values,
		const int& memory_step,
		const std::string& node_name,
		const NodeStatus& status_update,
		const std::string& value_type)
	{
		// check dimension mismatches
		// assumes the node exists
		if (nodes_.at(node_name)->getOutput().dimension(0) != values.dimension(0))
		{
			std::cout << "The number of input samples and the node batch size does not match." << std::endl;
			return;
		}

		// // copy over the input values
		// for (int j=0; j<values.dimension(0); ++j)
		// {
		//   if (value_type == "output")
		//   {
		//     nodes_.at(node_name)->getOutput()->operator()(j, memory_step) = values(j);
		//   }
		//   else if (value_type == "error")
		//   {
		//     nodes_.at(node_name)->getError()->operator()(j, memory_step) = values(j);
		//   }
		//   else if (value_type == "derivative")
		//   {
		//     nodes_.at(node_name)->getDerivative()->operator()(j, memory_step) = values(j);
		//   }
		//   else if (value_type == "dt")
		//   {
		//     nodes_.at(node_name)->getDt()->operator()(j, memory_step) = values(j);
		//   }
		// }

		// copy over the input values
		if (value_type == "output")
		{
			nodes_.at(node_name)->getOutput().chip(memory_step, 1) = values;
		}
		else if (value_type == "error")
		{
			nodes_.at(node_name)->getError().chip(memory_step, 1) = values;
		}
		else if (value_type == "derivative")
		{
			nodes_.at(node_name)->getDerivative().chip(memory_step, 1) = values;
		}
		else if (value_type == "dt")
		{
			nodes_.at(node_name)->getDt().chip(memory_step, 1) = values;
		}

		// update the status
		if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
			nodes_.at(node_name)->setStatus(status_update);
	}

	template<typename TensorT>
	void Model<TensorT>::mapValuesToNodes(
		const Eigen::Tensor<TensorT, 3>& values,
		const std::vector<std::string>& node_names,
		const NodeStatus& status_update,
		const std::string& value_type)
	{
		// check dimension mismatches
		if (node_names.size() != values.dimension(2))
		{
			printf("The number of input features %d and the number of nodes %d do not match.\n", (int)values.dimension(2), node_names.size());
			return;
		}
		// assumes the node exists
		else if (nodes_.at(node_names[0])->getOutput().dimension(0) != values.dimension(0))
		{
			printf("The number of input samples %d and the node batch size %d does not match.\n", (int)values.dimension(0), (int)nodes_.at(node_names[0])->getOutput().dimension(0));
			return;
		}
		else if (nodes_.at(node_names[0])->getOutput().dimension(1) != values.dimension(1))
		{
			printf("The number of input time steps %d and the node memory size %d does not match.\n", (int)values.dimension(1), (int)nodes_.at(node_names[0])->getOutput().dimension(1));
			return;
		}

		// copy over the input values
		for (int i = 0; i < node_names.size(); ++i)
		{
			for (int k = 0; k < values.dimension(1); ++k)
			{
				for (int j = 0; j < values.dimension(0); ++j)
				{
					if (value_type == "output")
					{
						// nodes_.at(node_names[i])->getOutputPointer()[k*values.dimension(0) + j] = values(j, k, i);
						nodes_.at(node_names[i])->getOutput()(j, k) = values(j, k, i);
					}
					else if (value_type == "error")
					{
						nodes_.at(node_names[i])->getError()(j, k) = values(j, k, i);
					}
					else if (value_type == "derivative")
					{
						nodes_.at(node_names[i])->getDerivative()(j, k) = values(j, k, i);
					}
					else if (value_type == "dt")
					{
						nodes_.at(node_names[i])->getDt()(j, k) = values(j, k, i);
					}
					if (status_update != NodeStatus::deactivated) // [TESTS:  add tests]
						nodes_.at(node_names[i])->setStatus(status_update);
				}
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::getNextInactiveLayer(
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
				//std::cout<<"Link source node name: "<< link_map.second->getSourceNodeName() <<std::endl
				arguments.source_node = nodes_.at(link_map.second->getSourceNodeName());
				//std::cout << "Link weight name: " << link_map.second->getWeightName() << std::endl;
				arguments.weight = weights_.at(link_map.second->getWeightName());
				arguments.time_step = 0;
				arguments.link_name = link_map.first;

				// std::cout<<"Addres of model source node: "<<&nodes_.at(link_map.second->getSourceNodeName())<<std::endl;
				// std::cout<<"Addres of arguments source node: "<<arguments.source_node<<std::endl;

				std::string ops_key = makeFPOpsKey(link_map.second->getSinkNodeName(), 0,
					nodes_.at(link_map.second->getSinkNodeName())->getIntegration()->getName(),
					nodes_.at(link_map.second->getSinkNodeName())->getActivation()->getName());
				auto found = FP_operations_map.emplace(ops_key, (int)FP_operations.size());
				if (!found.second)
				{
					FP_operations[FP_operations_map.at(link_map.second->getSinkNodeName())].arguments.push_back(arguments);
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

	template<typename TensorT>
	void Model<TensorT>::getNextInactiveLayerBiases(
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations,
		std::vector<std::string>& sink_nodes_with_biases)
	{

		// get all the biases for the sink nodes
		for (auto& link_map : links_)
		{
			std::string ops_key = makeFPOpsKey(link_map.second->getSinkNodeName(), 0,
				nodes_.at(link_map.second->getSinkNodeName())->getIntegration()->getName(),
				nodes_.at(link_map.second->getSinkNodeName())->getActivation()->getName());
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

	template<typename TensorT>
	void Model<TensorT>::getNextInactiveLayerCycles(
		std::map<std::string, int>& FP_operations_map,
		std::vector<OperationList<TensorT>>& FP_operations,
		std::vector<std::string>& sink_nodes_with_cycles)
	{

		// get cyclic source nodes
		for (auto& link_map : links_)
		{
			std::string ops_key = makeFPOpsKey(link_map.second->getSinkNodeName(), 0,
				nodes_.at(link_map.second->getSinkNodeName())->getIntegration()->getName(),
				nodes_.at(link_map.second->getSinkNodeName())->getActivation()->getName());
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

				// [PARRALLEL] can we check that we will not over exceed the memory
				//             and take appropriate measures here?
				// e.g.
				// memory_size = arguments.source_node->getOutput().dimension(1);
				// if (time_step + 1 >= memory_size) ...
				arguments.time_step = 1;
				arguments.link_name = link_map.first;
				FP_operations[FP_operations_map.at(ops_key)].arguments.push_back(arguments);
				sink_nodes_with_cycles.push_back(ops_key);
			}
		}
	}

	template<typename TensorT>
	bool Model<TensorT>::calculateNodeInput_(
		OperationResult<TensorT>* result,
		OperationArguments<TensorT>* arguments,
		const int& batch_size,
		const int& memory_size,
		const int& time_step)
	{
		std::lock_guard<std::mutex> lock(calculateNodeInput_mutex);

		Eigen::Tensor<TensorT, 1> weight_tensor(batch_size);
		weight_tensor.setConstant(arguments->weight->getWeight());
		//if (arguments->time_step == 0 || time_step + arguments->time_step < memory_size)
		//{
		result->sink_node->getIntegrationShared()->operator()(weight_tensor, arguments->source_node->getOutput().chip(time_step + arguments->time_step, 1));
		//}
		return true;
	}

	template<typename TensorT>
	bool Model<TensorT>::calculateNetNodeInput_(
		OperationList<TensorT>* operations,
		const int& batch_size,
		const int& memory_size,
		const int& time_step,
		int n_threads)
	{
		std::lock_guard<std::mutex> lock(calculateNetNodeInput_mutex);

		std::vector<std::future<bool>> task_results;
		operations->result.sink_node->getIntegrationShared()->initNetNodeInput(batch_size);
		int thread_cnt = 0;

		// for (const std::string& link : sink_links)
		for (int i = 0; i < operations->arguments.size(); ++i)
		{
			std::packaged_task<bool // encapsulate in a packaged_task
			(OperationResult<TensorT>*, OperationArguments<TensorT>*, int, int, int
				)> task(Model<TensorT>::calculateNodeInput_);

			// launch the thread
			task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				&operations->result, &operations->arguments[i], std::ref(batch_size), std::ref(memory_size), std::ref(time_step));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || i == operations->arguments.size() - 1)
			{
				for (auto& task_result : task_results)
				{
					if (task_result.valid())
					{
						try
						{
							bool result = task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				++thread_cnt;
			}
		}

		// calculate the output and the derivative
		Eigen::Tensor<TensorT, 1> output = calculateActivation<TensorT>(
			operations->result.sink_node->getActivationShared().get(), operations->result.sink_node->getIntegrationShared()->getNetNodeInput(),
			operations->result.sink_node->getDt().chip(time_step, 1),
			1);
		Eigen::Tensor<TensorT, 1> derivative = calculateDerivative<TensorT>(
			operations->result.sink_node->getActivationGradShared().get(), output, 1);

		operations->result.sink_node->setStatus(NodeStatus::activated);
		operations->result.sink_node->getInput().chip(time_step, 1) = operations->result.sink_node->getIntegrationShared()->getNetNodeInput();
		operations->result.sink_node->getOutput().chip(time_step, 1) = output;
		operations->result.sink_node->getDerivative().chip(time_step, 1) = derivative;

		return true;
	}

	template<typename TensorT>
	void Model<TensorT>::forwardPropogateLayerNetInput(
		std::vector<OperationList<TensorT>>& FP_operations,
		const int& time_step, int n_threads)
	{

		// get all the information needed to construct the tensors
		std::pair<int, int> bmsizes = getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second;

		// iterate through each sink node and calculate the net input
		// invoke the activation function once the net input is calculated
		std::vector<std::future<bool>> task_results;
		int thread_cnt = 0;
		const int threads_per_sub_process = 1; // [TODO: how to best divide up the allowable threads?]
		int operations_cnt = 0;
		for (auto& FP_operation : FP_operations)
		{
			std::packaged_task<bool // encapsulate in a packaged_task
			(OperationList<TensorT>*, int, int, int, int
				)> task(Model<TensorT>::calculateNetNodeInput_);

			// launch the thread
			task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				&FP_operation, std::ref(batch_size), std::ref(memory_size), std::ref(time_step),
				std::ref(threads_per_sub_process));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || operations_cnt == FP_operations.size() - 1)
			{
				for (auto& task_result : task_results)
				{
					if (task_result.valid())
					{
						try
						{
							bool success = task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				thread_cnt += threads_per_sub_process;
			}
			// std::cout<<"thread_count"<<thread_cnt<<std::endl;
			// std::cout<<"operations_cnt"<<operations_cnt<<std::endl;
			++operations_cnt;
		}
	}

	template<typename TensorT>
	void Model<TensorT>::forwardPropogate(const int& time_step, bool cache_FP_steps, bool use_cache, int n_threads)
	{
		if (use_cache)
		{
			for (auto& FP_operations : FP_operations_cache_)
				forwardPropogateLayerNetInput(FP_operations, time_step, n_threads);
		}
		else
		{
			const int max_iters = 1e6;
			for (int iter = 0; iter < max_iters; ++iter)
			{
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
				if (sink_nodes_cycles.size() > 0)
				{
					std::vector<std::string> sink_nodes_remove;
					std::vector<OperationList<TensorT>> FP_operations_list_copy = FP_operations_list;
					for (const std::string& sink_node : sink_nodes_cycles) {
						for (size_t i = FP_operations_list[FP_operations_map.at(sink_node)].arguments.size();
							i < FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].arguments.size(); ++i) {
							// check if the "cyclic" argument is actually involved in a cycle
							bool isCyclicOperation = false;
							for (const auto& cyclic_pair : cyclic_pairs_) {
								if (FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].arguments[i].source_node->getName() == cyclic_pair.first &&
									FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].result.sink_node->getName() == cyclic_pair.second) {
									isCyclicOperation = true;
									break;
								}
							}
							// copy over the cyclic operation
							if (isCyclicOperation)
								FP_operations_list_copy[FP_operations_map_cycles.at(sink_node)].arguments.push_back(FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].arguments[i]);
							// id the sink node for removal
							else {
								sink_nodes_remove.push_back(sink_node);
								break;
							}
						}
					}
					// remove all identified sink nodes
					if (sink_nodes_remove.size() > 0) {
						FP_operations_list.clear();
						for (const auto& FP_operation : FP_operations_list_copy)
							if (std::count(sink_nodes_remove.begin(), sink_nodes_remove.end(), FP_operation.result.sink_node->getName()) == 0)
								FP_operations_list.push_back(FP_operation);
					}
					else
						FP_operations_list = FP_operations_list_copy;
				}

				// check if all nodes have been activated
				if (FP_operations_list.size() == 0)
				{
					break;
				}

				if (cache_FP_steps)
					FP_operations_cache_.push_back(FP_operations_list);

				// calculate the net input
				forwardPropogateLayerNetInput(FP_operations_list, time_step, n_threads);
			}
		}
	}

	template<typename TensorT>
	std::string Model<TensorT>::makeFPOpsKey(const std::string & node_name, const int & time_step, const std::string & node_integration, const std::string & node_activation)
	{
		std::string ops_key = node_name + "/" + std::to_string(time_step) + "/" + node_integration + "/" + node_activation;
		return ops_key;
	}

	template<typename TensorT>
	void Model<TensorT>::getFPOperations()
	{
		const int max_iters = 1e6;
		for (int iter = 0; iter < max_iters; ++iter)
		{
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
			if (sink_nodes_cycles.size() > 0)
			{
				std::vector<std::string> sink_nodes_remove;
				std::vector<OperationList<TensorT>> FP_operations_list_copy = FP_operations_list;
				for (const std::string& sink_node : sink_nodes_cycles) {
					for (size_t i = FP_operations_list[FP_operations_map.at(sink_node)].arguments.size();
						i < FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].arguments.size(); ++i) {
						// check if the "cyclic" argument is actually involved in a cycle
						bool isCyclicOperation = false;
						for (const auto& cyclic_pair : cyclic_pairs_) {
							if (FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].arguments[i].source_node->getName() == cyclic_pair.first &&
								FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].result.sink_node->getName() == cyclic_pair.second) {
								isCyclicOperation = true;
								break;
							}
						}
						// copy over the cyclic operation
						if (isCyclicOperation)
							FP_operations_list_copy[FP_operations_map_cycles.at(sink_node)].arguments.push_back(FP_operations_list_cycles[FP_operations_map_cycles.at(sink_node)].arguments[i]);
						// id the sink node for removal
						else {
							sink_nodes_remove.push_back(sink_node);
							break;
						}
					}
				}
				// remove all identified sink nodes
				if (sink_nodes_remove.size() > 0) {
					FP_operations_list.clear();
					for (const auto& FP_operation : FP_operations_list_copy)
						if (std::count(sink_nodes_remove.begin(), sink_nodes_remove.end(), FP_operation.result.sink_node->getName()) == 0)
							FP_operations_list.push_back(FP_operation);
				}
				else
					FP_operations_list = FP_operations_list_copy;
			}

			// check if all nodes have been activated
			if (FP_operations_list.size() == 0)
			{
				break;
			}

			// add operations to the cache
			FP_operations_cache_.push_back(FP_operations_list);
			int source_nodes = 0;
			for (auto& FP_operation : FP_operations_list)
				source_nodes += FP_operation.arguments.size();
			FP_operations_dimensions_.push_back(std::make_pair(source_nodes, FP_operations_list.size()));

			// activate sink nodes
			for (auto& FP_operation : FP_operations_list)
				FP_operation.result.sink_node->setStatus(NodeStatus::activated);
		}
	}

	template<typename TensorT>
	void Model<TensorT>::executeFPOperations(const int& time_step)
	{
		// get all the information needed to construct the tensors
		std::pair<int, int> bmsizes = getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second;

		// pre-allocate all host and device tensor memory

		int FP_operations_cnt = 0;
		for (auto& FP_operations : FP_operations_cache_) {
			// Special case if all FP_operations with sink node of Sum IntegrationType and the same activation function
			//Eigen::Tensor<TensorT, 2> source_tensor(batch_size, FP_operations_dimensions_[FP_operations_cnt].first);
			//Eigen::Tensor<TensorT, 2> weight_tensor(FP_operations_dimensions_[FP_operations_cnt].first, FP_operations_dimensions_[FP_operations_cnt].second);
			//Eigen::Tensor<TensorT, 2> sink_tensor_output(batch_size, FP_operations_dimensions_[FP_operations_cnt].second);
			//Eigen::Tensor<TensorT, 2> sink_tensor_derivative(batch_size, FP_operations_dimensions_[FP_operations_cnt].second);

			for (auto& FP_operation : FP_operations) {
				// Create the source, weight, and sink tensors
				Eigen::Tensor<TensorT, 2> source_tensor(batch_size, FP_operation.arguments.size());
				Eigen::Tensor<TensorT, 2> weight_tensor(batch_size, FP_operation.arguments.size());
				Eigen::Tensor<TensorT, 1> sink_tensor_output(batch_size);
				Eigen::Tensor<TensorT, 1> sink_tensor_derivative(batch_size);
				int arguments_cnt = 0;
				for (auto& FP_argument : FP_operation.arguments) {
					// Fill the source and weight tensors
					source_tensor.chip(arguments_cnt, 1) = FP_argument.source_node->getOutput()->chip(time_step + FP_argument.time_step, 1);
					weight_tensor.chip(arguments_cnt, 0).setConstant(FP_argument.weight->getWeight());
					++arguments_cnt;
				}
				// Offload memory to device
				// executeFPOpsDevice(source_tensor, weight_tensor, sink_tensor, derivative_tensor);
				// sink_tensor.device(...) = 

				// Execute operations (i.e., integration, activation, and derivative)

				// Retrieve results

				// Update sink nodes
				FP_operation.result.sink_node->getOutput()->chip(time_step + FP_operation.result.time_step, 1) = sink_tensor_output;
				FP_operation.result.sink_node->getDerivative()->chip(time_step + FP_operation.result.time_step, 1) = sink_tensor_output;
			}
			++FP_operations_cnt;
		}
	}

	template<typename TensorT>
	void Model<TensorT>::FPTT(const int& time_steps,
		const Eigen::Tensor<TensorT, 3>& values,
		const std::vector<std::string> node_names,
		const Eigen::Tensor<TensorT, 2>& dt,
		bool cache_FP_steps, bool use_cache, int n_threads)
	{
		// check time_steps vs memory_size
		int max_steps = time_steps;
		if (time_steps >= nodes_.begin()->second->getOutput().dimension(1))
		{
			std::cout << "Time_steps will be scaled back to the memory_size - 1." << std::endl;
			max_steps = nodes_.begin()->second->getOutput().dimension(1) - 1;
		}

		for (int time_step = 0; time_step < max_steps; ++time_step)
		{
			const int time_step_cur = max_steps - 1 - time_step;

			// std::cout<<"Model<TensorT>::FPTT() time_step: "<<time_step<<std::endl;
			if (time_step > 0)
			{
				// move to the next memory step
				for (auto& node_map : nodes_)
				{
					if (std::count(node_names.begin(), node_names.end(), node_map.first) == 0)
					{
						node_map.second->setStatus(NodeStatus::initialized); // reinitialize non-input nodes
					}
					// std::cout<<"Model<TensorT>::FPTT() output: "<<node_map.second->getOutput()<<" for node_name: "<<node_map.first<<std::endl;
				}
			}

			// initialize nodes for the next time-step
			const Eigen::Tensor<TensorT, 1> dt_values = dt.chip(time_step, 1);
			mapValuesToNodes(dt_values, time_step_cur, NodeStatus::deactivated, "dt"); // [TESTS: setting this to "initialized" caused one hell of a headache to debug...]
			const Eigen::Tensor<TensorT, 2> active_values = values.chip(time_step, 1);
			//std::cout<<"Model<TensorT>::FPTT() active_values: "<<active_values<<std::endl;
			mapValuesToNodes(active_values, time_step_cur, node_names, NodeStatus::activated, "output");

			if (cache_FP_steps && time_step == 0)
				forwardPropogate(time_step_cur, true, false, n_threads);
			else if (cache_FP_steps && time_step > 0)
				forwardPropogate(time_step_cur, false, true, n_threads);
			else
				forwardPropogate(time_step_cur, cache_FP_steps, use_cache, n_threads); // always working at the current head of memory
		}
	}

	template<typename TensorT>
	Eigen::Tensor<TensorT, 1> Model<TensorT>::calculateModelError_(
		Node<TensorT>* output_node,
		const Eigen::Tensor<TensorT, 1>& expected,
		LossFunctionOp<TensorT>* loss_function,
		const int& batch_size,
		const int& time_step
	) {
		std::lock_guard<std::mutex> lock(calculateModelError_mutex);

		Eigen::Tensor<TensorT, 1> model_error(batch_size);
		model_error = loss_function->operator()(output_node->getOutput().chip(time_step, 1), expected);
		return model_error;
	};

	template<typename TensorT>
	bool Model<TensorT>::calculateOutputNodeError_(
		Node<TensorT>* output_node,
		const Eigen::Tensor<TensorT, 1>& expected,
		LossFunctionGradOp<TensorT>* loss_function_grad,
		const int& time_step
	) {
		std::lock_guard<std::mutex> lock(calculateOutputNodeError_mutex);

		output_node->getError().chip(time_step, 1) += loss_function_grad->operator()(
			output_node->getOutput().chip(time_step, 1), expected) *
			output_node->getDerivative().chip(time_step, 1);
		//output_node->setStatus(NodeStatus::corrected); // corrected status will be updated in CETT based on the tagged NodeType
		return true;
	};

	template<typename TensorT>
	void Model<TensorT>::calculateError(
		const Eigen::Tensor<TensorT, 2>& values, const std::vector<std::string>& node_names,
		const int& time_step, int n_threads)
	{
		// infer the batch size from the first source node
		std::pair<int, int> bmsizes = getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second;

		//TODO: encapsulate into a seperate method
		// check dimension mismatches
		if (node_names.size() != values.dimension(1))
		{
			std::cout << "The number of output features and the number of nodes do not match." << std::endl;
			return;
		}
		// assumes the node exists
		else if (batch_size != values.dimension(0))
		{
			std::cout << "The number of output samples and the node batch size does not match." << std::endl;
			return;
		}

		// collect the loss functions
		std::shared_ptr<LossFunctionOp<TensorT>> loss_function = loss_function_;
		std::shared_ptr<LossFunctionGradOp<TensorT>> loss_function_grad = loss_function_grad_;

		// collect the output nodes
		std::vector<std::shared_ptr<Node<TensorT>>> output_nodes;
		for (int i = 0; i < node_names.size(); ++i)
		{
			std::shared_ptr<Node<TensorT>> output_node = nodes_.at(node_names[i]);
			output_nodes.push_back(output_node);
		}

		// loop over all nodes and calculate the error for the model
		std::vector<std::future<Eigen::Tensor<TensorT, 1>>> model_error_task_results;
		Eigen::Tensor<TensorT, 1> model_error(batch_size);
		model_error.setConstant(0.0f);
		int thread_cnt = 0;
		for (int i = 0; i < node_names.size(); ++i)
		{
			// encapsulate in a packaged_task
			std::packaged_task<Eigen::Tensor<TensorT, 1>
				(Node<TensorT>*, Eigen::Tensor<TensorT, 1>, LossFunctionOp<TensorT>*,
					int, int
					)> task(Model<TensorT>::calculateModelError_);

			// launch the thread
			model_error_task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				output_nodes[i].get(), values.chip(i, 1), loss_function.get(), std::ref(batch_size), std::ref(time_step));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || i == node_names.size() - 1)
			{
				for (auto& task_result : model_error_task_results)
				{
					if (task_result.valid())
					{
						try
						{
							model_error += task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				model_error_task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				++thread_cnt;
			}
		}
		error_.chip(time_step, 1) += model_error; // add on the model_error

		// loop over all nodes and calculate the error for the nodes
		std::vector<std::future<bool>> output_node_error_task_results;
		thread_cnt = 0;
		for (int i = 0; i < node_names.size(); ++i)
		{
			// encapsulate in a packaged_task
			std::packaged_task<bool
			(Node<TensorT>*, Eigen::Tensor<TensorT, 1>, LossFunctionGradOp<TensorT>*,
				int
				)> task(Model<TensorT>::calculateOutputNodeError_);

			// launch the thread
			output_node_error_task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				output_nodes[i].get(), values.chip(i, 1), loss_function_grad.get(), std::ref(time_step));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || i == node_names.size() - 1)
			{
				for (auto& task_result : output_node_error_task_results)
				{
					if (task_result.valid())
					{
						try
						{
							bool result = task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				output_node_error_task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				++thread_cnt;
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::CETT(const Eigen::Tensor<TensorT, 3>& values, const std::vector<std::string>& node_names, const int & time_steps, int n_threads)
	{
		// check time_steps vs memory_size
		// [NOTE: was changed form memory_size to memory_size - 1]
		int max_steps = time_steps;
		if (time_steps >= nodes_.begin()->second->getOutput().dimension(1))
		{
			std::cout << "Time_steps will be scaled back to the memory_size - 1." << std::endl;
			max_steps = nodes_.begin()->second->getOutput().dimension(1) - 1;
		}

		if (values.dimension(1) - 1 > nodes_.begin()->second->getOutput().dimension(1))
			std::cout << "The sequence for CETT needs to be the memory_size - 1!" << std::endl;;

		// NOTE: the output are stored [Tmax, Tmax - 1, ..., T=0, T=-1]
		//	     while the expected output (values) are stored [T=0, T=1, ..., Tmax, Tmax]
		for (int i = 0; i < max_steps; ++i)
		{
			int next_time_step = values.dimension(1) - 1 - i;
			// [TESTS: Test for the expected output error at each time step]
			//std::cout<<"Expected output for time point "<< i << " is " << values.chip(next_time_step, 1)<<std::endl;

			// calculate the error for each batch of memory
			calculateError(values.chip(next_time_step, 1), node_names, i, n_threads);
			//calculateError(values.chip(i, 1), node_names, i);

			// set the output nodes as corrected
			for (auto& node : output_nodes_)
				node->setStatus(NodeStatus::corrected);
		}
	}

	template<typename TensorT>
	void Model<TensorT>::getNextUncorrectedLayer(
		std::map<std::string, int>& BP_operations_map,
		std::vector<OperationList<TensorT>>& BP_operations,
		std::vector<std::string>& source_nodes)
	{
		// get all links where the source node is corrected and the sink node is active
		// including biases
		for (auto& link_map : links_)
		{
			if (nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected &&
				nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = nodes_.at(link_map.second->getSinkNodeName());
				arguments.weight = weights_.at(link_map.second->getWeightName());
				arguments.time_step = 0;
				arguments.link_name = link_map.first;

				// std::cout<<"Addres of model source node: "<<&nodes_.at(link_map.second->getSourceNodeName())<<std::endl;
				// std::cout<<"Addres of arguments source node: "<<arguments.source_node<<std::endl;

				auto found = BP_operations_map.emplace(link_map.second->getSourceNodeName(), (int)BP_operations.size());
				if (!found.second)
				{
					BP_operations[BP_operations_map.at(link_map.second->getSourceNodeName())].arguments.push_back(arguments);
				}
				else
				{
					OperationList<TensorT> operation_list;
					OperationResult<TensorT> result;
					result.sink_node = nodes_.at(link_map.second->getSourceNodeName());
					operation_list.result = result;
					operation_list.arguments.push_back(arguments);
					BP_operations.push_back(operation_list);
				}

				if (std::count(source_nodes.begin(), source_nodes.end(), link_map.second->getSinkNodeName()) == 0)
				{
					source_nodes.push_back(link_map.second->getSinkNodeName());
				}
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::getNextUncorrectedLayerBiases(
		std::map<std::string, int>& BP_operations_map,
		std::vector<OperationList<TensorT>>& BP_operations,
		std::vector<std::string>& source_nodes,
		std::vector<std::string>& sink_nodes_with_biases)
	{

		// allows for cycles
		for (auto& link_map : links_)
		{
			if (nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::activated &&
				nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::activated &&
				BP_operations_map.count(link_map.second->getSourceNodeName()) != 0 // sink node has already been identified
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = nodes_.at(link_map.second->getSinkNodeName());
				arguments.weight = weights_.at(link_map.second->getWeightName());
				arguments.time_step = 0;
				arguments.link_name = link_map.first;
				BP_operations[BP_operations_map.at(link_map.second->getSourceNodeName())].arguments.push_back(arguments);

				// [TODO: update name to sink_nodes...
				if (std::count(sink_nodes_with_biases.begin(), sink_nodes_with_biases.end(), link_map.second->getSourceNodeName()) == 0)
				{
					sink_nodes_with_biases.push_back(link_map.second->getSourceNodeName());
				}
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::getNextUncorrectedLayerCycles(
		std::map<std::string, int>& BP_operations_map,
		std::vector<OperationList<TensorT>>& BP_operations,
		std::vector<std::string>& source_nodes,
		std::vector<std::string>& sink_nodes_with_cycles)
	{

		// allows for cycles
		for (auto& link_map : links_)
		{
			bool isCyclicOperation = false;
			for (const auto& cyclic_pair : cyclic_pairs_) {
				if (link_map.second->getSourceNodeName() == cyclic_pair.first &&
					link_map.second->getSinkNodeName() == cyclic_pair.second) {
					isCyclicOperation = true;
					break;
				}
			}
			if (isCyclicOperation &&
				nodes_.at(link_map.second->getSourceNodeName())->getStatus() == NodeStatus::corrected &&
				nodes_.at(link_map.second->getSinkNodeName())->getStatus() == NodeStatus::corrected
				)
			{
				OperationArguments<TensorT> arguments;
				arguments.source_node = nodes_.at(link_map.second->getSinkNodeName());
				arguments.weight = weights_.at(link_map.second->getWeightName());
				arguments.time_step = 0;
				arguments.link_name = link_map.first;

				auto found = BP_operations_map.emplace(link_map.second->getSourceNodeName(), (int)BP_operations.size());
				if (!found.second)
				{
					BP_operations[BP_operations_map.at(link_map.second->getSourceNodeName())].arguments.push_back(arguments);
				}
				else
				{
					OperationList<TensorT> operation_list;
					OperationResult<TensorT> result;
					result.sink_node = nodes_.at(link_map.second->getSourceNodeName());
					result.time_step = 1;
					operation_list.result = result;
					operation_list.arguments.push_back(arguments);
					BP_operations.push_back(operation_list);
				}

				if (std::count(sink_nodes_with_cycles.begin(), sink_nodes_with_cycles.end(), link_map.second->getSourceNodeName()) == 0)
				{
					sink_nodes_with_cycles.push_back(link_map.second->getSourceNodeName());
				}
			}
		}
	}

	template<typename TensorT>
	bool Model<TensorT>::calculateNodeError_(
		OperationResult<TensorT>* result,
		OperationArguments<TensorT>* arguments,
		const int& batch_size,
		const int& memory_size,
		const int& time_step)
	{
		std::lock_guard<std::mutex> lock(calculateNodeError_mutex);

		Eigen::Tensor<TensorT, 1> weight_tensor(batch_size);
		weight_tensor.setConstant(arguments->weight->getWeight());
		Eigen::Tensor<TensorT, 1> n_input_nodes(arguments->source_node->getOutput().dimension(0));
		n_input_nodes.setConstant(arguments->source_node->getIntegrationShared()->getN());
		result->sink_node->getError().chip(time_step + result->time_step, 1) += (arguments->source_node->getIntegrationErrorShared()->operator()(
			weight_tensor,
			arguments->source_node->getError().chip(time_step, 1),
			arguments->source_node->getInput().chip(time_step, 1),
			result->sink_node->getOutput().chip(time_step + result->time_step, 1),
			n_input_nodes) * result->sink_node->getDerivative().chip(time_step + result->time_step, 1));
		//result->sink_node->getIntegrationErrorShared()->operator()(
		//	weight_tensor,
		//	arguments->source_node->getError().chip(time_step, 1),
		//	arguments->source_node->getInput().chip(time_step, 1),
		//	sink_output);
		return true;
	}

	template<typename TensorT>
	bool Model<TensorT>::calculateNetNodeError_(
		OperationList<TensorT>* operations,
		const int& batch_size,
		const int& memory_size,
		const int& time_step,
		int n_threads)
	{
		std::lock_guard<std::mutex> lock(calculateNetNodeError_mutex);

		std::vector<std::future<bool>> task_results;
		int thread_cnt = 0;

		// for (const std::string& link : sink_links)
		for (int i = 0; i < operations->arguments.size(); ++i)
		{
			std::packaged_task<bool // encapsulate in a packaged_task
			(OperationResult<TensorT>*, OperationArguments<TensorT>*, int, int, int
				)> task(Model<TensorT>::calculateNodeError_);

			// launch the thread
			task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				&operations->result, &operations->arguments[i], std::ref(batch_size), std::ref(memory_size), std::ref(time_step));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || i == operations->arguments.size() - 1)
			{
				for (auto& task_result : task_results)
				{
					if (task_result.valid())
					{
						try
						{
							bool result = task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				++thread_cnt;
			}
		}
		// scale the error by the derivative and add in any residual error
		// update the node error
		operations->result.sink_node->setStatus(NodeStatus::corrected);
		return true;
	}

	template<typename TensorT>
	void Model<TensorT>::backPropogateLayerError(
		std::vector<OperationList<TensorT>>& BP_operations,
		const int& time_step, int n_threads)
	{
		// get all the information needed to construct the tensors
		std::pair<int, int> bmsizes = getBatchAndMemorySizes();
		int batch_size = bmsizes.first;
		int memory_size = bmsizes.second;

		if (time_step >= memory_size)
		{
			std::cout << "time step: " << time_step << " exceeds the memory_size!" << std::endl;
			return;
		}

		// iterate through each sink node and calculate the error
		std::vector<std::future<bool>> task_results;
		int thread_cnt = 0;
		const int threads_per_sub_process = 1; // [TODO: how to best divide up the allowable threads?]
		int operations_cnt = 0;
		for (auto& BP_operation : BP_operations)
		{
			std::packaged_task<bool // encapsulate in a packaged_task
			(OperationList<TensorT>*, int, int, int, int
				)> task(Model<TensorT>::calculateNetNodeError_);

			// launch the thread
			task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				&BP_operation, std::ref(batch_size), std::ref(memory_size), std::ref(time_step),
				std::ref(threads_per_sub_process));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || operations_cnt == BP_operations.size() - 1)
			{
				for (auto& task_result : task_results)
				{
					if (task_result.valid())
					{
						try
						{
							bool success = task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
					}
				}
				task_results.clear();
				thread_cnt = 0;
			}
			else
			{
				thread_cnt += threads_per_sub_process;
			}
			// std::cout<<"thread_count"<<thread_cnt<<std::endl;
			// std::cout<<"operations_cnt"<<operations_cnt<<std::endl;
			++operations_cnt;
		}
	}

	template<typename TensorT>
	void Model<TensorT>::backPropogate(const int& time_step, bool cache_BP_steps, bool use_cache, int n_threads)
	{
		if (use_cache)
		{
			for (auto& BP_operations : BP_operations_cache_)
				backPropogateLayerError(BP_operations, time_step, n_threads);
		}
		else
		{
			const int max_iters = 1e6;
			std::vector<std::string> sink_nodes_cycles_found;
			for (int iter = 0; iter < max_iters; ++iter)
			{
				// get the next uncorrected layer
				std::map<std::string, int> BP_operations_map;
				std::vector<OperationList<TensorT>> BP_operations_list;
				std::vector<std::string> source_nodes;
				getNextUncorrectedLayer(BP_operations_map, BP_operations_list, source_nodes);

				// get biases (not a good name...these are just sinks with other sources that have not yet been corrected)
				std::map<std::string, int> BP_operations_map_biases = BP_operations_map;
				std::vector<OperationList<TensorT>> BP_operations_list_biases = BP_operations_list;
				std::vector<std::string> sink_nodes_biases;
				getNextUncorrectedLayerBiases(BP_operations_map_biases, BP_operations_list_biases, source_nodes, sink_nodes_biases);

				// Remove all operations involving sink nodes where not all of the sources
				// have been calculated
				if (sink_nodes_biases.size() > 0)
				{
					std::vector<std::string> sink_nodes_remove;
					std::vector<OperationList<TensorT>> BP_operations_list_copy = BP_operations_list;
					for (const std::string& sink_node : sink_nodes_biases) {
						for (size_t i = BP_operations_list[BP_operations_map.at(sink_node)].arguments.size();
							i < BP_operations_list_biases[BP_operations_map_biases.at(sink_node)].arguments.size(); ++i) {
							// check if the "cyclic" argument is actually involved in a cycle
							bool isCyclicOperation = false;
							for (const auto& cyclic_pair : cyclic_pairs_) {
								if (BP_operations_list_biases[BP_operations_map_biases.at(sink_node)].arguments[i].source_node->getName() == cyclic_pair.second &&
									BP_operations_list_biases[BP_operations_map_biases.at(sink_node)].result.sink_node->getName() == cyclic_pair.first) {
									isCyclicOperation = true;
									break;
								}
							}
							// remove non cyclic sinks and ignore cyclic arguments (we will get to them after all nodes have been correct)
							if (!isCyclicOperation) {
								sink_nodes_remove.push_back(sink_node);
								break;
							}
						}
					}
					// remove all identified sink nodes
					if (sink_nodes_remove.size() > 0) {
						BP_operations_list.clear();
						for (const auto& BP_operation : BP_operations_list_copy)
							if (std::count(sink_nodes_remove.begin(), sink_nodes_remove.end(), BP_operation.result.sink_node->getName()) == 0)
								BP_operations_list.push_back(BP_operation);
					}
					else
						BP_operations_list = BP_operations_list_copy;
				}

				// check if all nodes have been corrected
				if (BP_operations_list.size() == 0)
				{
					// check for cyclic nodes
					std::vector<std::string> sink_nodes_cycles;
					getNextUncorrectedLayerCycles(BP_operations_map, BP_operations_list, source_nodes, sink_nodes_cycles);
					if (BP_operations_list.size() == 0)
						break;
					else {
						bool new_sink_node_cycle = false;
						for (const std::string& sink_node : sink_nodes_cycles) {
							if (std::count(sink_nodes_cycles_found.begin(), sink_nodes_cycles_found.end(), sink_node) == 0) {
								sink_nodes_cycles_found.push_back(sink_node);
								new_sink_node_cycle = true;
							}
						}
						if (!new_sink_node_cycle)
							break;
					}
				}

				// seperate nodes by node integration/activation

				// calculate the net input
				backPropogateLayerError(BP_operations_list, time_step, n_threads);

				if (cache_BP_steps)
					BP_operations_cache_.push_back(BP_operations_list);
			}
		}
	}

	template<typename TensorT>
	void Model<TensorT>::TBPTT(const int& time_steps, bool cache_BP_steps, bool use_cache, int n_threads)
	{
		// check time_steps vs memory_size
		int max_steps = time_steps;
		if (time_steps >= nodes_.begin()->second->getOutput().dimension(1))
		{
			std::cout << "Time_steps will be scaled back to the memory_size - 1." << std::endl;
			max_steps = nodes_.begin()->second->getOutput().dimension(1) - 1;
		}
		for (int time_step = 0; time_step < max_steps; ++time_step) {
			if (time_step > 0) {
				for (auto& node_map : nodes_) {
					if (node_map.second->getType() == NodeType::output)
						node_map.second->setStatus(NodeStatus::corrected); // reinitialize nodes
					else
						node_map.second->setStatus(NodeStatus::activated); // reinitialize nodes
				}
			}

			// calculate the error for each batch of memory
			if (cache_BP_steps && time_step == 0)
				backPropogate(time_step, true, false, n_threads);
			else if (cache_BP_steps && time_step > 0)
				backPropogate(time_step, false, true, n_threads);
			else
				backPropogate(time_step, cache_BP_steps, use_cache, n_threads);
		}
		// for (auto& node_map: nodes_)
		// {
		//   std::cout<<"Model<TensorT>::TBPTT() error: "<<node_map.second->getError()<<" for node_name: "<<node_map.first<<std::endl;
		// }
	}

	template<typename TensorT>
	void Model<TensorT>::updateWeights(const int& time_steps, std::vector<std::string> weight_names)
	{
		// check time_steps vs memory_size
		// [TODO: changed from memory_size to memory_size - 1]
		int max_steps = time_steps;
		if (time_steps >= nodes_.begin()->second->getOutput().dimension(1))
		{
			std::cout << "Time_steps will be scaled back to the memory_size - 1." << std::endl;
			max_steps = nodes_.begin()->second->getOutput().dimension(1) - 1;
		}

		std::map<std::string, TensorT> weight_derivatives;

		// calculate the average derivative for all weights
		// sum the average derivative for all time steps
		// and sum the average derivate for all time steps across shared weights
		for (const auto& link_map : links_)
		{
			// check if the weight is in the optional update list
			// [TODO: add tests]
			if (weight_names.size() != 0 &&
				std::count(weight_names.begin(), weight_names.end(), link_map.second->getWeightName()) == 0)
				continue;

			std::shared_ptr<Node<TensorT>> sink_node = nodes_.at(link_map.second->getSinkNodeName()); // which IntegrationWeightGradOp is determined by the sink node
			sink_node->getIntegrationWeightGradShared()->initNetWeightError();
			if (sink_node->getStatus() == NodeStatus::corrected) // [TODO: Skip dummy nodes?]
			{
				// Sum the error from current and previous time-steps
				// [PARALLEL: implement threads here]
				std::shared_ptr<Node<TensorT>> source_node = nodes_.at(link_map.second->getSourceNodeName());
				Eigen::Tensor<TensorT, 1> weights(source_node->getOutput().dimension(0));
				weights.setConstant(weights_.at(link_map.second->getWeightName())->getWeight());
				Eigen::Tensor<TensorT, 1> n_input_nodes(sink_node->getOutput().dimension(0));
				n_input_nodes.setConstant(sink_node->getIntegrationShared()->getN());
				for (int i = 0; i <= max_steps; ++i)
				{
					// [PARALLEL: move to threadPool/CUDA implementations]
					// [Tests: update tests accordingly]
					sink_node->getIntegrationWeightGradShared()->operator()(
						sink_node->getError().chip(i, 1),
						source_node->getOutput().chip(i, 1),
						weights,
						source_node->getInput().chip(i, 1),
						n_input_nodes);
				}
				// [PARALELL: collect threads here sum the error]
				auto found = weight_derivatives.emplace(link_map.second->getWeightName(), sink_node->getIntegrationWeightGradShared()->getNetWeightError());
				if (!found.second)
				{
					weight_derivatives.at(link_map.second->getWeightName()) += sink_node->getIntegrationWeightGradShared()->getNetWeightError();
				}
			}
		}

		// update the weights
		// [PARALLEL: implement threads here]
		for (const auto& weight_derivative : weight_derivatives)
			weights_.at(weight_derivative.first)->updateWeight(weight_derivative.second);
	}

	template<typename TensorT>
	void Model<TensorT>::reInitializeNodeStatuses()
	{
		for (auto& node_map : nodes_)
		{
			node_map.second->setStatus(NodeStatus::initialized);
		}
	}

	template<typename TensorT>
	bool Model<TensorT>::checkNodeNames(const std::vector<std::string> node_names)
	{
		bool nodes_found = true;
		for (const std::string& node_name : node_names)
		{
			if (nodes_.empty() || nodes_.count(node_name) == 0)
			{
				nodes_found = false;
				std::cout << "Node name " << node_name << " not found!" << std::endl;
			}
		}
		return nodes_found;
	}

	template<typename TensorT>
	bool Model<TensorT>::checkLinkNames(const std::vector<std::string> link_names)
	{
		bool links_found = true;
		for (const std::string& link_name : link_names)
		{
			if (links_.empty() || links_.count(link_name) == 0)
			{
				links_found = false;
				std::cout << "Link name " << link_name << " not found!" << std::endl;
			}
		}
		return links_found;
	}

	template<typename TensorT>
	bool Model<TensorT>::checkWeightNames(const std::vector<std::string> weight_names)
	{
		bool weights_found = true;
		for (const std::string& weight_name : weight_names)
		{
			if (weights_.empty() || weights_.count(weight_name) == 0)
			{
				weights_found = false;
				std::cout << "Weight name " << weight_name << " not found!" << std::endl;
			}
		}
		return weights_found;
	}

	template<typename TensorT>
	bool Model<TensorT>::checkCompleteInputToOutput(
		//const std::vector<std::string>& input_nodes, 
		//const std::vector<std::string>& output_nodes,
		int n_threads)
	{

		// [NOTE: Should not be needed now that the input/output nodes are cached upon model creation]
		//// check that all input/output nodes exist!
		//if (!checkNodeNames(input_nodes) || !checkNodeNames(output_nodes))
		//	return false;

		// infer the batch and memory size
		// [BUG: modifying the batch_size or memory_size causes a memory corruption error when
		//			 using the training the population after replicating and modifying the models
		//			 potential cause: the batch/memory sizes are not updated during training?]
		std::pair<int, int> bmsizes = getBatchAndMemorySizes();
		int batch_size_cur = bmsizes.first;
		int memory_size_cur = bmsizes.second;

		// check for uninitialized nodes
		int batch_size = 2;
		int memory_size = 2;
		if (batch_size_cur != 0)
			batch_size = batch_size_cur;
		if (memory_size_cur != 0)
			memory_size = memory_size_cur;

		// set all node outputs to zero except for the input
		// set all node derivatives to one
		// set all node errors to zero except for the output
		Eigen::Tensor<TensorT, 2> zero(batch_size, memory_size);
		zero.setConstant(0.0f);
		Eigen::Tensor<TensorT, 2> one(batch_size, memory_size);
		one.setConstant(1.0f);
		for (auto& node : input_nodes_)
		{
			node->getNodeData()->setOutput(one);
			node->getNodeData()->setInput(one);
			node->getNodeData()->setError(zero);
			node->getNodeData()->setDerivative(one);
			node->getNodeData()->setDt(one);
		}
		for (auto& node : output_nodes_)
		{
			node->getNodeData()->setOutput(zero);
			node->getNodeData()->setInput(zero);
			node->getNodeData()->setError(one);
			node->getNodeData()->setDerivative(one);
			node->getNodeData()->setDt(one);
		}
		for (auto& node_map : nodes_)
		{
			if (node_map.second->getType() != NodeType::input && node_map.second->getType() != NodeType::output)
			{
				node_map.second->getNodeData()->setOutput(zero);
				node_map.second->getNodeData()->setInput(zero);
				node_map.second->getNodeData()->setError(zero);
				node_map.second->getNodeData()->setDerivative(one);
				node_map.second->getNodeData()->setDt(one);
				node_map.second->setStatus(NodeStatus::initialized);
			}
			if (node_map.second->getType() == NodeType::input || node_map.second->getType() == NodeType::bias)
			{
				node_map.second->setStatus(NodeStatus::activated);
			}
			node_map.second->setActivation(std::shared_ptr<ActivationOp<TensorT>>(new LinearOp<TensorT>()));  // safer but requires setting																																																		
			node_map.second->setActivationGrad(std::shared_ptr<ActivationOp<TensorT>>(new LinearGradOp<TensorT>())); // the node activation back to its original value
		}

		// set all weights to 1
		for (auto& weight_map : weights_)
			weight_map.second->setWeight(1.0f);

		// Forward propogate
		try {
			forwardPropogate(0, false, false, n_threads);
		}
		catch (std::exception& e) {
			printf("Exception: %s; CheckCompleteInputToOutput failed during forward propogation.\n", e.what());
			return false;
		}

		// check that all output nodes are greater than 0
		for (auto& node : output_nodes_)
		{
			Eigen::Tensor<TensorT, 0> output = node->getOutput().sum();
			if (output(0) == 0.0f)
				return false;
		}

		// backward propagation
		for (auto& node : output_nodes_)
			node->setStatus(NodeStatus::corrected);
		try {
			backPropogate(0, false, false, n_threads);
		}
		catch (std::exception& e) {
			printf("Exception: %s; CheckCompleteInputToOutput failed during back propogation.\n", e.what());
			return false;
		}

		// check that all input nodes are greater than 0
		for (auto& node : input_nodes_)
		{
			Eigen::Tensor<TensorT, 0> error = node->getError().sum();
			if (error(0) == 0.0f)
				return false;
		}

		return true;
	}

	template<typename TensorT>
	bool Model<TensorT>::checkLinksNodeAndWeightNames(std::vector<std::string>& nodes_not_found, std::vector<std::string>& weights_not_found)
	{
		bool link_names_check = true;
		for (const auto& link_map : links_)
		{
			if (!checkNodeNames({ link_map.second->getSourceNodeName() }))
			{
				link_names_check = false;
				nodes_not_found.push_back(link_map.second->getSourceNodeName());
			}
			if (!checkNodeNames({ link_map.second->getSinkNodeName() }))
			{
				link_names_check = false;
				nodes_not_found.push_back(link_map.second->getSinkNodeName());
			}
			if (!checkWeightNames({ link_map.second->getWeightName() }))
			{
				link_names_check = false;
				weights_not_found.push_back(link_map.second->getWeightName());
			}
		}
		return link_names_check;
	}

	template<typename TensorT>
	bool Model<TensorT>::removeIsolatedNodes()
	{
		// key/value pair of node name and source/sink count pair
		std::map<std::string, std::pair<int, int>> node_counts;

		// count all sink/source connections for each node
		for (const auto& link_map : links_)
		{
			// source
			if (nodes_.at(link_map.second->getSourceNodeName())->getType() == NodeType::hidden)
			{
				auto found = node_counts.emplace(link_map.second->getSourceNodeName(), std::make_pair(1, 0));
				if (!found.second)
				{
					node_counts[link_map.second->getSourceNodeName()].first += 1;
				}
			}

			// sink
			if (nodes_.at(link_map.second->getSinkNodeName())->getType() == NodeType::hidden
				&& nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias)
			{
				auto found = node_counts.emplace(link_map.second->getSinkNodeName(), std::make_pair(0, 1));
				if (!found.second)
				{
					node_counts[link_map.second->getSinkNodeName()].second += 1;
				}
			}
		}

		bool dead_end_node_found = false;
		for (const auto& node_count : node_counts)
		{
			if (node_count.second.first == 0 || node_count.second.second == 0)
			{
				removeNodes({ node_count.first });
				dead_end_node_found = true;
			}
		}
		return dead_end_node_found;
	}

	template<typename TensorT>
	void Model<TensorT>::clearCache()
	{
		FP_operations_cache_.clear();
		BP_operations_cache_.clear();
		cyclic_pairs_.clear();
	}

	template<typename TensorT>
	std::list<int>* Model<TensorT>::convertToAdjacencyList(std::map<int, std::string>& node_id_map, int& node_cnt)
	{
		// create a map of node id to node name (excluding bias nodes)
		node_id_map.clear();
		node_cnt = 0;
		for (auto& node_map : nodes_) {
			if (node_map.second->getType() != NodeType::bias) {
				++node_cnt;
				node_map.second->setId(node_cnt);
				node_id_map.emplace(node_cnt, node_map.first);
			}
			else {
				node_map.second->setId(-1);
			}
		}

		// create the DFS trees (excluding bias nodes)
		std::list<int> *adj;
		adj = new std::list<int>[node_cnt];

		// add the actual nodes
		for (auto& link_map : links_)
			if (nodes_.at(link_map.second->getSourceNodeName())->getType() != NodeType::bias)
				adj[nodes_.at(link_map.second->getSourceNodeName())->getId() - 1].push_back(nodes_.at(link_map.second->getSinkNodeName())->getId());

		return adj;
	}

	template<typename TensorT>
	void Model<TensorT>::findCycles()
	{
		std::map<int, std::string> node_id_map;
		int node_cnt;
		std::list<int> *adj = convertToAdjacencyList(node_id_map, node_cnt);

		CircuitFinder CF(adj, node_cnt);
		CF.run();

		cyclic_pairs_.clear();
		for (const auto& source_sink : CF.getCycles()) {
			if (nodes_.at(node_id_map.at(source_sink.second))->getType() == NodeType::recursive) // enforce order of recursive nodes
				cyclic_pairs_.push_back(std::make_pair(node_id_map.at(source_sink.second), node_id_map.at(source_sink.first)));
			else
				cyclic_pairs_.push_back(std::make_pair(node_id_map.at(source_sink.first), node_id_map.at(source_sink.second)));
		}
	}

	template<typename TensorT>
	std::vector<std::pair<std::string, std::string>> Model<TensorT>::getCyclicPairs()
	{
		return cyclic_pairs_;
	}
}

#endif //SMARTPEAK_MODEL_H