/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/LossFunction.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <tuple>
#include <list>

namespace SmartPeak
{

  struct OperationResult
  {
    std::shared_ptr<Node> sink_node;
    int time_step = 0;
  };

  struct OperationArguments
  {
    std::shared_ptr<Node> source_node;
    std::shared_ptr<Weight> weight;
		std::string link_name;
    int time_step = 0;
  };

  struct OperationList
  {
    OperationResult result;
    std::vector<OperationArguments> arguments;
  };

  /**
    @brief Directed Network Model

    Assumptions about the model structure:
    1. Inputs can only be sources
    2. Outputs can only be sinks (will break back propogation algorithm)
  */
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
      const Eigen::Tensor<float, 3>& values,
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
      const Eigen::Tensor<float, 2>& values,
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
      const Eigen::Tensor<float, 1>& values,
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
      const Eigen::Tensor<float, 1>& values,
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
      std::vector<OperationList>& FP_operations);
 
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
      std::vector<OperationList>& FP_operations,
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
      std::vector<OperationList>& FP_operations,
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
      std::vector<OperationList>& FP_operations,
      const int& time_step, int n_threads = 1);

    static bool calculateNodeInput_(
			OperationResult* result,
      OperationArguments* arguments, 
      const int& batch_size,
      const int& memory_size,
      const int& time_step
    );
    static bool calculateNetNodeInput_(
      OperationList* operations, 
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
      const Eigen::Tensor<float, 3>& values,
      const std::vector<std::string> node_names,
      const Eigen::Tensor<float, 2>& dt, 
      bool cache_FP_steps = false, 
      bool use_cache = false,
      int n_threads = 1);
 
    /**
    @brief Calculates the error of the model with respect to
      the expected values

    @param[in] values Expected node output values
    @param[in] node_names Output nodes
    */ 
    void calculateError(const Eigen::Tensor<float, 2>& values, const std::vector<std::string>& node_names,
			const int& time_step, int n_threads = 1);

		/**
		@brief Calculates the error of the model for a given node
		*/
		static Eigen::Tensor<float, 1> calculateModelError_(
			Node* output_node,
			const Eigen::Tensor<float, 1>& expected,
			LossFunctionOp<float>* loss_function,
			const int& batch_size,
			const int& time_step
			);

		/**
		@brief Calculates the error of the output node
		*/
		static bool calculateOutputNodeError_(
			Node* output_node,
			const Eigen::Tensor<float, 1>& expected,
			LossFunctionGradOp<float>* loss_function_grad,
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
    void CETT(const Eigen::Tensor<float, 3>& values, const std::vector<std::string>& node_names, const int& time_steps, int n_threads = 1);
 
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
      std::vector<OperationList>& BP_operations,
      std::vector<std::string>& source_nodes);

		void getNextUncorrectedLayerBiases(
			std::map<std::string, int>& BP_operations_map, 
			std::vector<OperationList>& BP_operations, 
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
      std::vector<OperationList>& BP_operations,
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
      std::vector<OperationList>& BP_operations,
      const int& time_step, int n_threads = 1);

    static bool calculateNodeError_(
			OperationResult* operations,
      OperationArguments* arguments, 
      const int& batch_size,
      const int& memory_size,
      const int& time_step
    );
    static bool calculateNetNodeError_(
      OperationList* operations, 
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

    void setError(const Eigen::Tensor<float, 2>& error); ///< error setter
    Eigen::Tensor<float, 2> getError() const; ///< error getter

    void setLossFunction(const std::shared_ptr<LossFunctionOp<float>>& loss_function); ///< loss_function setter
    LossFunctionOp<float>* getLossFunction() const; ///< loss_function getter

		void setLossFunctionGrad(const std::shared_ptr<LossFunctionGradOp<float>>& loss_function); ///< loss_function grad setter
		LossFunctionGradOp<float>* getLossFunctionGrad() const; ///< loss_function grad getter

		std::vector<std::shared_ptr<Node>> getInputNodes(); ///< input_node getter
		std::vector<std::shared_ptr<Node>> getOutputNodes(); ///< output_node getter
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
    void addNodes(const std::vector<Node>& nodes);
    Node getNode(const std::string& node_name) const; ///< node getter
    std::vector<Node> getNodes() const; ///< nodes getter
		std::map<std::string, std::shared_ptr<Node>> getNodesMap();  ///< return a modifiable version of weights
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
    void addWeights(const std::vector<Weight>& weights);
    Weight getWeight(const std::string& weight_name) const; ///< weight getter
    std::vector<Weight> getWeights() const;  ///< weights getter
		std::map<std::string, std::shared_ptr<Weight>> getWeightsMap();  ///< return a modifiable version of weights_		
 
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
    std::map<std::string, std::shared_ptr<Node>> nodes_; ///< Model nodes
    std::map<std::string, std::shared_ptr<Weight>> weights_; ///< Model nodes
    Eigen::Tensor<float, 2> error_; ///< Model error
    std::shared_ptr<LossFunctionOp<float>> loss_function_; ///< Model loss function
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad_; ///< Model loss function
		std::vector<std::pair<std::string, std::string>> cyclic_pairs_;
		std::vector<std::shared_ptr<Node>> input_nodes_;
		std::vector<std::shared_ptr<Node>> output_nodes_;

    // Internal structures to allow for efficient multi-threading
    // and off-loading of computation from host to devices
    std::vector<std::vector<OperationList>> FP_operations_cache_;
    std::vector<std::vector<OperationList>> BP_operations_cache_;
  };
}

#endif //SMARTPEAK_MODEL_H