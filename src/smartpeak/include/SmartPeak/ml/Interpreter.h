/**TODO:  Add copyright*/

#ifndef SMARTPEAK_INTERPRETER_H
#define SMARTPEAK_INTERPRETER_H

#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Interpreter.h>

#include <vector>

namespace SmartPeak
{

  /**
    @brief Execution graph interpreter for a network model.

    The execution graph is modeled as a DAG of tensors
      (composed of multiple scalar nodes and scalar weights) with input tensors,
      output tensors, and intemediate tensors.
      The tensors are defined based on the network model structure
      and node types of the model.

    Intended sequence of events:
      Construct execution graph from the network model
      For n epochs:
        Set the input data
        Set the expected data (if training/validating)
        Foward propogation:
          1. f(source * weights) = sinks
          2. calculate the derivatives for back propogation
        Back propogation (if training):
          1. sinks * weights . derivatives = sources
          2. adjust the weights
      Update the network model from the execution graph tensors (if training)
  */
  class Interpreter
  {
public:
    Interpreter(); ///< Default constructor
    ~Interpreter(); ///< Default destructor
 
    /**
      @brief Provide a list of tensor indexes that are inputs to the model.
        Each index is bound check and this modifies the consistent_ flag of the
        interpreter.

      @param[in] input_tensors Tensor input indexes

      @returns Status True on success, False if not
    */ 
    bool setInputTensors(std::vector<int> input_tensors);
 
    /**
      @brief Provide a list of tensor indexes that are outputs to the model
        Each index is bound check and this modifies the consistent_ flag of the
        interpreter.

      @param[in] input_tensors Tensor output indexes

      @returns Status True on success, False if not
    */ 
    bool setOutputTensors(std::vector<int> output_tensors);
 
    /**
      @brief Adds intermediate layers.

      @param[in] source_tensors Tensor output indexes
      @param[in] sink_tensors Tensor output indexes

      @returns Status True on success, False if not
    */ 
    bool addIntermediateTensor(const std::vector<int>& source_tensors,
      const std::vector<int>& sink_tensors);
 
    /**
      @brief Adds `tensors_to_add` tensors, preserving pre-existing Tensor entries.
        The value pointed to by `first_new_tensor_index` will be set to the
        index of the first new tensor if `first_new_tensor_index` is non-null.

      @param[in] tensors_to_add New tensors to add
      @param[in] first_new_tensor_index Index to insert new tensors

      @returns Status True on success, False if not
    */ 
    bool addTensors(const std::vector<int>& tensors_to_add,
      int& first_new_tensor_index);
 
    /**
      @brief Removes `tensors_to_remove` tensors.

      @param[in] tensors_to_remove Existing tensors to remove

      @returns Status True on success, False if not
    */ 
    bool removeTensors(const std::vector<int>& tensors_to_remove);
 
    /**
      @brief Allocate tensor dimensions.

      @returns Status True on success, False if not
    */ 
    bool allocateTensorMemory();
 
    /**
      @brief Check if tensor dimensions are consistent with
        respect to input and out tensors.

      this changes consistent_ to be false if dimensions do not match.

      @returns Status True on success, False if not
    */ 
    bool checkTensorDimensions();
 
    /**
      @brief Check if the tensor indices for each layer are constent.

      this changes consistent_ to be false if indices are out of bounds.

      @returns Status True on success, False if not
    */ 
    bool checkTensorIndices();
 
    /**
      @brief Add an instruction involving two tensors to the execution graph.

      @param[in] instruction Instruction to add

      @returns Status True on success, False if not
    */ 
    bool addInstructions();
 
    /**
      @brief Remove an instruction involving two tensors in the execution graph.

      @param[in] instruction Instruction to remove

      @returns Status True on success, False if not
    */ 
    bool removeInstructions();
 
    /**
      @brief Map a network model to the execution graph.

      @param[in] model The model to construct the execution graph from

      @returns Status True on success, False if not
    */ 
    bool mapModelToTensors();
 
    /**
      @brief Update a network model parameters from the tensors of the
        execution graph.

      @param[in, out] model The model update

      @returns Status True on success, False if not
    */ 
    bool updateModelFromTensors();
 
    /**
      @brief Forward propogation

      @param[in] Input data

      @returns Status True on success, False if not
    */ 
    bool forwardPropogate();
 
    /**
      @brief Error calculation

      @param[in] Expected values

      @returns Status True on success, False if not
    */ 
    bool errorCalculation();
 
    /**
      @brief Backward propogation

      @param[in] Error

      @returns Status True on success, False if not
    */ 
    bool backwardPropogate();
 
    /**
      @brief Update the weights

      @returns Status True on success, False if not
    */ 
    bool weightUpdate();

private:

    // Array of indices representing the tensors that are inputs to the
    // interpreter.
    std::vector<int> inputs_tensors_;

    // Array of indices representing the tensors that are outputs to the
    // interpreter.
    std::vector<int> output_tensors_;

    // Array of indices representing the order of tensors
    // in the execution graph.
    std::vector<int> execution_graph_;

    // Whether the tensors of the execution graph are consistent
    // with respect to dimensions and indices.
    bool consistent_ = true;

    // Whether the model is safe to invoke (if any errors occurred this
    // will be false).
    bool invokable_ = false;

  };
}

#endif //SMARTPEAK_INTERPRETER_H