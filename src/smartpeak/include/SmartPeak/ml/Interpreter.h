/**TODO:  Add copyright*/

#ifndef SMARTPEAK_INTERPRETER_H
#define SMARTPEAK_INTERPRETER_H

#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Link.h>

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

    TODO: rename to Trainer
  */
  class Interpreter
  {
public:
    Interpreter(); ///< Default constructor
    ~Interpreter(); ///< Default destructor
 
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