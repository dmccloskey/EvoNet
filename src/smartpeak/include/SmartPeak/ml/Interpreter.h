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

    The execution graph is modeled as a DAG of tensor layers
      (as opposed to scalar nodes) with input tensor layers,
      output tensor layers, and intemediate tensor layers.
      The layers are defined based on the network model structure
      and node types of the model.
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

      @param[in] input_layers Tensor input indexes

      @returns Status True on success, False if not
    */ 
    bool setInputLayers(std::vector<int> input_layers);
 
    /**
      @brief Provide a list of tensor indexes that are outputs to the model
        Each index is bound check and this modifies the consistent_ flag of the
        interpreter.

      @param[in] input_layers Tensor output indexes

      @returns Status True on success, False if not
    */ 
    bool setOutputLayers(std::vector<int> output_layers);
 
    /**
      @brief Adds intermediate layers.

      @param[in] source_layers Tensor output indexes
      @param[in] sink_layers Tensor output indexes

      @returns Status True on success, False if not
    */ 
    bool addIntermediateLayer(const std::vector<int>& source_layers,
      const std::vector<int>& sink_layers);

    // Adds `tensors_to_add` tensors, preserving pre-existing Tensor entries.
    // The value pointed to by `first_new_tensor_index` will be set to the
    // index of the first new tensor if `first_new_tensor_index` is non-null.
 
    /**
      @brief Adds `tensors_to_add` tensors, preserving pre-existing Tensor entries.
        The value pointed to by `first_new_tensor_index` will be set to the
        index of the first new tensor if `first_new_tensor_index` is non-null.

      @param[in] tensors_to_add New tensors to add
      @param[in] first_new_tensor_index Index to insert new tensors

      @returns Status True on success, False if not
    */ 
    bool addTensors(int& tensors_to_add,
      int& first_new_tensor_index);

    // // Prepare the given 'node' for execution.
    // TfLiteStatus OpPrepare(const TfLiteRegistration& op_reg, TfLiteNode* node) {
    //   if (op_reg.prepare == nullptr) return kTfLiteOk;
    //   return op_reg.prepare(&context_, node);
    // }

    // // Invoke the operator represented by 'node'.
    // TfLiteStatus OpInvoke(const TfLiteRegistration& op_reg, TfLiteNode* node) {
    //   if (op_reg.invoke == nullptr) return kTfLiteError;
    //   return op_reg.invoke(&context_, node);
    // }

    // Call OpPrepare() for as many ops as possible, allocating memory for their
    // tensors. If an op containing dynamic tensors is found, preparation will be
    // postponed until this function is called again. This allows the interpreter
    // to wait until Invoke() to resolve the sizes of dynamic tensors.
 
    /**
      @brief Allocate tensor dimensions.

      @returns Status True on success, False if not
    */ 
    bool allocateTensorMemory();
 
    /**
      @brief Check if tensor dimensions are consistent with
        respect to input and out tensor layers.

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
      @brief Execute the execution graph layer by layer from
        input layer to output layer

      @returns Status True on success, False if not
    */ 
    bool execute();

private:

    // Array of indices representing the tensors that are inputs to the
    // interpreter.
    std::vector<int> inputs_layers_;

    // Array of indices representing the tensors that are outputs to the
    // interpreter.
    std::vector<int> output_layers_;

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