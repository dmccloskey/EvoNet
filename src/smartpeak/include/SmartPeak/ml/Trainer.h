/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TRAINER_H
#define SMARTPEAK_TRAINER_H

#include <SmartPeak/ml/Model.h>

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
  class Trainer
  {
public:
    Trainer(); ///< Default constructor
    ~Trainer(); ///< Default destructor
 
    /**
      @brief Load model from file

      @param filename The name of the model file
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool loadModel(const std::string& filename, Model& model);
 
    /**
      @brief Load the model node output, derivative, and error data.

      @param filename The name of the node data
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool loadNodeStates(const std::string& filename, Model& model);
 
    /**
      @brief Store the model to file

      @param filename The name of the model file
      @param model The model to store data for

      @returns Status True on success, False if not
    */ 
    bool storeModel(const std::string& filename, Model& model);
 
    /**
      @brief Store the model node output, derivative, and error data.

      @param filename The name of the node data
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool storeNodeStates(const std::string& filename, Model& model);

private:
    int batch_size_;
    int memory_size_;
    int n_epochs_;
    float dt_;

  };
}

#endif //SMARTPEAK_TRAINER_H