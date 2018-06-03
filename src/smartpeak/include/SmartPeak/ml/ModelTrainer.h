/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINER_H
#define SMARTPEAK_MODELTRAINER_H

#include <SmartPeak/ml/Model.h>

#include <vector>
#include <string>

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

    TODO: rename to ModelTrainer
  */
  class ModelTrainer
  {
public:
    ModelTrainer(); ///< Default constructor
    ~ModelTrainer(); ///< Default destructor

    void setBatchSize(const int& batch_size); ///< batch_size setter
    void setMemorySize(const int& memory_size); ///< memory_size setter
    void setNEpochs(const int& n_epochs); ///< n_epochs setter

    int getBatchSize() const; ///< batch_size setter
    int getMemorySize() const; ///< memory_size setter
    int getNEpochs() const; ///< n_epochs setter
 
    /**
      @brief Load model from file

      @param filename_nodes The name of the nodes file
      @param filename_links The name of the links file
      @param filename_weights The name of the weights file
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool loadModel(const std::string& filename_nodes,
      const std::string& filename_links,
      const std::string& filename_weights,
      Model& model);
 
    /**
      @brief Load the model node output, derivative, and error data.

      @param filename The name of the node data
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool loadNodeStates(const std::string& filename, Model& model);
 
    /**
      @brief Load the model weights.

      @param filename The name of the node data
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool loadWeights(const std::string& filename, Model& model);
 
    /**
      @brief Store the model to file

      @param filename_nodes The name of the nodes file
      @param filename_links The name of the links file
      @param filename_weights The name of the weights file
      @param model The model to store data for

      @returns Status True on success, False if not
    */ 
    bool storeModel(const std::string& filename_nodes,
      const std::string& filename_links,
      const std::string& filename_weights,
      const Model& model);
 
    /**
      @brief Store the model node output, derivative, and error data.

      @param filename The name of the node data
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool storeNodeStates(const std::string& filename, const Model& model);
 
    /**
      @brief Store the model weights.

      @param filename The name of the node data
      @param model The model to load data into

      @returns Status True on success, False if not
    */ 
    bool storeWeights(const std::string& filename, const Model& model);
 
    /**
      @brief Load input data from file

      @param filename The name of the model file
      @param

      @returns Status True on success, False if not
    */ 
    bool loadInputData(const std::string& filename, Eigen::Tensor<float, 4>& input);
 
    /**
      @brief Load output data from file

      @param filename The name of the model file
      @param

      @returns Status True on success, False if not
    */ 
    bool loadOutputData(const std::string& filename, Eigen::Tensor<float, 3>& output);
 
    /**
      @brief Check input dimensions.

      @param [TODO: add params docstrings]

      @returns True on success, False if not
    */ 
    bool checkInputData(const int& n_epochs,
      const Eigen::Tensor<float, 4>& input,
      const int& batch_size,
      const int& memory_size,
      const std::vector<std::string>& input_nodes);
 
    /**
      @brief Check output dimensions.

      @param [TODO: add params docstrings]

      @returns True on success, False if not
    */ 
    bool checkOutputData(const int& n_epochs,
      const Eigen::Tensor<float, 3>& output,
      const int& batch_size,
      const std::vector<std::string>& output_nodes);
 
    /**
      @brief [TODO: add method] Check time step dimensions required for FPTT.

      @param [TODO: add params docstrings]

      @returns True on success, False if not
    */ 
    bool checkTimeSteps(const int& n_epochs,
      const Eigen::Tensor<float, 3>& time_steps,
      const int& batch_size,
      const int& memory_size);
 
    /**
      @brief Entry point for users to code their script
        for model training

      @param[in, out] model The model to train
      @param[in] n_epochs The number of epochs to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names
      @param[in] output_nodes Output node names
    */ 
    virtual void trainModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes) = 0;
 
    /**
      @brief Entry point for users to code their script
        for model validation

      @param[in, out] model The model to train
      @param[in] n_epochs The number of epochs to train
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names
      @param[in] output_nodes Output node names

      @returns vector of average model error scores
    */ 
    virtual std::vector<float> validateModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes) = 0;
 
    /**
      @brief Entry point for users to code their script
        to build the model

      @returns The constructed model
    */ 
    virtual Model makeModel() = 0;

private:
    int batch_size_;
    int memory_size_;
    int n_epochs_;
    bool is_trained_ = false;

  };
}

#endif //SMARTPEAK_MODELTRAINER_H