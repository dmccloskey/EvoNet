/**TODO:  Add copyright*/

#pragma once

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h> 
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>

#include <random>
#include <fstream>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/*
  @brief implementation of the add problem that
    has been used to test sequence prediction in 
    RNNS

  References:
    [TODO]

  @input[in] sequence_length
  @input[in, out] random_sequence
  @input[in, out] mask_sequence

  @returns the result of the two random numbers in the sequence
**/
static float AddProb(
  const int& sequence_length,
  std::vector<float>& random_sequence,
  std::vector<float>& mask_sequence)
{
  float result = 0.0;
  
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> zero_to_one(0, 1);
    std::uniform_int_distribution<> zero_to_length(0, sequence_length);

  // generate 2 random and unique indexes between 
  // [0, sequence_length) for the mask
  int mask_index_1 = zero_to_length(gen);
  int mask_index_2 = 0;
  do {
    mask_index_2 = zero_to_length(gen);
  } while (mask_index_1 == mask_index_2);

  // generate the random sequence
  // and the mask sequence
  random_sequence.clear();
  random_sequence.reserve(sequence_length);
  mask_sequence.clear();
  mask_sequence.reserve(sequence_length);
  for (int i=0; i<sequence_length; ++i)
  {
    // the random sequence
    random_sequence[i] = zero_to_one(gen);
    // the mask
    if (i == mask_index_1 || i == mask_index_2)
      mask_sequence[i] = 1.0;
    else
      mask_sequence[i] = 0.0;

    // result update
    result += mask_sequence[i] * random_sequence[i];
  }

  return result;
};

// ModelTrainer used for all tests
class ModelTrainerTest: public ModelTrainer
{
public:
  Model makeModel()
  {
    Model model;
    return model;
  };
  void trainModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // printf("Training the model\n");

    // Check input and output data
    if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
    {
      return;
    }
    if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
    {
      return;
    }
    if (!model.checkNodeNames(input_nodes))
    {
      return;
    }
    if (!model.checkNodeNames(output_nodes))
    {
      return;
    }
    // printf("Data checks passed\n");
    
    // Initialize the model
    model.clearCache();
    model.initNodes(getBatchSize(), getMemorySize());
    // printf("Initialized the model\n");

    for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
    {
      // printf("Training epoch: %d\t", iter);

      // forward propogate
      if (iter == 0)
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true); 
      else      
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true); 

      // calculate the model error and node output error
      model.calculateError(output.chip(iter, 2), output_nodes);
      // std::cout<<"Model error: "<<model.getError().sum()<<std::endl;

      // back propogate
      if (iter == 0)
        model.TBPTT(getMemorySize()-1, true, true);
      else
        model.TBPTT(getMemorySize()-1, false, true);

      // update the weights
      model.updateWeights(getMemorySize());   

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
    }
  }
  std::vector<float> validateModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // printf("Validating model %s\n", model.getName().data());

    std::vector<float> model_error;

    // Check input and output data
    if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
    {
      return model_error;
    }
    if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
    {
      return model_error;
    }
    if (!model.checkNodeNames(input_nodes))
    {
      return model_error;
    }
    if (!model.checkNodeNames(output_nodes))
    {
      return model_error;
    }
    // printf("Data checks passed\n");
    
    // Initialize the model
    model.clearCache();
    model.initNodes(getBatchSize(), getMemorySize());
    // printf("Initialized the model\n");

    for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
    {
      // printf("validation epoch: %d\t", iter);

      // forward propogate
      model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2)); 

      // calculate the model error and node output error
      model.calculateError(output.chip(iter, 2), output_nodes); 
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      model_error.push_back(total_error(0));  
      // std::cout<<"Model error: "<<total_error(0)<<std::endl;

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
    }
    return model_error;
  }
};