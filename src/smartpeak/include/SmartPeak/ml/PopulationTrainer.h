/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINER_H
#define SMARTPEAK_POPULATIONTRAINER_H

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelTrainer.h>

#include <vector>
#include <string>

namespace SmartPeak
{

  /**
    @brief Replicates a model with or without modification (i.e., mutation)
  */
  class PopulationTrainer
  {
public:
    PopulationTrainer(); ///< Default constructor
    ~PopulationTrainer(); ///< Default destructor
 
    /**
      @brief Select the top N models with the least error
      [TODO: add method and tests]

      Use cases with different parameters:
      - Top N selection: set n_top ? 0, set n_random == 0
      - Top N random selection: set n_top > 0, set n_random > 0 && n_random <= n_top
      - Random selection: set n_top == 0, set n_random > 0
      - Binary selection: given models.size() == 2, set n_top == 1, set n_random == 0

      @param[in] n_top The number models to select
      @param[in] n_random The number of random models to select from the pool of top models
      @param[in, out] models The vector (i.e., population) of models to select from

      @returns A vector of models (i.e., subset of the population)
    */ 
    std::vector<Model> selectModels(
      const int& n_top,
      const int& n_random,
      std::vector<Model>& models,
      ModelTrainer& model_trainer,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes);
 
    /**
      @brief Replicates the models in the population.  Replicates
        are modified while the original models are persisted.

      Example use case:
        - 2 selected models are replicated 4 times with modifications
          resulting in a population of 10 models (2 original, and 8 
          modified)

      [TODO: add method and tests]

      @param[in, out] models The vector (i.e., population) of models to modify
      @param[in] model_replicator The replicator to use
      @param[in] n_replicates_per_model The number of replications per model
    */ 
    void replicateModels(
      std::vector<Model>& models,
      ModelReplicator& model_replicator,
      const int& n_replicates_per_model);
 
    /**
      @brief Trains each of the models in the population
        using the same test data set

      [TODO: add method and tests]

      @param[in, out] models The vector of models to copy
      @param[in] model_trainer The trainer to use
    */ 
    void trainModels(
      std::vector<Model>& models,
      ModelTrainer& model_trainer,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes);
  };
}

#endif //SMARTPEAK_POPULATIONTRAINER_H