/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINER_H
#define SMARTPEAK_POPULATIONTRAINER_H

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/ModelReplicator.h>

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

      Modes:
      - Top N selection: set n_top ? 0, set n_random == 0
      - Top N random selection: set n_top > 0, set n_random > 0 && n_random <= n_top
      - Random selection: set n_top == 0, set n_random > 0
      - Binary selection: given models.size() == 2, set n_top == 1, set n_random == 0

      @param n_top The number models to select
      @param n_random The number of random models to select from the pool of top models
      @param models The vector (i.e., population) of models to select from

      @returns A vector of models (i.e., subset of the population)
    */ 
    std::vector<Model> selectModels(
      const int& n_top,
      const int& n_random,
      const std::vector<Model>& models);
 
    /**
      @brief Copies the models in the population

      @param models The vector of models to copy

      @returns A vector of replicated models
    */ 
    std::vector<Model> copyModels(const std::vector<Model>& models);
 
    /**
      @brief Replicates the models in the population

      @param models The vector (i.e., population) of models to modify
      @param model_replicator The replicator to use
    */ 
    void modifyModels(
      std::vector<Model>& models,
      const ModelReplicator& model_replicator);
  };
}

#endif //SMARTPEAK_POPULATIONTRAINER_H