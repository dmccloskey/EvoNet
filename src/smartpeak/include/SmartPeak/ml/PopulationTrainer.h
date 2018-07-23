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
      @brief Remove models with non-unique names from the population of models

      @param[in, out] models The vector (i.e., population) of models to select from
    */ 
    void removeDuplicateModels(std::vector<Model>& models);
 
    /**
      @brief Select the top N models with the least error

      Use cases with different parameters:
      - Top N selection: set n_top ? 0, set n_random == 0
      - Top N random selection: set n_top > 0, set n_random > 0 && n_random <= n_top
      - Random selection: set n_top == 0, set n_random > 0
      - Binary selection: given models.size() == 2, set n_top == 1, set n_random == 0

      [TESTS: add thread tests]

      @param[in] n_top The number models to select
      @param[in] n_random The number of random models to select from the pool of top models
      @param[in, out] models The vector (i.e., population) of models to select from

			@returns a list of pairs of model_name to average validation error
    */ 
		std::vector<std::pair<std::string, float>> selectModels(
      const int& n_top,
      const int& n_random,
      std::vector<Model>& models,
      ModelTrainer& model_trainer,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 4>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes,
      int n_threads = 1);
 
    /**
      @brief validate all of the models

      @returns key value pair of model_name and model_error
    */ 
    // [DEPRECATED]
    std::vector<std::pair<std::string, float>> validateModels_(
      std::vector<Model>& models,
      ModelTrainer& model_trainer,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 4>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes);
    static std::pair<std::string, float> validateModel_(
      Model* model,
      ModelTrainer* model_trainer,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 4>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes);
 
    /**
      @brief return the top N models with the lowest error.

      @returns key value pair of model_name and model_error
    */ 
    static std::vector<std::pair<std::string, float>> getTopNModels_(
      std::vector<std::pair<std::string, float>> model_validation_scores,
      const int& n_top);
 
    /**
      @brief return a random list of model names.

      @returns key value pair of model_name and model_error
    */ 
    static std::vector<std::pair<std::string, float>> getRandomNModels_(
      std::vector<std::pair<std::string, float>> model_validation_scores,
      const int& n_random);
 
    /**
      @brief Replicates the models in the population.  Replicates
        are modified while the original models are persisted.

      Example use case:
        - 2 selected models are replicated 4 times with modifications
          resulting in a population of 10 models (2 original, and 8 
          modified)

      [TESTS: add thread tests]

      @param[in, out] models The vector (i.e., population) of models to modify
      @param[in] model_replicator The replicator to use
      @param[in] n_replicates_per_model The number of replications per model

      @returns A vector of models
    */ 
    void replicateModels(
      std::vector<Model>& models,
      ModelReplicator& model_replicator,
      const int& n_replicates_per_model,
      std::string unique_str = "",
      int n_threads = 1);

    static Model replicateModel_(
      Model* model,
      ModelReplicator* model_replicator,
      std::string unique_str, int cnt, int i);
 
    /**
      @brief Trains each of the models in the population
        using the same test data set

      [TESTS: add thread tests]

      @param[in, out] models The vector of models to copy
      @param[in] model_trainer The trainer to use
    */ 
    void trainModels(
      std::vector<Model>& models,
      ModelTrainer& model_trainer,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 4>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes,
      int n_threads = 1);

    static std::pair<bool, Model> trainModel_(
      Model* model,
      ModelTrainer* model_trainer,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 4>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes);
  };
}

#endif //SMARTPEAK_POPULATIONTRAINER_H