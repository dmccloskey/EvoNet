/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINER_H
#define SMARTPEAK_POPULATIONTRAINER_H

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/simulator/DataSimulator.h>

#include <vector>
#include <string>

namespace SmartPeak
{

  /**
    @brief Trains a vector of models
  */
  class PopulationTrainer
  {
public:
    PopulationTrainer() = default; ///< Default constructor
    ~PopulationTrainer() = default; ///< Default destructor 

		void setNTop(const int& n_top); ///< n_top setter
		void setNRandom(const int& n_random); ///< n_random setter
		void setNReplicatesPerModel(const int& n_replicates_per_model); ///< n_replicates_per_model setter
		void setNGenerations(const int& n_generations); ///< n_generations setter

		int getNTop() const; ///< batch_size setter
		int getNRandom() const; ///< memory_size setter
		int getNReplicatesPerModel() const; ///< n_epochs setter
		int getNGenerations() const; ///< n_epochs setter

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

      @param[in, out] models The vector (i.e., population) of models to select from

			@returns a list of pairs of model_name to average validation error
    */ 
		std::vector<std::pair<int, float>> selectModels(
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
    static std::pair<int, float> validateModel_(
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
    static std::vector<std::pair<int, float>> getTopNModels_(
      std::vector<std::pair<int, float>> model_validation_scores,
      const int& n_top);
 
    /**
      @brief return a random list of model names.

      @returns key value pair of model_name and model_error
    */ 
    static std::vector<std::pair<int, float>> getRandomNModels_(
      std::vector<std::pair<int, float>> model_validation_scores,
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

      @returns A vector of models
    */ 
    void replicateModels(
      std::vector<Model>& models,
      ModelReplicator& model_replicator,
			const std::vector<std::string>& input_nodes,
			const std::vector<std::string>& output_nodes,
      std::string unique_str = "",
      int n_threads = 1);

    static Model replicateModel_(
      Model* model,
      ModelReplicator* model_replicator,
			const std::vector<std::string>& input_nodes,
			const std::vector<std::string>& output_nodes,
      std::string unique_str, int cnt);
 
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

		int getNextID(); ///< iterate and return the next id in the sequence
		void setID(const int& id);  ///< unique_id setter
 
		/**
		@brief Train the population

		@param[in, out] models The vector of models to copy
		@param[in] model_trainer The trainer to use
		@param[in] model_replicator The replicator to use
		@param[in] data_simulator The data simulate/generator to use
		*/
		std::vector<std::vector<std::pair<int, float>>> evolveModels(
			std::vector<Model>& models,
			ModelTrainer& model_trainer,
			ModelReplicator& model_replicator,
			DataSimulator& data_simulator,
			const std::vector<std::string>& input_nodes,
			const std::vector<std::string>& output_nodes,
			int n_threads = 1);

		/**
		@brief Entry point for users to code their adaptive scheduler
		to modify models population dynamic parameters based on a given trigger

		@param[in] n_generations The number of evolution generations
		@param[in] models The models in the population
		@param[in] model_errors The trace of models errors from validation at each generation
		*/
		void adaptivePopulationScheduler(
			const int& n_generations,
			std::vector<Model>& models,
			std::vector<std::vector<std::pair<int, float>>>& models_errors_per_generations);

private:
		int unique_id_ = 0;

		// population dynamics
		int n_top_ = 0; ///< The number models to select
		int n_random_ = 0; ///< The number of random models to select from the pool of top models
		int n_replicates_per_model_ = 0; ///< The number of replications per model
		int n_generations_ = 0; ///< The number of generations to evolve the models
  };
}

#endif //SMARTPEAK_POPULATIONTRAINER_H