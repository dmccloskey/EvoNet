/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINEREXPERIMENTAL_H
#define SMARTPEAK_POPULATIONTRAINEREXPERIMENTAL_H

// .h
#include <SmartPeak/ml/PopulationTrainer.h>

namespace SmartPeak
{
  /**
    @brief Experimental methods for PopulationTrainer
  */
	template<typename TensorT, typename InterpreterT>
  class PopulationTrainerExperimental: public PopulationTrainer<TensorT, InterpreterT>
  {
public:
    PopulationTrainerExperimental() = default; ///< Default constructor
    ~PopulationTrainerExperimental() = default; ///< Default destructor 

    /*
    @brief Adjust the population size based on the number of generations
      error rates of training

    @param[in] n_generations The number of generations
    @param[in] models A vector of models representing the population
    @param[in] models_errors_per_generations A record of model errors per generation
    */
    void setPopulationSizeFixed(
      const int& n_generations,
      std::vector<Model<TensorT>>& models,
      std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations);

    /*
    @brief Adjust the population size for growth and selection modes
    1. growth phase: each model doubles for a period of time (e.g., 1, 2, 4, 8, 16, 32, 64, 128, ...)
    2. selection phase: best models are selected (e.g., from 64 to 8)

    @param[in] n_generations The number of generations
    @param[in] models A vector of models representing the population
    @param[in] models_errors_per_generations A record of model errors per generation
    */
    void setPopulationSizeDoubling(
      const int& n_generations,
      std::vector<Model<TensorT>>& models,
      std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations);

    /*
    @brief Adjust the number of training steps based on the average model size

    @param[in] models A vector of models representing the population
    */
    void setTrainingStepsByModelSize(std::vector<Model<TensorT>>& models);
  };
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainerExperimental<TensorT, InterpreterT>::setPopulationSizeFixed(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) {
    // Adjust the population sizes
    const size_t population_size = 32;
    const size_t selection_ratio = 4; ///< options include 2, 4, 8
    const size_t selection_size = population_size / selection_ratio;
    if (n_generations == 0) {
      this->setNTop(selection_size);
      this->setNRandom(selection_size);
      this->setNReplicatesPerModel(population_size - 1);
    }
    else {
      this->setNTop(selection_size);
      this->setNRandom(selection_size);
      this->setNReplicatesPerModel(selection_ratio - 1);
    }

    // Set additional model replicator settings
    this->setRemoveIsolatedNodes(true);
    this->setPruneModelNum(10);
    this->setCheckCompleteModelInputToOutput(true);

    // Adjust the training steps
    this->setTrainingStepsByModelSize(models);
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainerExperimental<TensorT, InterpreterT>::setPopulationSizeDoubling(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) {
    // Adjust the population sizes
    const size_t max_population_size = 128;
    //const size_t selection_ratio = 16; ///< options include 2, 4, 8, 16, 32, etc.
    //const size_t selection_size = models.size() / selection_ratio;
    const size_t selection_size = 8;
    if (models.size() >= max_population_size) {
      this->setNTop(selection_size);
      this->setNRandom(selection_size);
      this->setNReplicatesPerModel(1); // doubling
      this->setRemoveIsolatedNodes(true);
      this->setPruneModelNum(10);
      this->setCheckCompleteModelInputToOutput(true);
      this->setNEpochsTraining(1001);
      this->setSelectModels(true);
    }
    else {
      this->setNTop(models.size());
      this->setNRandom(models.size());
      this->setNReplicatesPerModel(1); // doubling
      this->setRemoveIsolatedNodes(false);
      this->setPruneModelNum(0);
      this->setCheckCompleteModelInputToOutput(false);
      this->setNEpochsTraining(0);
      this->setSelectModels(false);
    }
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainerExperimental<TensorT, InterpreterT>::setTrainingStepsByModelSize(std::vector<Model<TensorT>>& models) {
    // Calculate the average model size
    TensorT mean_model_size = 0;
    for (Model<TensorT>& model : models) {
      int links = model.getLinksMap().size();
      mean_model_size += links;
    }
    mean_model_size = mean_model_size / models.size();

    // Adjust the number of training steps
    if (mean_model_size <= 8)
      this->setNEpochsTraining(100);
    else if (mean_model_size <= 16)
      this->setNEpochsTraining(200);
    else if (mean_model_size <= 32)
      this->setNEpochsTraining(400);
    else if (mean_model_size <= 64)
      this->setNEpochsTraining(800);
  }
}
#endif //SMARTPEAK_POPULATIONTRAINEREXPERIMENTAL_H