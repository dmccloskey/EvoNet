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

    /// Overrides and members used in all examples
    bool set_population_size_fixed_ = false;
    bool set_population_size_doubling_ = false;
    bool set_training_steps_by_model_size_ = false;

    /*
    @brief Implementation of the `adaptivePopulationScheduler`
    */
    void adaptivePopulationScheduler(const int& n_generations, std::vector<Model<TensorT>>& models, std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) override;
    
    /*
    @brief Implementation of the `trainingPopulationLogger`
    */
    void trainingPopulationLogger(const int& n_generations, std::vector<Model<TensorT>>& models, PopulationLogger<TensorT>& population_logger, const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation) override;
    
    /*
    @brief `adaptivePopulationScheduler` helper method to adjust the population size based on the number of generations
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
    @brief `adaptivePopulationScheduler` helper method to adjust the population size for growth and selection modes
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
  private: 
    int n_top__;
    int n_random__;
    int prune_model_num__;
  };
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainerExperimental<TensorT, InterpreterT>::adaptivePopulationScheduler(const int& n_generations, std::vector<Model<TensorT>>& models, std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
  {
    // Adjust the population size
    if (set_population_size_fixed_) this->setPopulationSizeFixed(n_generations, models, models_errors_per_generations);
    else if (set_population_size_doubling_) this->setPopulationSizeDoubling(n_generations, models, models_errors_per_generations);

    // Adjust the training steps
    if (set_training_steps_by_model_size_) this->setTrainingStepsByModelSize(models);
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainerExperimental<TensorT, InterpreterT>::trainingPopulationLogger(const int& n_generations, std::vector<Model<TensorT>>& models, PopulationLogger<TensorT>& population_logger, const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation)
  {		
    // Export the selected models
    for (auto& model : models) {
      ModelFile<TensorT> data;
      data.storeModelCsv(model.getName() + "_" + std::to_string(n_generations) + "_nodes.csv",
        model.getName() + "_" + std::to_string(n_generations) + "_links.csv",
        model.getName() + "_" + std::to_string(n_generations) + "_weights.csv", model);
    }
    // Log the population statistics
    population_logger.writeLogs(n_generations, models_validation_errors_per_generation);
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainerExperimental<TensorT, InterpreterT>::setPopulationSizeFixed(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) {
    // Adjust the population sizes
    const size_t selection_ratio = this->getPopulationSize() / this->getNRandom();
    if (n_generations == 0) {
      this->setNReplicatesPerModel(this->getPopulationSize() - 1);
    }
    else {
      this->setNReplicatesPerModel(selection_ratio - 1);
    }
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainerExperimental<TensorT, InterpreterT>::setPopulationSizeDoubling(
    const int& n_generations,
    std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) {
    // Save the initial top/random selection sizes
    if (n_generations == 0) {
      n_top__ = this->getNTop();
      n_random__ = this->getNRandom();
      prune_model_num__ = this->getPruneModelNum();
    }
    // Adjust the population sizes
    if (models.size() >= this->getPopulationSize()) {
      this->setNTop(n_top__);
      this->setNRandom(n_random__);
      this->setNReplicatesPerModel(1); // doubling
      this->setRemoveIsolatedNodes(true);
      this->setPruneModelNum(prune_model_num__);
      this->setCheckCompleteModelInputToOutput(true);
      this->setNEpochsTraining(1001); // NOTE: this will be overwritten (so long as it is greater than 0) by the value of `ModelTrainer::n_epochs_training` during the call to `updateNEpochsTraining`
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