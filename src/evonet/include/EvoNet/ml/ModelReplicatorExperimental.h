/**TODO:  Add copyright*/

#ifndef EVONET_MODELREPLICATOREXPERIMENTAL_H
#define EVONET_MODELREPLICATOREXPERIMENTAL_H

// .h
#include <EvoNet/ml/ModelReplicator.h>

namespace EvoNet
{
  /**
    @brief Experimental methods for `ModelReplicator`
  */
	template<typename TensorT>
  class ModelReplicatorExperimental: public ModelReplicator<TensorT>
  {
public:
    ModelReplicatorExperimental() = default; ///< Default constructor
    ~ModelReplicatorExperimental() = default; ///< Default destructor

    /// Overrides and members used in all examples
    bool set_modification_rate_by_prev_error_ = false;
    bool set_modification_rate_fixed_ = false;

    /*
    @brief Implementation of the `adaptiveReplicatorScheduler`
    */
    void adaptiveReplicatorScheduler(const int& n_generations, std::vector<Model<TensorT>>& models, std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) override;
    
    /*
    @brief Adjust the model replicator modification rate based on a fixed population size error rates

    @param[in] n_generations The number of generations
    @param[in] models A vector of models representing the population
    @param[in] models_errors_per_generations A record of model errors per generation
    */
    void setModificationRateByPrevError(const int& n_generations, std::vector<Model<TensorT>>& models,
      std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations);

    /*
    @brief Set the modification rate

    @param[in] n_generations The number of generations
    @param[in] models A vector of models representing the population
    @param[in] models_errors_per_generations A record of model errors per generation
    */
    void setModificationRateFixed(const int& n_generations, std::vector<Model<TensorT>>& models,
      std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations);

  };
  template<typename TensorT>
  inline void ModelReplicatorExperimental<TensorT>::adaptiveReplicatorScheduler(const int& n_generations, std::vector<Model<TensorT>>& models, std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
  {
    // Adjust the models modifications rates
    if (set_modification_rate_by_prev_error_) this->setModificationRateByPrevError(n_generations, models, models_errors_per_generations);
    if (set_modification_rate_fixed_) this->setModificationRateFixed(n_generations, models, models_errors_per_generations);
  }
  template<typename TensorT>
	void ModelReplicatorExperimental<TensorT>::setModificationRateByPrevError(const int& n_generations, std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
	{
    if (n_generations > 2) {
      // Calculate the mean of the previous and current model errors
      TensorT mean_errors_per_generation_prev = 0, mean_errors_per_generation_cur = 0;
      for (const std::tuple<int, std::string, TensorT>& models_errors : models_errors_per_generations[n_generations - 1])
        mean_errors_per_generation_prev += std::get<2>(models_errors);
      mean_errors_per_generation_prev = mean_errors_per_generation_prev / models_errors_per_generations[n_generations - 1].size();
      for (const std::tuple<int, std::string, TensorT>& models_errors : models_errors_per_generations[n_generations])
        mean_errors_per_generation_cur += std::get<2>(models_errors);
      mean_errors_per_generation_cur = mean_errors_per_generation_cur / models_errors_per_generations[n_generations].size();

      // Lambdas to ensure the lb/ub of random modifications stay within certain limits
      auto clipLinkMod = [](const std::pair<int, int>& value) {
        std::pair<int, int> value_copy = value;
        if (value.second > 32) value_copy.second = 32;
        if (value.first > 16) value_copy.first = 16;
        if (value.second < 4) value_copy.second = 4;
        if (value.first < 0) value_copy.first = 0;
        return value_copy;
      };
      auto clipNodeMod = [](const std::pair<int, int>& value) {
        std::pair<int, int> value_copy = value;
        if (value.second > 16) value_copy.second = 16;
        if (value.first > 8) value_copy.first = 8;
        if (value.second < 2) value_copy.second = 2;
        if (value.first < 0) value_copy.first = 0;
        return value_copy; };

      // update the # of random modifications
      TensorT abs_percent_diff = abs(mean_errors_per_generation_prev - mean_errors_per_generation_cur) / mean_errors_per_generation_prev;
      if (abs_percent_diff < 0.1) {
        this->setRandomModifications(
          clipNodeMod(std::make_pair(this->getRandomModifications()[0].first * 2, this->getRandomModifications()[0].second * 2)),
          clipNodeMod(std::make_pair(this->getRandomModifications()[1].first * 2, this->getRandomModifications()[1].second * 2)),
          std::make_pair(this->getRandomModifications()[2].first * 2, this->getRandomModifications()[2].second * 2),
          std::make_pair(this->getRandomModifications()[3].first * 2, this->getRandomModifications()[3].second * 2),
          clipLinkMod(std::make_pair(this->getRandomModifications()[4].first * 2, this->getRandomModifications()[4].second * 2)),
          std::make_pair(this->getRandomModifications()[5].first * 2, this->getRandomModifications()[5].second * 2),
          clipNodeMod(std::make_pair(this->getRandomModifications()[6].first * 2, this->getRandomModifications()[6].second * 2)),
          clipLinkMod(std::make_pair(this->getRandomModifications()[7].first * 2, this->getRandomModifications()[7].second * 2)),
          clipNodeMod(std::make_pair(this->getRandomModifications()[8].first * 2, this->getRandomModifications()[8].second * 2)),
          clipNodeMod(std::make_pair(this->getRandomModifications()[9].first * 2, this->getRandomModifications()[9].second * 2)),
          std::make_pair(this->getRandomModifications()[10].first * 2, this->getRandomModifications()[10].second * 2),
          std::make_pair(this->getRandomModifications()[11].first * 2, this->getRandomModifications()[11].second * 2),
          std::make_pair(this->getRandomModifications()[12].first * 2, this->getRandomModifications()[12].second * 2));
      }
      else if (abs_percent_diff >= 0.1 && abs_percent_diff < 0.5) {
        // Keep the same parameters
      }
      else {
        this->setRandomModifications(
          clipNodeMod(std::make_pair(this->getRandomModifications()[0].first / 2, this->getRandomModifications()[0].second / 2)),
          clipNodeMod(std::make_pair(this->getRandomModifications()[1].first / 2, this->getRandomModifications()[1].second / 2)),
          std::make_pair(this->getRandomModifications()[2].first / 2, this->getRandomModifications()[2].second / 2),
          std::make_pair(this->getRandomModifications()[3].first / 2, this->getRandomModifications()[3].second / 2),
          clipLinkMod(std::make_pair(this->getRandomModifications()[4].first / 2, this->getRandomModifications()[4].second / 2)),
          std::make_pair(this->getRandomModifications()[5].first / 2, this->getRandomModifications()[5].second / 2),
          clipNodeMod(std::make_pair(this->getRandomModifications()[6].first / 2, this->getRandomModifications()[6].second / 2)),
          clipLinkMod(std::make_pair(this->getRandomModifications()[7].first / 2, this->getRandomModifications()[7].second / 2)),
          clipNodeMod(std::make_pair(this->getRandomModifications()[8].first / 2, this->getRandomModifications()[8].second / 2)),
          clipNodeMod(std::make_pair(this->getRandomModifications()[9].first / 2, this->getRandomModifications()[9].second / 2)),
          std::make_pair(this->getRandomModifications()[10].first / 2, this->getRandomModifications()[10].second / 2),
          std::make_pair(this->getRandomModifications()[11].first / 2, this->getRandomModifications()[11].second / 2),
          std::make_pair(this->getRandomModifications()[12].first / 2, this->getRandomModifications()[12].second / 2));
      }
    }
    else {
      this->setRandomModifications(
        std::make_pair(0, 2),
        std::make_pair(0, 2),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 4),
        std::make_pair(0, 0),
        std::make_pair(0, 2),
        std::make_pair(0, 4),
        std::make_pair(0, 2),
        std::make_pair(0, 2),
        std::make_pair(0, 0),
        std::make_pair(0, 0),
        std::make_pair(0, 0));
    }
	}
  template<typename TensorT>
  void ModelReplicatorExperimental<TensorT>::setModificationRateFixed(const int& n_generations, std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations) {
    this->setRandomModifications(
      std::make_pair(0, 4),
      std::make_pair(0, 4),
      std::make_pair(0, 0),
      std::make_pair(0, 0),
      std::make_pair(0, 8),
      std::make_pair(0, 0),
      std::make_pair(0, 2),
      std::make_pair(0, 4),
      std::make_pair(0, 4),
      std::make_pair(0, 4),
      std::make_pair(0, 0),
      std::make_pair(0, 0),
      std::make_pair(0, 0));
  }
}

#endif //EVONET_MODELREPLICATOREXPERIMENTAL_H