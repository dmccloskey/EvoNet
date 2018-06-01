/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>

#include <random> // random number generator
#include <ctime> // time format
#include <chrono> // current time

namespace SmartPeak
{
  PopulationTrainer::PopulationTrainer(){};
  PopulationTrainer::~PopulationTrainer(){};

  std::vector<Model> PopulationTrainer::selectModels(
    const int& n_top,
    const int& n_random,
    const std::vector<Model>& models)
  {
    // score the models
    std::map<std::string, float> models_validation_errors;
    for (int i=0; i<models.size(); ++i)
    {
      try
      {
        std::vector<float> model_errors = model_trainer.validateModel(
          models[i], input_data, output_data, time_steps,
          input_nodes, output_nodes);
        float model_ave_error = accumulate(model_errors.begin(), model_errors.end(), 0.0)/model_errors.size();
        models_validation_errors.emplace(models[i].getName(), model_ave_error);
      }
      catch (std::exception& e)
      {
        printf("The model %s is broken.\n", models[i].getName().data());
        models_validation_errors.emplace(models[i].getName(), 1e6f);
      }
    }    

    // sort each model based on their scores in ascending order
    std::vector<std::pair<std::string, float>> pairs;
    for (auto itr = models_validation_errors.begin(); itr != models_validation_errors.end(); ++itr)
        pairs.push_back(*itr);

    std::sort(
      pairs.begin(), pairs.end(), 
      [=](std::pair<std::string, float>& a, std::pair<std::string, float>& b)
      {
        return a.second < b.second;
      }
    );

    // select the top N from the models
    std::vector<std::string> top_n_model_names;
    for (int i=0; i<n_top; ++i) {top_n_model_names.push_back(pairs[i].first);}
    std::vector<Model> top_n_models;
    for (int i=0; i<n_top; ++i)
      for (int j=0; j<models.size(); ++j)
        if (models[j].getName() == top_n_model_names[i])
          top_n_models.push_back(models[j]);

    // select a random subset of the top N    
    std::random_device seed;
    std::mt19937 engine(seed());
    std::random_shuffle (top_n_models.begin(), top_n_models.end(), engine);
    std::vector<Model> random_n_models;
    for (int i=0; i<n_random; ++i) {random_n_models.push_back(top_n_models[i]);}
  }

  std::vector<Model> PopulationTrainer::copyModels(const std::vector<Model>& models)
  {
    // [TODO: add method body]
  }

  void PopulationTrainer::modifyModels(
    std::vector<Model>& models,
    const ModelReplicator& model_replicator)
  {
    // [TODO: add method body]  
  }

  void PopulationTrainer::trainModels(std::vector<Model>& models,
    const ModelTrainer& model_trainer)
  {
    // train the models
    for (int i=0; i<models.size(); ++i)
    {
      try
      {
        model_trainer.trainModel(
          models[i], input_data, output_data, time_steps,
          input_nodes, output_nodes);
      }
      catch (std::exception& e)
      {
        printf("The model %s is broken.\n", models[i].getName().data());
        // need to remove the model somehow...
      }
    }    
  }

  // float PopulationTrainer::calculateMean(std::vector<float> values)
  // {
  //   if (values.empty())
  //     return 0;
  //   return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  // }

  // float PopulationTrainer::calculateStdDev(std::vector<float> values)
  // {
  //   if (numbers.size() <= 1u)
  //     return 0;
  //   auto const add_square = [mean](float sum, float i)
  //   {
  //     auto d = i - mean;
  //     return sum + d*d;
  //   };
  //   float total = std::accumulate(numbers.begin(), numbers.end(), 0.0, add_square);
  //   return total / (numbers.size() - 1);
  // }

}