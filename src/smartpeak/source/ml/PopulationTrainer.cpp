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
    std::vector<Model>& models,
    ModelTrainer& model_trainer,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // [TODO: refactor to its own method]
    // score the models
    std::map<std::string, float> models_validation_errors;
    for (int i=0; i<models.size(); ++i)
    {
      try
      {
        std::vector<float> model_errors = model_trainer.validateModel(
          models[i], input, output, time_steps,
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
    // [TODO: add test that models_validation_errors has expected keys and values]

    // [TODO: refactor to its own method]
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

    // [TODO: add test to check the correct sorting]
    // for(auto p: pairs) printf("Sorted models: model %s\t%.2f\n", p.first.data(), p.second);

    // [TODO: refactor to its own method]
    // select the top N from the models
    std::vector<std::string> top_n_model_names;
    for (int i=0; i<n_top; ++i) {top_n_model_names.push_back(pairs[i].first);}
    std::vector<Model> top_n_models;
    for (int i=0; i<n_top; ++i)
      for (int j=0; j<models.size(); ++j)
        if (models[j].getName() == top_n_model_names[i])
          top_n_models.push_back(models[j]);

    // [TODO: add test to check the correct top N]
    // printf("Top N models size: %d", top_n_models.size());

    // [TODO: refactor to its own method]
    // select a random subset of the top N    
    std::random_device seed;
    std::mt19937 engine(seed());
    std::shuffle(top_n_models.begin(), top_n_models.end(), engine);
    std::vector<Model> random_n_models;
    for (int i=0; i<n_random; ++i) {random_n_models.push_back(top_n_models[i]);}

    // [TODO: add test to check the correct models]
    // printf("Top N random models size: %d", random_n_models.size());

    return random_n_models;
  }

  void PopulationTrainer::replicateModels(
    std::vector<Model>& models,
    ModelReplicator& model_replicator,
    const int& n_replicates_per_model)
  {
    // printf("Models size: %i\t", models.size());
    // replicate and modify
    std::vector<Model> models_copy = models;
    for (const Model model: models_copy)
    {
      for (int i=0; i<n_replicates_per_model; ++i)
      {
        // printf("Modifying model %s iteration %d\n", model.getName().data(), i);
        Model model_copy(model);
        
        //[TODO: add check to make sure that all names in `models` are unique]
        // rename the model
        char model_name_char[128];
        sprintf(model_name_char, "%s@replicateModel:", model.getName().data());
        std::string model_name = model_replicator.makeUniqueHash(model_name_char, std::to_string(i));
        model_copy.setName(model_name);

        model_replicator.modifyModel(model_copy, std::to_string(i));
        model_copy.initWeights();
        models.push_back(model_copy);
      }
    } 
    // [TODO: add test for size change]
    // [TODO: add test for new names]
    // printf("Models size: %i\n", models.size());
  }

  void PopulationTrainer::trainModels(
    std::vector<Model>& models,
    ModelTrainer& model_trainer,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // train the models
    for (int i=0; i<models.size(); ++i)
    {
      try
      {
        model_trainer.trainModel(
          models[i], input, output, time_steps,
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