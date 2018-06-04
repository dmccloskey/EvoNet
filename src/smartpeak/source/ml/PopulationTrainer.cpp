/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>

#include <random> // random number generator
#include <ctime> // time format
#include <chrono> // current time
#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <utility>

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
    std::vector<std::pair<std::string, float>> models_validation_errors;
    // [TODO: refactor to its own method]
    // score the models
    models_validation_errors = validateModels_(
      models, model_trainer, input, output, time_steps, input_nodes, output_nodes
    );
    
    // [TODO: refactor to its own method]
    // sort each model based on their scores in ascending order
    models_validation_errors = getTopNModels_(
      models_validation_errors, n_top
    );

    // [TODO: refactor to its own method]
    // select a random subset of the top N
    models_validation_errors = getRandomModels_(
      models_validation_errors, n_random
    );
    
    std::vector<Model> selected_models;
    for (const std::pair<std::string, float>& model_error: models_validation_errors)
      for (int j=0; j<models.size(); ++j)
        if (models[j].getName() == model_error.first)
          selected_models.push_back(models[j]);

    return selected_models;
  }

 std::vector<std::pair<std::string, float>> PopulationTrainer::validateModels_(
    std::vector<Model>& models,
    ModelTrainer& model_trainer,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // score the models
    std::vector<std::pair<std::string, float>> models_validation_errors;
    for (int i=0; i<models.size(); ++i)
    {
      try
      {
        std::vector<float> model_errors = model_trainer.validateModel(
          models[i], input, output, time_steps,
          input_nodes, output_nodes);
        float model_ave_error = 1e6;
        if (model_errors.size()>0)
          model_ave_error = accumulate(model_errors.begin(), model_errors.end(), 0.0)/model_errors.size();
        models_validation_errors.push_back(std::make_pair(models[i].getName(), model_ave_error));
      }
      catch (std::exception& e)
      {
        printf("The model %s is broken.\n", models[i].getName().data());
        models_validation_errors.push_back(std::make_pair(models[i].getName(),1e6f));
      }
    }
    // [TODO: add test that models_validation_errors has expected keys and values]

    return models_validation_errors;    
  }

  std::vector<std::pair<std::string, float>> PopulationTrainer::getTopNModels_(
    std::vector<std::pair<std::string, float>> model_validation_scores,
    const int& n_top)
  {
    // sort each model based on their scores in ascending order
    std::sort(
      model_validation_scores.begin(), model_validation_scores.end(), 
      [=](std::pair<std::string, float>& a, std::pair<std::string, float>& b)
      {
        return a.second < b.second;
      }
    );

    // [TODO: add test to check the correct sorting]
    // for(auto p: pairs) printf("Sorted models: model %s\t%.2f\n", p.first.data(), p.second);

    // select the top N from the models
    int n_ = n_top;
    if (n_ > model_validation_scores.size())
      n_ = model_validation_scores.size();
      
    std::vector<std::pair<std::string, float>> top_n_models;
    for (int i=0; i<n_; ++i) {top_n_models.push_back(model_validation_scores[i]);}

    // [TODO: add test to check the correct top N]
    // printf("Top N models size: %d", top_n_models.size());

    return top_n_models;
  }

  std::vector<std::pair<std::string, float>> PopulationTrainer::getRandomModels_(
    std::vector<std::pair<std::string, float>> model_validation_scores,
    const int& n_random)
  {
    int n_ = n_random;
    if (n_ > model_validation_scores.size())
      n_ = model_validation_scores.size();

    // select a random subset of the top N
    std::random_device seed;
    std::mt19937 engine(seed());
    std::shuffle(model_validation_scores.begin(), model_validation_scores.end(), engine);
    std::vector<std::pair<std::string, float>> random_n_models;
    for (int i=0; i<n_; ++i) {random_n_models.push_back(model_validation_scores[i]);}

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
        std::regex re("@");
        std::vector<std::string> str_tokens;
        std::string model_name_new = model.getName();
        std::copy(
          std::sregex_token_iterator(model_name_new.begin(), model_name_new.end(), re, -1),
          std::sregex_token_iterator(),
          std::back_inserter(str_tokens));
        if (str_tokens.size() > 1)
          model_name_new = str_tokens[0]; // only retain the last timestamp

        char model_name_char[128];
        sprintf(model_name_char, "%s@replicateModel:", model_name_new.data());
        std::string model_name = model_replicator.makeUniqueHash(model_name_char, std::to_string(i));
        model_copy.setName(model_name);

        model_replicator.modifyModel(model_copy, std::to_string(i));
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