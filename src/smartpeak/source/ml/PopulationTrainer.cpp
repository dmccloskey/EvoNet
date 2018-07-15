/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>

#include <random> // random number generator
#include <ctime> // time format
#include <chrono> // current time
#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <utility>
#include <numeric> // accumulate
#include <thread>
#include <future>

namespace SmartPeak
{
  PopulationTrainer::PopulationTrainer(){};
  PopulationTrainer::~PopulationTrainer(){};

  void PopulationTrainer::removeDuplicateModels(std::vector<Model>& models)
  {
    std::map<std::string, Model> unique_models;
    for (const Model& model: models)
      unique_models.emplace(model.getName(), model);    
    
    if (unique_models.size() < models.size())
    {
      models.clear();
      for (const auto& model: unique_models)
      {
        models.push_back(model.second);
      }
    }
    
    // models.erase(std::unique(models.begin(), models.end(), 
    //   [=](const Model& a, const Model& b)
    //     {
    //       printf("Model a %s Model b %s are equal? ", a.getName().data(), b.getName().data());
    //       bool areequal = a.getName() == b.getName();
    //       std::cout<<areequal<<std::endl;
    //       return a.getName() == b.getName();
    //     }),
    //   models.end()
    // );
  }

  void PopulationTrainer::selectModels(
    const int& n_top,
    const int& n_random,
    std::vector<Model>& models,
    ModelTrainer& model_trainer,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes,
    int n_threads)
  {
    // printf("PopulationTrainer::selectModels, Models size: %i\n", models.size());
    // score the models
    std::vector<std::pair<std::string, float>> models_validation_errors;

    // models_validation_errors = validateModels_(
    //   models, model_trainer, input, output, time_steps, input_nodes, output_nodes
    // );

    std::vector<std::future<std::pair<std::string, float>>> task_results;
    int thread_cnt = 0;
    for (int i=0; i<models.size(); ++i)
    {

      std::packaged_task<std::pair<std::string, float> // encapsulate in a packaged_task
        (Model*,
          ModelTrainer*,
          Eigen::Tensor<float, 4>,
          Eigen::Tensor<float, 3>,
          Eigen::Tensor<float, 3>,
          std::vector<std::string>,
          std::vector<std::string>
        )> task(PopulationTrainer::validateModel_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &models[i], &model_trainer, 
        std::ref(input), std::ref(output), std::ref(time_steps), 
        std::ref(input_nodes), std::ref(output_nodes));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || i == models.size() - 1)
      {
        for (auto& task_result: task_results)
        {
          models_validation_errors.push_back(task_result.get());
        }
        task_results.clear();
        thread_cnt = 0;
      }
      else
      {
        ++thread_cnt;
      }      
    }
    // printf("PopulationTrainer::selectModels, models_validation_errors1 size: %i\n", models_validation_errors.size());
    
    // sort each model based on their scores in ascending order
    models_validation_errors = getTopNModels_(
      models_validation_errors, n_top
    );
    // printf("PopulationTrainer::selectModels, models_validation_errors2 size: %i\n", models_validation_errors.size());

    // select a random subset of the top N
    models_validation_errors = getRandomNModels_(
      models_validation_errors, n_random
    );
    // printf("PopulationTrainer::selectModels, models_validation_errors3 size: %i\n", models_validation_errors.size());
    
    std::vector<std::string> selected_models;
    for (const std::pair<std::string, float>& model_error: models_validation_errors)
      selected_models.push_back(model_error.first);

    // purge non-selected models
    if (selected_models.size() != models.size())
    {
      models.erase(
        std::remove_if(models.begin(), models.end(),
          [=](const Model& model)
          {
            return std::count(selected_models.begin(), selected_models.end(), model.getName()) == 0;
          }
        ),
        models.end()
      );
      // printf("PopulationTrainer::selectModels, Models size: %i\n", models.size());
    }

    if (models.size() > n_random)
      removeDuplicateModels(models);
    // printf("PopulationTrainer::selectModels, Models size: %i\n", models.size());
  }

  // [DEPRECATED]
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
          model_ave_error = std::accumulate(model_errors.begin(), model_errors.end(), 0.0)/model_errors.size();
        if (isnan(model_ave_error))
          model_ave_error = 1e6;
        models_validation_errors.push_back(std::make_pair(models[i].getName(), model_ave_error));
      }
      catch (std::exception& e)
      {
        printf("The model %s is broken.\n", models[i].getName().data());
        printf("Error: %s.\n", e.what());
        models_validation_errors.push_back(std::make_pair(models[i].getName(),1e6f));
      }
    }
    // [TODO: add test that models_validation_errors has expected keys and values]

    return models_validation_errors;    
  }

  std::pair<std::string, float> PopulationTrainer::validateModel_(
    Model* model,
    ModelTrainer* model_trainer,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // score the model
    try
    {
      std::vector<float> model_errors = model_trainer->validateModel(
        *model, input, output, time_steps,
        input_nodes, output_nodes);
      float model_ave_error = 1e6;
      if (model_errors.size()>0)
        model_ave_error = std::accumulate(model_errors.begin(), model_errors.end(), 0.0)/model_errors.size();
      if (isnan(model_ave_error))
        model_ave_error = 1e6;      

      char cout_char[512];
      sprintf(cout_char, "Model %s (Nodes: %d, Links: %d) error: %.2f\n", 
        model->getName().data(), model->getNodes().size(), model->getLinks().size(), model_ave_error);
      std::cout<<cout_char;

      return std::make_pair(model->getName(), model_ave_error);
    }
    catch (std::exception& e)
    {
      printf("The model %s is broken.\n", model->getName().data());
      printf("Error: %s.\n", e.what());
      return std::make_pair(model->getName(),1e6f);
    }  
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

    // select the top N from the models
    int n_ = n_top;
    if (n_ > model_validation_scores.size())
      n_ = model_validation_scores.size();
      
    std::vector<std::pair<std::string, float>> top_n_models;
    for (int i=0; i<n_; ++i) {top_n_models.push_back(model_validation_scores[i]);}

    return top_n_models;
  }

  std::vector<std::pair<std::string, float>> PopulationTrainer::getRandomNModels_(
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

    return random_n_models;
  }

  void PopulationTrainer::replicateModels(
    std::vector<Model>& models,
    ModelReplicator& model_replicator,
    const int& n_replicates_per_model,
    std::string unique_str,
    int n_threads)
  {
    // replicate and modify
    std::vector<Model> models_copy = models;
    int cnt = 0;
    std::vector<std::future<Model>> task_results;
    int thread_cnt = 0;
    for (Model& model: models_copy)
    {
      for (int i=0; i<n_replicates_per_model; ++i)
      {
        // models.push_back(replicateModel_(&model, &model_replicator, unique_str, cnt, i));

        std::packaged_task<Model // encapsulate in a packaged_task
          (Model*, ModelReplicator*, std::string, int, int
          )> task(PopulationTrainer::replicateModel_);
        
        // launch the thread
        task_results.push_back(task.get_future());
        std::thread task_thread(std::move(task),
          &model, &model_replicator, 
          std::ref(unique_str), std::ref(cnt), std::ref(i));
        task_thread.detach();

        // retreive the results
        if (thread_cnt == n_threads - 1 || cnt == models_copy.size() - 1)
        {
          for (auto& task_result: task_results)
          {
            models.push_back(task_result.get());
          }
          task_results.clear();
          thread_cnt = 0;
        }
        else
        {
          ++thread_cnt;
        }

        cnt += 1;
      }
    } 

    // removeDuplicateModels(models);  // safer to use, but does hurt performance
  }

  Model PopulationTrainer::replicateModel_(
    Model* model,
    ModelReplicator* model_replicator,
    std::string unique_str, int cnt, int i)
  {
    Model model_copy(*model);
    
    // rename the model
    std::regex re("@");
    std::vector<std::string> str_tokens;
    std::string model_name_new = model->getName();
    std::copy(
      std::sregex_token_iterator(model_name_new.begin(), model_name_new.end(), re, -1),
      std::sregex_token_iterator(),
      std::back_inserter(str_tokens));
    if (str_tokens.size() > 1)
      model_name_new = str_tokens[0]; // only retain the last timestamp

    char model_name_char[128];
    sprintf(model_name_char, "%s@replicateModel:%s", model_name_new.data(), unique_str.data());
    std::string model_name = model_replicator->makeUniqueHash(model_name_char, std::to_string(cnt));
    model_copy.setName(model_name);

    model_replicator->modifyModel(model_copy, unique_str + "-" + std::to_string(i));
    model_copy.pruneModel(1);
    return model_copy; // return the model not the pointer
  }

  void PopulationTrainer::trainModels(
    std::vector<Model>& models,
    ModelTrainer& model_trainer,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes,
    int n_threads)
  {
    // std::vector<std::string> broken_model_names;
    std::vector<Model> trained_models;
    std::vector<std::future<std::pair<bool, Model>>> task_results;
    int thread_cnt = 0;

    // train the models
    for (int i=0; i<models.size(); ++i)
    {
      // std::pair<std::string, bool> status = trainModel_(
      //   &models[i], &model_trainer, input, output, time_steps, input_nodes, output_nodes);         
      // if (!status.second)
      // {
      //   broken_model_names.push_back(status.first);
      // }

      // std::pair<bool, Model> status = trainModel_(
      //   &models[i], &model_trainer, input, output, time_steps, input_nodes, output_nodes);  
      // if (status.first)
      //   trained_models.push_back(status.second);

      std::packaged_task<std::pair<bool, Model> // encapsulate in a packaged_task
        (Model*,
          ModelTrainer*,
          Eigen::Tensor<float, 4>,
          Eigen::Tensor<float, 3>,
          Eigen::Tensor<float, 3>,
          std::vector<std::string>,
          std::vector<std::string>
        )> task(PopulationTrainer::trainModel_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &models[i], &model_trainer, 
        std::ref(input), std::ref(output), std::ref(time_steps), 
        std::ref(input_nodes), std::ref(output_nodes));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || i == models.size() - 1)
      {
        for (auto& task_result: task_results)
        // for (int j=0; j<task_results.size(); ++j)
        {
          std::pair<bool, Model> status = task_result.get();  
          // std::pair<bool, Model> status status = task_results[j].get();         
          if (status.first)
          {
            trained_models.push_back(status.second);
          }
        }
        task_results.clear();
        thread_cnt = 0;
      }
      else
      {
        ++thread_cnt;
      }
    }

    // update models
    models = trained_models;

    // // purge broken models
    // if (broken_model_names.size() > 0)
    // {
    //   models.erase(
    //     std::remove_if(models.begin(), models.end(),
    //       [=](const Model& model)
    //       {
    //         return std::count(broken_model_names.begin(), broken_model_names.end(), model.getName()) != 0;
    //       }
    //     ),
    //     models.end()
    //   );
    // }
  }
  
  std::pair<bool, Model> PopulationTrainer::trainModel_(
    Model* model,
    ModelTrainer* model_trainer,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    Model model_copy(*model);
    try
    {
      model_trainer->trainModel(
        model_copy, input, output, time_steps,
        input_nodes, output_nodes);
      return std::make_pair(true, model_copy);
    }
    catch (std::exception& e)
    {
      printf("The model %s is broken.\n", model_copy.getName().data());
      printf("Error: %s.\n", e.what());
      return std::make_pair(false, model_copy);
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