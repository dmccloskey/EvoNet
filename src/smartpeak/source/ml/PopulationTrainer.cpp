/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>

#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/ModelFile.h>

#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <utility>
#include <numeric> // accumulate
#include <thread>
#include <future>
#include <mutex>

static std::mutex trainModel_mutex;
static std::mutex validateModel_mutex;
static std::mutex replicateModel_mutex;
static std::mutex evalModel_mutex;

namespace SmartPeak
{
	template<typename TensorT>
	void PopulationTrainer<TensorT>::setNTop(const int & n_top)
	{
		n_top_ = n_top;
	}
	template<typename TensorT>
	void PopulationTrainer<TensorT>::setNRandom(const int & n_random)
	{
		n_random_ = n_random;
	}
	template<typename TensorT>
	void PopulationTrainer<TensorT>::setNReplicatesPerModel(const int & n_replicates_per_model)
	{
		n_replicates_per_model_ = n_replicates_per_model;
	}
	template<typename TensorT>
	void PopulationTrainer<TensorT>::setNGenerations(const int & n_generations)
	{
		n_generations_ = n_generations;
	}
	template<typename TensorT>
	int PopulationTrainer<TensorT>::getNTop() const
	{
		return n_top_;
	}
	template<typename TensorT>
	int PopulationTrainer<TensorT>::getNRandom() const
	{
		return n_random_;
	}
	template<typename TensorT>
	int PopulationTrainer<TensorT>::getNReplicatesPerModel() const
	{
		return n_replicates_per_model_;
	}
	template<typename TensorT>
	int PopulationTrainer<TensorT>::getNGenerations() const
	{
		return n_generations_;
	}

	template<typename TensorT>
  void PopulationTrainer<TensorT>::removeDuplicateModels(std::vector<Model<TensorT>>& models)
  {
    std::map<std::string, Model<TensorT>> unique_models;
    for (const Model<TensorT>& model: models)
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
    //   [=](const Model<TensorT>& a, const Model<TensorT>& b)
    //     {
    //       printf("Modela %s Model*b %s are equal? ", a.getName().data(), b.getName().data());
    //       bool areequal = a.getName() == b.getName();
    //       std::cout<<areequal<<std::endl;
    //       return a.getName() == b.getName();
    //     }),
    //   models.end()
    // );
  }

	template<typename TensorT>
	std::vector<std::pair<int, TensorT>> PopulationTrainer<TensorT>::selectModels(
    std::vector<Model<TensorT>>& models,
    ModelTrainer<TensorT>& model_trainer,
		ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 4>& input,
    const Eigen::Tensor<TensorT, 4>& output,
    const Eigen::Tensor<TensorT, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    int n_threads)
  {
    // printf("PopulationTrainer<TensorT>::selectModels, Models size: %i\n", models.size());
    // score the models
    std::vector<std::pair<int, TensorT>> models_validation_errors;

    // models_validation_errors = validateModels_(
    //   models, model_trainer, input, output, time_steps, input_nodes, output_nodes
    // );

    std::vector<std::future<std::pair<int, TensorT>>> task_results;
    int thread_cnt = 0;
    for (int i=0; i<models.size(); ++i)
    {

      std::packaged_task<std::pair<int, TensorT> // encapsulate in a packaged_task
        (Model<TensorT>*,
          ModelTrainer<TensorT>*,
					ModelLogger<TensorT>*,
          Eigen::Tensor<TensorT, 4>,
          Eigen::Tensor<TensorT, 4>,
          Eigen::Tensor<TensorT, 3>,
          std::vector<std::string>
        )> task(PopulationTrainer<TensorT>::validateModel_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &models[i], &model_trainer, &model_logger,
        std::ref(input), std::ref(output), std::ref(time_steps), 
        std::ref(input_nodes));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || i == models.size() - 1)
      {
        for (auto& task_result: task_results)
        {
          if (task_result.valid())
          {
            try
            {
              models_validation_errors.push_back(task_result.get());
            }
            catch (std::exception& e)
            {
              printf("Exception: %s", e.what());
            }
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
    // printf("PopulationTrainer<TensorT>::selectModels, models_validation_errors1 size: %i\n", models_validation_errors.size());
    
    // sort each model based on their scores in ascending order
    models_validation_errors = getTopNModels_(
      models_validation_errors, getNTop()
    );
    // printf("PopulationTrainer<TensorT>::selectModels, models_validation_errors2 size: %i\n", models_validation_errors.size());

    // select a random subset of the top N
    models_validation_errors = getRandomNModels_(
      models_validation_errors, getNRandom()
    );
    // printf("PopulationTrainer<TensorT>::selectModels, models_validation_errors3 size: %i\n", models_validation_errors.size());
    
    std::vector<int> selected_models;
    for (const std::pair<int, TensorT>& model_error: models_validation_errors)
      selected_models.push_back(model_error.first);

    // purge non-selected models
    if (selected_models.size() != models.size())
    {
      models.erase(
        std::remove_if(models.begin(), models.end(),
          [=](const Model<TensorT>& model)
          {
            return std::count(selected_models.begin(), selected_models.end(), model.getId()) == 0;
          }
        ),
        models.end()
      );
      // printf("PopulationTrainer<TensorT>::selectModels, Models size: %i\n", models.size());
    }

    if (models.size() > getNRandom())
      removeDuplicateModels(models);
    // printf("PopulationTrainer<TensorT>::selectModels, Models size: %i\n", models.size());

		return models_validation_errors;
  }

	template<typename TensorT>
  std::pair<int, TensorT> PopulationTrainer<TensorT>::validateModel_(
    Model<TensorT>* model,
    ModelTrainer<TensorT>* model_trainer,
		ModelLogger<TensorT>* model_logger,
    const Eigen::Tensor<TensorT, 4>& input,
    const Eigen::Tensor<TensorT, 4>& output,
    const Eigen::Tensor<TensorT, 3>& time_steps,
    const std::vector<std::string>& input_nodes)
  {
    std::lock_guard<std::mutex> lock(validateModel_mutex);
    // score the model
    try
    {
      std::vector<TensorT> model_errors = model_trainer->validateModel(
        *model, input, output, time_steps,
        input_nodes, *model_logger);
      TensorT model_ave_error = 1e6;
      if (model_errors.size()>0)
        model_ave_error = std::accumulate(model_errors.begin(), model_errors.end(), 0.0)/model_errors.size();
      if (isnan(model_ave_error))
        model_ave_error = 1e32; // a large number

      char cout_char[512];
      sprintf(cout_char, "Model%s (Nodes: %d, Links: %d) error: %.6f\n", 
        model->getName().data(), model->getNodes().size(), model->getLinks().size(), model_ave_error);
      std::cout<<cout_char;

      return std::make_pair(model->getId(), model_ave_error);
    }
    catch (std::exception& e)
    {
      printf("The model %s is broken.\n", model->getName().data());
      printf("Error: %s.\n", e.what());
      return std::make_pair(model->getId(),1e6f);
    }  
  }

	template<typename TensorT>
  std::vector<std::pair<int, TensorT>> PopulationTrainer<TensorT>::getTopNModels_(
    std::vector<std::pair<int, TensorT>> model_validation_scores,
    const int& n_top)
  {
    // sort each model based on their scores in ascending order
    std::sort(
      model_validation_scores.begin(), model_validation_scores.end(), 
      [=](std::pair<int, TensorT>& a, std::pair<int, TensorT>& b)
      {
        return a.second < b.second;
      }
    );

    // select the top N from the models
    int n_ = n_top;
    if (n_ > model_validation_scores.size())
      n_ = model_validation_scores.size();
      
    std::vector<std::pair<int, TensorT>> top_n_models;
    for (int i=0; i<n_; ++i) {top_n_models.push_back(model_validation_scores[i]);}

    return top_n_models;
  }

	template<typename TensorT>
  std::vector<std::pair<int, TensorT>> PopulationTrainer<TensorT>::getRandomNModels_(
    std::vector<std::pair<int, TensorT>> model_validation_scores,
    const int& n_random)
  {
    int n_ = n_random;
    if (n_ > model_validation_scores.size())
      n_ = model_validation_scores.size();

    // select a random subset of the top N
    std::random_device seed;
    std::mt19937 engine(seed());
    std::shuffle(model_validation_scores.begin(), model_validation_scores.end(), engine);
    std::vector<std::pair<int, TensorT>> random_n_models;
    for (int i=0; i<n_; ++i) {random_n_models.push_back(model_validation_scores[i]);}

    return random_n_models;
  }

	template<typename TensorT>
  void PopulationTrainer<TensorT>::replicateModels(
    std::vector<Model<TensorT>>& models,
    ModelReplicator<TensorT>& model_replicator,
    std::string unique_str,
    int n_threads)
  {
    // replicate and modify
    std::vector<Model<TensorT>> models_copy = models;
    int cnt = 0;
    std::vector<std::future<Model<TensorT>>> task_results;
    int thread_cnt = 0;
    for (Model<TensorT>& model: models_copy)
    {
      for (int i=0; i<getNReplicatesPerModel(); ++i)
      {
        std::packaged_task<Model<TensorT>// encapsulate in a packaged_task
          (Model<TensorT>*, ModelReplicator<TensorT>*, 
						std::string, int
          )> task(PopulationTrainer<TensorT>::replicateModel_);
        
        // launch the thread
        task_results.push_back(task.get_future());
        std::thread task_thread(std::move(task),
          &model, &model_replicator, 
          std::ref(unique_str), std::ref(cnt));
        task_thread.detach();

        // retreive the results
        if (thread_cnt == n_threads - 1 || cnt == models_copy.size() - 1)
        {
          for (auto& task_result: task_results)
          {       
            if (task_result.valid())
            {
              try
              {
								Model<TensorT> model_task_result = task_result.get();
								model_task_result.setId(getNextID());
                models.push_back(model_task_result);
              }              
              catch (std::exception& e)
              {
                printf("Exception: %s", e.what());
              }
            }
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

	template<typename TensorT>
  Model<TensorT> PopulationTrainer<TensorT>::replicateModel_(
    Model<TensorT>* model,
    ModelReplicator<TensorT>* model_replicator,
    std::string unique_str, int cnt)
  {    
    std::lock_guard<std::mutex> lock(replicateModel_mutex);
    
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
		sprintf(model_name_char, "%s@replicateModel#%s", model_name_new.data(), unique_str.data());
		std::string model_name = model_replicator->makeUniqueHash(model_name_char, std::to_string(cnt));

		int max_iters = 5;
		for (int iter=0; iter<max_iters; ++iter)
		{
			Model<TensorT> model_copy(*model);
			model_copy.setName(model_name);

			model_replicator->makeRandomModifications();
			model_replicator->modifyModel(model_copy, unique_str);

			// model checks
			model_copy.removeIsolatedNodes();
			model_copy.pruneModel(10);

			// additional model checks
			Model<TensorT> model_check(model_copy);
			bool complete_model = model_check.checkCompleteInputToOutput();

			if (complete_model)
				return model_copy;
		}

		throw std::runtime_error("All modified models were broken!");
  }

	template<typename TensorT>
  void PopulationTrainer<TensorT>::trainModels(
    std::vector<Model<TensorT>>& models,
    ModelTrainer<TensorT>& model_trainer,
		ModelLogger<TensorT>& model_logger,
    const Eigen::Tensor<TensorT, 4>& input,
    const Eigen::Tensor<TensorT, 4>& output,
    const Eigen::Tensor<TensorT, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    int n_threads)
  {
    // std::vector<std::string> broken_model_names;
    std::vector<Model<TensorT>> trained_models;
    std::vector<std::future<std::pair<bool, Model<TensorT>>>> task_results;
    int thread_cnt = 0;

    // train the models
    for (int i=0; i<models.size(); ++i)
    {
      std::packaged_task<std::pair<bool, Model<TensorT>> // encapsulate in a packaged_task
        (Model<TensorT>*,
          ModelTrainer<TensorT>*,
					ModelLogger<TensorT>*,
          Eigen::Tensor<TensorT, 4>,
          Eigen::Tensor<TensorT, 4>,
          Eigen::Tensor<TensorT, 3>,
          std::vector<std::string>
        )> task(PopulationTrainer<TensorT>::trainModel_);
      
      // launch the thread
      task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        &models[i], &model_trainer, &model_logger,
        std::ref(input), std::ref(output), std::ref(time_steps), 
        std::ref(input_nodes));
      task_thread.detach();

      // retreive the results
      if (thread_cnt == n_threads - 1 || i == models.size() - 1)
      {
        for (auto& task_result: task_results)
        // for (int j=0; j<task_results.size(); ++j)
        {
          if (task_result.valid())
          {
            try
            {
              std::pair<bool, Model<TensorT>> status = task_result.get();  
              // std::pair<bool, Model<TensorT>> status status = task_results[j].get();         
              if (status.first)
              {
                trained_models.push_back(status.second);
              }
            }
            catch (std::exception& e)
            {
              printf("Exception: %s", e.what());
            }
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
    //       [=](const Model<TensorT>& model)
    //       {
    //         return std::count(broken_model_names.begin(), broken_model_names.end(), model.getName()) != 0;
    //       }
    //     ),
    //     models.end()
    //   );
    // }
  }

	template<typename TensorT>
  std::pair<bool, Model<TensorT>> PopulationTrainer<TensorT>::trainModel_(
    Model<TensorT>* model,
    ModelTrainer<TensorT>* model_trainer,
		ModelLogger<TensorT>* model_logger,
    const Eigen::Tensor<TensorT, 4>& input,
    const Eigen::Tensor<TensorT, 4>& output,
    const Eigen::Tensor<TensorT, 3>& time_steps,
    const std::vector<std::string>& input_nodes)
  {
    std::lock_guard<std::mutex> lock(trainModel_mutex);

    //Model<TensorT> model_copy(*model);
    try
    {
      model_trainer->trainModel(
        //model_copy,
				*model,
				input, output, time_steps,
        input_nodes, *model_logger);
      return std::make_pair(true, *model);
    }
    catch (std::exception& e)
    {
      printf("The model %s is broken.\n", model->getName().data());
      printf("Error: %s.\n", e.what());
      return std::make_pair(false, *model);
    }
  }

	template<typename TensorT>
	void PopulationTrainer<TensorT>::evalModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT>& model_trainer,
		ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes,
		int n_threads)
	{
		// std::vector<std::string> broken_model_names;
		std::vector<std::future<bool>> task_results;
		int thread_cnt = 0;

		// train the models
		for (int i = 0; i < models.size(); ++i)
		{
			std::packaged_task<bool // encapsulate in a packaged_task
				(Model<TensorT>*,
					ModelTrainer<TensorT>*,
					ModelLogger<TensorT>*,
					Eigen::Tensor<TensorT, 4>,
					Eigen::Tensor<TensorT, 3>,
					std::vector<std::string>
					)> task(PopulationTrainer<TensorT>::evalModel_);

			// launch the thread
			task_results.push_back(task.get_future());
			std::thread task_thread(std::move(task),
				&models[i], &model_trainer, &model_logger,
				std::ref(input), std::ref(time_steps),
				std::ref(input_nodes));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == n_threads - 1 || i == models.size() - 1)
			{
				for (auto& task_result : task_results)
				{
					if (task_result.valid())
					{
						try
						{
							bool status = task_result.get();
						}
						catch (std::exception& e)
						{
							printf("Exception: %s", e.what());
						}
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
	}

	template<typename TensorT>
	bool PopulationTrainer<TensorT>::evalModel_(
		Model<TensorT>* model,
		ModelTrainer<TensorT>* model_trainer,
		ModelLogger<TensorT>* model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes)
	{
		std::lock_guard<std::mutex> lock(evalModel_mutex);

		try
		{
			model_trainer->evaluateModel(
				*model, input, time_steps,
				input_nodes, *model_logger);
			return true;
		}
		catch (std::exception& e)
		{
			printf("The model %s is broken.\n", model->getName().data());
			printf("Error: %s.\n", e.what());
			return false;
		}
	}

	template<typename TensorT>
	int PopulationTrainer<TensorT>::getNextID()
	{
		return ++unique_id_;
	}

	template<typename TensorT>
	void PopulationTrainer<TensorT>::setID(const int & id)
	{
		unique_id_ = id;
	}

	template<typename TensorT>
	std::vector<std::vector<std::pair<int, TensorT>>> PopulationTrainer<TensorT>::evolveModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT>& model_trainer,
		ModelReplicator<TensorT>& model_replicator,
		DataSimulator& data_simulator,
		ModelLogger<TensorT>& model_logger,
		const std::vector<std::string>& input_nodes,
		int n_threads)
	{
		std::vector<std::vector<std::pair<int, TensorT>>> models_validation_errors_per_generation;

		std::vector<std::string> output_nodes;
		for (const std::vector<std::string>& output_nodes_vec : model_trainer.getOutputNodes())
			for (const std::string& output_node : output_nodes_vec)
				output_nodes.push_back(output_node);

		// generate the input/output data for validation		
		std::cout << "Generating the input/output data for validation..." << std::endl;
		Eigen::Tensor<TensorT, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsValidation());
		Eigen::Tensor<TensorT, 4> output_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(),(int)output_nodes.size(), model_trainer.getNEpochsValidation());
		Eigen::Tensor<TensorT, 3> time_steps_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsValidation());
		data_simulator.simulateValidationData(input_data_validation, output_data_validation, time_steps_validation);

		// Population initial conditions
		setID(models.size());

		// Evolve the population
		for (int iter = 0; iter<getNGenerations(); ++iter)
		{
			char iter_char[128];
			sprintf(iter_char, "Iteration #: %d\n", iter);
			std::cout << iter_char;

			// Generate the input and output data for training [BUG FREE]
			std::cout << "Generating the input/output data for training..." << std::endl;
			Eigen::Tensor<TensorT, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsTraining());
			Eigen::Tensor<TensorT, 4> output_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochsTraining());
			Eigen::Tensor<TensorT, 3> time_steps_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsTraining());
			data_simulator.simulateTrainingData(input_data_training, output_data_training, time_steps_training);

			// train the population
			std::cout << "Training the models..." << std::endl;
		  trainModels(models, model_trainer, model_logger,
				input_data_training, output_data_training, time_steps_training, input_nodes, n_threads);

			// select the top N from the population
			std::cout << "Selecting the models..." << std::endl;
			std::vector<std::pair<int, TensorT>> models_validation_errors = selectModels(
				models, model_trainer, model_logger,
				input_data_validation, output_data_validation, time_steps_validation, input_nodes, n_threads);
			models_validation_errors_per_generation.push_back(models_validation_errors);

			if (iter < getNGenerations() - 1)
			{
				// update the model replication attributes and population dynamics
				model_replicator.adaptiveReplicatorScheduler(iter, models, models_validation_errors_per_generation);
				adaptivePopulationScheduler(iter, models, models_validation_errors_per_generation);

				// replicate and modify models
				// [TODO: add options for verbosity]
				std::cout << "Replicating and modifying the models..." << std::endl;
				replicateModels(models, model_replicator,	std::to_string(iter), n_threads);
				std::cout << "Population size of " << models.size() << std::endl;
			}
		}
		return models_validation_errors_per_generation;
	}

	template<typename TensorT>
	void PopulationTrainer<TensorT>::evaluateModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT>& model_trainer,
		ModelReplicator<TensorT>& model_replicator,
		DataSimulator& data_simulator,
		ModelLogger<TensorT>& model_logger,
		const std::vector<std::string>& input_nodes,
		int n_threads)
	{
		// generate the input/output data for evaluation		
		std::cout << "Generating the input/output data for evaluation..." << std::endl;
		Eigen::Tensor<TensorT, 4> input_data_evaluation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsEvaluation());
		Eigen::Tensor<TensorT, 3> time_steps_evaluation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsEvaluation());
		data_simulator.simulateEvaluationData(input_data_evaluation, time_steps_evaluation);

		// Population initial conditions
		setID(models.size());

		// Evaluate the population
		std::cout << "Evaluating the model..." << std::endl;
		evalModels(models, model_trainer, model_logger,
			input_data_evaluation, time_steps_evaluation, input_nodes, n_threads);
	}

  // TensorT PopulationTrainer<TensorT>::calculateMean(std::vector<TensorT> values)
  // {
  //   if (values.empty())
  //     return 0;
  //   return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  // }

  // TensorT PopulationTrainer<TensorT>::calculateStdDev(std::vector<TensorT> values)
  // {
  //   if (numbers.size() <= 1u)
  //     return 0;
  //   auto const add_square = [mean](TensorT sum, TensorT i)
  //   {
  //     auto d = i - mean;
  //     return sum + d*d;
  //   };
  //   TensorT total = std::accumulate(numbers.begin(), numbers.end(), 0.0, add_square);
  //   return total / (numbers.size() - 1);
  // }

}