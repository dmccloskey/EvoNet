/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINER_H
#define SMARTPEAK_POPULATIONTRAINER_H

// .h
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/PopulationLogger.h>
#include <SmartPeak/simulator/DataSimulator.h>

// .cpp
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/ModelFile.h>
#include <SmartPeak/io/ModelInterpreterFile.h>
#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <utility>
#include <numeric> // accumulate
#include <thread>
#include <future>
#include <mutex>

namespace SmartPeak
{
  // Concurrency helpers
  static std::mutex trainModel_mutex;
  static std::mutex validateModel_mutex;
  static std::mutex replicateModel_mutex;
  static std::mutex evalModel_mutex;
  std::atomic_size_t train_models_iter_{ 0 };
  std::atomic_size_t validate_models_iter_{ 0 };
  std::atomic_size_t replicate_models_iter_{ 0 };
  std::atomic_size_t eval_models_iter_{ 0 };

  /**
    @brief Class to train a vector of models
  */
	template<typename TensorT, typename InterpreterT>
  class PopulationTrainer
  {
public:
    PopulationTrainer() = default; ///< Default constructor
    ~PopulationTrainer() = default; ///< Default destructor 

		void setNTop(const int& n_top); ///< n_top setter
		void setNRandom(const int& n_random); ///< n_random setter
		void setNReplicatesPerModel(const int& n_replicates_per_model); ///< n_replicates_per_model setter
		void setNGenerations(const int& n_generations); ///< n_generations setter
		void setLogging(bool log_training = false); ///< enable_logging setter
    void setRemoveIsolatedNodes(const bool& remove_isolated_nodes);
    void setPruneModelNum(const int& prune_model_num);
    void setCheckCompleteModelInputToOutput(const bool& check_complete_input_to_output);
    void setSelectModels(const bool& select_models);
    void setResetModelCopyWeights(const bool& reset_model_copy_weights);

		int getNTop() const; ///< batch_size setter
		int getNRandom() const; ///< memory_size setter
		int getNReplicatesPerModel() const; ///< n_epochs setter
		int getNGenerations() const; ///< n_epochs setter
		bool getLogTraining() const; ///< log_training getter
    bool getRemoveIsolatedNodes() const;
    int getPruneModelNum() const;
    bool getCheckCompleteModelInputToOutput() const;
    bool getSelectModels() const;
    bool getResetModelCopyWeights() const;

    /**
      @brief Remove models with non-unique names from the population of models

      @param[in, out] models The vector (i.e., population) of models to select from
    */ 
    void removeDuplicateModels(std::vector<Model<TensorT>>& models);
 
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
		std::vector<std::tuple<int, std::string, TensorT>> selectModels(
      std::vector<Model<TensorT>>& models,
      ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
			ModelLogger<TensorT>& model_logger,
      const Eigen::Tensor<TensorT, 4>& input,
      const Eigen::Tensor<TensorT, 4>& output,
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const std::vector<std::string>& input_nodes);
 
    /**
      @brief validate all of the models

      @returns key value pair of model_name and model_error
    */ 
    static bool validateModels_(
      std::vector<Model<TensorT>>& model,
      ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
      ModelLogger<TensorT>& model_logger,
      const Eigen::Tensor<TensorT, 4>& input,
      const Eigen::Tensor<TensorT, 4>& output,
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      std::vector<std::tuple<int, std::string, TensorT>>& model_validation_errors);
    static std::tuple<int, std::string, TensorT> validateModel_(
      Model<TensorT>& model,
      ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
			ModelLogger<TensorT>& model_logger,
      const Eigen::Tensor<TensorT, 4>& input,
      const Eigen::Tensor<TensorT, 4>& output,
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const std::vector<std::string>& input_nodes);
 
    /**
      @brief return the top N models with the lowest error.

      @returns key value pair of model_name and model_error
    */ 
    static std::vector<std::tuple<int, std::string, TensorT>> getTopNModels_(
      std::vector<std::tuple<int, std::string, TensorT>> model_validation_scores,
      const int& n_top);
 
    /**
      @brief return a random list of model names.

      @returns key value pair of model_name and model_error
    */ 
    static std::vector<std::tuple<int, std::string, TensorT>> getRandomNModels_(
      std::vector<std::tuple<int, std::string, TensorT>> model_validation_scores,
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
      std::vector<Model<TensorT>>& models,
      ModelReplicator<TensorT>& model_replicator,
      std::string unique_str = "",
      int n_threads = 1);

    static std::pair<bool, Model<TensorT>> replicateModel_(
      Model<TensorT>& model,
      ModelReplicator<TensorT>& model_replicator,
      const std::string& unique_str, const int& cnt,
      const bool& remove_isolated_nodes, const int& prune_model_num, const bool& check_complete_input_to_output, const bool& reset_model_copy_weights);
 
    /**
      @brief Trains each of the models in the population
        using the same test data set

      [TESTS: add thread tests]

      @param[in, out] models The vector of models to train
      @param[in] model_trainer The trainer to use
    */ 
    void trainModels(
      std::vector<Model<TensorT>>& models,
      ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
			ModelLogger<TensorT>& model_logger,
      const Eigen::Tensor<TensorT, 4>& input,
      const Eigen::Tensor<TensorT, 4>& output,
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const std::vector<std::string>& input_nodes);

    static bool trainModels_(
      std::vector<Model<TensorT>>& models,
      ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
      ModelLogger<TensorT>& model_logger,
      const Eigen::Tensor<TensorT, 4>& input,
      const Eigen::Tensor<TensorT, 4>& output,
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const std::vector<std::string>& input_nodes);

    static bool trainModel_(
      Model<TensorT>& model,
      ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
			ModelLogger<TensorT>& model_logger,
      const Eigen::Tensor<TensorT, 4>& input,
      const Eigen::Tensor<TensorT, 4>& output,
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const std::vector<std::string>& input_nodes);

		/**
			@brief Evaluates each of the models in the population
				using the same test data set

			[TESTS: add thread tests]

			@param[in, out] models The vector of models to evaluate
			@param[in] model_trainer The trainer to use
		*/
		void evalModels(
			std::vector<Model<TensorT>>& models,
			ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
			ModelLogger<TensorT>& model_logger,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes);

		static bool evalModel_(
			Model<TensorT>& model,
			ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
			ModelLogger<TensorT>& model_logger,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes);

		int getNextID(); ///< iterate and return the next id in the sequence
		void setID(const int& id);  ///< unique_id setter
 
		/**
		@brief Train the population

		@param[in, out] models The vector of models to copy
		@param[in] model_trainer The trainer to use
		@param[in] model_replicator The replicator to use
		@param[in] data_simulator The data simulate/generator to use
		*/
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>> evolveModels(
			std::vector<Model<TensorT>>& models,
			ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
			ModelReplicator<TensorT>& model_replicator,
			DataSimulator<TensorT>& data_simulator,
			ModelLogger<TensorT>& model_logger,
			PopulationLogger<TensorT>& population_logger,
			const std::vector<std::string>& input_nodes);

		/**
		@brief Evaluate the population

		@param[in, out] models The vector of models to copy
		@param[in] model_trainer The trainer to use
		@param[in] model_replicator The replicator to use
		@param[in] data_simulator The data simulate/generator to use
		*/
		void evaluateModels(
			std::vector<Model<TensorT>>& models,
			ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
			ModelReplicator<TensorT>& model_replicator,
			DataSimulator<TensorT>& data_simulator,
			ModelLogger<TensorT>& model_logger,
			const std::vector<std::string>& input_nodes);

		/**
		@brief Entry point for users to code their adaptive scheduler
		to modify models population dynamic parameters based on a given trigger

		@param[in] n_generations The number of evolution generations
		@param[in] models The models in the population
		@param[in] model_errors The trace of models errors from validation at each generation
		*/
		virtual void adaptivePopulationScheduler(
			const int& n_generations,
			std::vector<Model<TensorT>>& models,
			std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations);

		/**
		@brief Entry point for users to code their training logger

		[TODO: add tests]

    @param[in] n_generations The number of evolution generations
    @param[in, out] population_logger The population logger
    @param[in] models_validation_errors_per_generation The model validation errors in the population
		*/
		virtual void trainingPopulationLogger(
      const int& n_generations,
      std::vector<Model<TensorT>>& models,
      PopulationLogger<TensorT>& population_logger,
      const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation);

    void updateNEpochsTraining(ModelTrainer<TensorT, InterpreterT>& model_trainer); ///< Update the number of training epochs
    void setNEpochsTraining(const int& n_epochs); ///< n_epochs setter
    int getNEpochsTraining() const; ///< n_epochs setter

private:
    int unique_id_ = 0;

		// population dynamics
		int n_top_ = 0; ///< The number models to select
		int n_random_ = 0; ///< The number of random models to select from the pool of top models
		int n_replicates_per_model_ = 0; ///< The number of replications per model
		int n_generations_ = 0; ///< The number of generations to evolve the models
    int n_epochs_training_ = -1; ///< The number of epochs to train the models

		bool log_training_ = false;
    bool select_models_ = true; ///< Whether to skip the selection step or not

    // model replicator settings
    bool remove_isolated_nodes_ = true;
    int prune_model_num_ = 10;
    bool check_complete_input_to_output_ = true;
    bool reset_model_copy_weights_ = false;
  };
	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::setNTop(const int & n_top)
	{
		n_top_ = n_top;
	}
	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::setNRandom(const int & n_random)
	{
		n_random_ = n_random;
	}
	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::setNReplicatesPerModel(const int & n_replicates_per_model)
	{
		n_replicates_per_model_ = n_replicates_per_model;
	}
	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::setNGenerations(const int & n_generations)
	{
		n_generations_ = n_generations;
	}
	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::setLogging(bool log_training)
	{
		log_training_ = log_training;
	}
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::setRemoveIsolatedNodes(const bool & remove_isolated_nodes)
  {
    remove_isolated_nodes_ = remove_isolated_nodes;
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::setPruneModelNum(const int & prune_model_num)
  {
    prune_model_num_ = prune_model_num;
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::setCheckCompleteModelInputToOutput(const bool & check_complete_input_to_output)
  {
    check_complete_input_to_output_ = check_complete_input_to_output;
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::setSelectModels(const bool & select_models)
  {
    select_models_ = select_models;
  }
  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::setResetModelCopyWeights(const bool & reset_model_copy_weights)
  {
    reset_model_copy_weights_ = reset_model_copy_weights;
  }
	template<typename TensorT, typename InterpreterT>
	int PopulationTrainer<TensorT, InterpreterT>::getNTop() const
	{
		return n_top_;
	}
	template<typename TensorT, typename InterpreterT>
	int PopulationTrainer<TensorT, InterpreterT>::getNRandom() const
	{
		return n_random_;
	}
	template<typename TensorT, typename InterpreterT>
	int PopulationTrainer<TensorT, InterpreterT>::getNReplicatesPerModel() const
	{
		return n_replicates_per_model_;
	}
	template<typename TensorT, typename InterpreterT>
	int PopulationTrainer<TensorT, InterpreterT>::getNGenerations() const
	{
		return n_generations_;
	}

	template<typename TensorT, typename InterpreterT>
	inline bool PopulationTrainer<TensorT, InterpreterT>::getLogTraining() const
	{
		return log_training_;
	}

  template<typename TensorT, typename InterpreterT>
  inline bool PopulationTrainer<TensorT, InterpreterT>::getRemoveIsolatedNodes() const
  {
    return remove_isolated_nodes_;
  }

  template<typename TensorT, typename InterpreterT>
  inline int PopulationTrainer<TensorT, InterpreterT>::getPruneModelNum() const
  {
    return prune_model_num_;
  }

  template<typename TensorT, typename InterpreterT>
  inline bool PopulationTrainer<TensorT, InterpreterT>::getCheckCompleteModelInputToOutput() const
  {
    return check_complete_input_to_output_;
  }

  template<typename TensorT, typename InterpreterT>
  inline bool PopulationTrainer<TensorT, InterpreterT>::getSelectModels() const
  {
    return select_models_;
  }

  template<typename TensorT, typename InterpreterT>
  inline bool PopulationTrainer<TensorT, InterpreterT>::getResetModelCopyWeights() const
  {
    return reset_model_copy_weights_;
  }

	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::removeDuplicateModels(std::vector<Model<TensorT>>& models)
	{
		std::map<std::string, Model<TensorT>> unique_models;
		for (const Model<TensorT>& model : models)
			unique_models.emplace(model.getName(), model);

		if (unique_models.size() < models.size())
		{
			models.clear();
			for (const auto& model : unique_models)
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

	template<typename TensorT, typename InterpreterT>
	std::vector<std::tuple<int, std::string, TensorT>> PopulationTrainer<TensorT, InterpreterT>::selectModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
		ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 4>& output,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes)
	{
		// score the models
		std::vector<std::tuple<int, std::string, TensorT>> models_validation_errors;

    // Make the list of workers
    std::vector<std::packaged_task<bool
    (std::vector<Model<TensorT>>&,
      ModelTrainer<TensorT, InterpreterT>&, InterpreterT&,
      ModelLogger<TensorT>&,
      const Eigen::Tensor<TensorT, 4>&,
      const Eigen::Tensor<TensorT, 4>&,
      const Eigen::Tensor<TensorT, 3>&,
      const std::vector<std::string>&,
      std::vector<std::tuple<int, std::string, TensorT>>&
      )>> tasks;
    for (size_t i = 0; i < model_interpreters.size(); ++i) {
      std::packaged_task<bool
      (std::vector<Model<TensorT>>&,
        ModelTrainer<TensorT, InterpreterT>&, InterpreterT&,
        ModelLogger<TensorT>&,
        const Eigen::Tensor<TensorT, 4>&,
        const Eigen::Tensor<TensorT, 4>&,
        const Eigen::Tensor<TensorT, 3>&,
        const std::vector<std::string>&,
        std::vector<std::tuple<int, std::string, TensorT>>&
        )> task(PopulationTrainer<TensorT, InterpreterT>::validateModels_);
      tasks.push_back(std::move(task));
    }

    // Launch the tasks asyncronously
    validate_models_iter_ = 0;
    for (int i = 0; i < model_interpreters.size(); ++i) {

      // create a copy of the model logger
      ModelLogger<TensorT> model_logger_copy = model_logger;

      // launch the interpreter
      std::future<bool> task_result = tasks.at(i).get_future();
      std::thread task_thread(std::move(tasks.at(i)),
        std::ref(models), std::ref(model_trainer), std::ref(model_interpreters[i]), std::ref(model_logger_copy),
        std::ref(input), std::ref(output), std::ref(time_steps),
        std::ref(input_nodes),
        std::ref(models_validation_errors));
      task_thread.detach();

      // retreive the results
      try {
        const bool result = task_result.get();
      }
      catch (std::exception & e) {
        printf("Exception: %s", e.what());
      }
    }

		// sort each model based on their scores in ascending order
		models_validation_errors = getTopNModels_(
			models_validation_errors, getNTop()
		);

		// select a random subset of the top N
		models_validation_errors = getRandomNModels_(
			models_validation_errors, getNRandom()
		);

		std::vector<int> selected_models;
		for (const std::tuple<int, std::string, TensorT>& model_error : models_validation_errors)
			selected_models.push_back(std::get<0>(model_error));

		// purge non-selected models
		if (selected_models.size() != models.size())
		{
			models.erase(std::remove_if(models.begin(), models.end(),
					[=](const Model<TensorT>& model){return std::count(selected_models.begin(), selected_models.end(), model.getId()) == 0;}
				),models.end());
		}

		if (models.size() > getNRandom())
			removeDuplicateModels(models);

		return models_validation_errors;
	}

  template<typename TensorT, typename InterpreterT>
  inline bool PopulationTrainer<TensorT, InterpreterT>::validateModels_(std::vector<Model<TensorT>>& models, ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps, const std::vector<std::string>& input_nodes, std::vector<std::tuple<int, std::string, TensorT>>& model_validation_errors)
  {
    bool result = false;
    while (true) {
      // get the next available thread
      const size_t validate_models_iter = validate_models_iter_.fetch_add(1);
      if (validate_models_iter >= models.size()) {
        break;
      }

      // create the task and the future
      std::packaged_task<std::tuple<int, std::string, TensorT>
        (Model<TensorT>&,
          ModelTrainer<TensorT, InterpreterT>&, InterpreterT&,
          ModelLogger<TensorT>&,
          const Eigen::Tensor<TensorT, 4>&,
          const Eigen::Tensor<TensorT, 4>&,
          const Eigen::Tensor<TensorT, 3>&,
          const std::vector<std::string>&
          )> task(PopulationTrainer<TensorT, InterpreterT>::validateModel_);
      std::future<std::tuple<int, std::string, TensorT>> task_result = task.get_future();

      // launch the thread
      std::thread task_thread(std::move(task),
        std::ref(models[validate_models_iter]), std::ref(model_trainer), std::ref(model_interpreter), std::ref(model_logger),
        std::ref(input), std::ref(output), std::ref(time_steps),
        std::ref(input_nodes));
      task_thread.detach();

      // retreive the results
      try {
        model_validation_errors.at(validate_models_iter) = task_result.get();
        result = true;
      }
      catch (std::exception & e) {
        printf("Exception: %s", e.what());
      }
    }
    return result;
  }

  template<typename TensorT, typename InterpreterT>
	std::tuple<int, std::string, TensorT> PopulationTrainer<TensorT, InterpreterT>::validateModel_(
		Model<TensorT>& model,
		ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
		ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 4>& output,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes)
	{
		std::lock_guard<std::mutex> lock(validateModel_mutex);
		// score the model
		try
		{
			std::vector<TensorT> model_errors = model_trainer.validateModel(
				model, input, output, time_steps,
				input_nodes, model_logger, model_interpreter);
			TensorT model_ave_error = 1e6;
			if (model_errors.size() > 0)
				model_ave_error = std::accumulate(model_errors.begin(), model_errors.end(), 0.0) / model_errors.size();
			if (isnan(model_ave_error))
				model_ave_error = 1e32; // a large number

			char cout_char[512];
			sprintf(cout_char, "Model%s (Nodes: %d, Links: %d) error: %.6f\n",
				model.getName().data(), model.getNodes().size(), model.getLinks().size(), model_ave_error);
			std::cout << cout_char;

			return std::make_tuple(model.getId(), model.getName(), model_ave_error);
		}
		catch (std::exception& e)
		{
			printf("The model %s is broken.\n", model.getName().data());
			printf("Error: %s.\n", e.what());
			return std::make_tuple(model.getId(), model.getName(), TensorT(1e6));
		}
	}

	template<typename TensorT, typename InterpreterT>
	std::vector<std::tuple<int, std::string, TensorT>> PopulationTrainer<TensorT, InterpreterT>::getTopNModels_(
		std::vector<std::tuple<int, std::string, TensorT>> model_validation_scores,
		const int& n_top)
	{
		// sort each model based on their scores in ascending order
		std::sort(
			model_validation_scores.begin(), model_validation_scores.end(),
			[=](std::tuple<int, std::string, TensorT>& a, std::tuple<int, std::string, TensorT>& b)
		{
			return std::get<2>(a) < std::get<2>(b);
		}
		);

		// select the top N from the models
		int n_ = n_top;
		if (n_ > model_validation_scores.size())
			n_ = model_validation_scores.size();

		std::vector<std::tuple<int, std::string, TensorT>> top_n_models;
		for (int i = 0; i < n_; ++i) { top_n_models.push_back(model_validation_scores[i]); }

		return top_n_models;
	}

	template<typename TensorT, typename InterpreterT>
	std::vector<std::tuple<int, std::string, TensorT>> PopulationTrainer<TensorT, InterpreterT>::getRandomNModels_(
		std::vector<std::tuple<int, std::string, TensorT>> model_validation_scores,
		const int& n_random)
	{
		int n_ = n_random;
		if (n_ > model_validation_scores.size())
			n_ = model_validation_scores.size();

		// select a random subset of the top N
		std::random_device seed;
		std::mt19937 engine(seed());
		std::shuffle(model_validation_scores.begin(), model_validation_scores.end(), engine);
		std::vector<std::tuple<int, std::string, TensorT>> random_n_models;
		for (int i = 0; i < n_; ++i) { random_n_models.push_back(model_validation_scores[i]); }

		return random_n_models;
	}

	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::replicateModels(
		std::vector<Model<TensorT>>& models,
		ModelReplicator<TensorT>& model_replicator,
		std::string unique_str,
		int n_threads)
	{
		// replicate and modify
		std::vector<Model<TensorT>> models_copy = models;
		int cnt = 0;
		std::vector<std::future<std::pair<bool, Model<TensorT>>>> task_results;
		int thread_cnt = 0;
		for (Model<TensorT>& model : models_copy)
		{
			for (int i = 0; i < getNReplicatesPerModel(); ++i)
			{
				std::packaged_task<std::pair<bool, Model<TensorT>>// encapsulate in a packaged_task
					(Model<TensorT>&, ModelReplicator<TensorT>&,
						const std::string&, const int&, const bool&, const int&, const bool&, const bool&
						)> task(PopulationTrainer<TensorT, InterpreterT>::replicateModel_);

				// launch the thread
				task_results.push_back(task.get_future());
				std::thread task_thread(std::move(task),
					std::ref(model), std::ref(model_replicator),
					std::ref(unique_str), std::ref(cnt),
          std::ref(remove_isolated_nodes_), std::ref(prune_model_num_), std::ref(check_complete_input_to_output_), std::ref(reset_model_copy_weights_));
				task_thread.detach();

				// retreive the results
				if (thread_cnt == n_threads - 1 || cnt == models_copy.size()*getNReplicatesPerModel() - 1)
				{
					for (auto& task_result : task_results)
					{
						if (task_result.valid())
						{
							try
							{
								std::pair<bool, Model<TensorT>> model_task_result = task_result.get();
								if (model_task_result.first) {
									model_task_result.second.setId(getNextID());
									models.push_back(model_task_result.second);
								}
								else
									std::cout << "All models were broken." << std::endl;
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

	template<typename TensorT, typename InterpreterT>
	std::pair<bool, Model<TensorT>> PopulationTrainer<TensorT, InterpreterT>::replicateModel_(
		Model<TensorT>& model,
		ModelReplicator<TensorT>& model_replicator,
		const std::string& unique_str, const int& cnt,
    const bool& remove_isolated_nodes, const int& prune_model_num, const bool& check_complete_input_to_output, const bool& reset_model_copy_weights)
	{
		std::lock_guard<std::mutex> lock(replicateModel_mutex);

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

		char model_name_char[512];
		sprintf(model_name_char, "%s@replicateModel#%s", model_name_new.data(), unique_str.data());
		std::string model_name = model_replicator.makeUniqueHash(model_name_char, std::to_string(cnt));

    int max_iters = 8; // changed from 32
		for (int iter = 0; iter < max_iters; ++iter)
		{
			Model<TensorT> model_copy(model);
			model_copy.setName(model_name);

			model_replicator.makeRandomModifications();
			model_replicator.modifyModel(model_copy, unique_str, prune_model_num);

			// model checks
      // TODO: add unit test coverage for these cases
      if (remove_isolated_nodes)	model_copy.removeIsolatedNodes();
			if (prune_model_num > 0) model_copy.pruneModel(prune_model_num);
      bool complete_model = true;
      if (check_complete_input_to_output) complete_model = model_copy.checkCompleteInputToOutput();

      if (complete_model) {
        // reset the weights
        if (reset_model_copy_weights) {
          for (auto& weight_map : model.getWeightsMap()) {
            weight_map.second->setInitWeight(true);
          }
        }
        return std::make_pair(true, model_copy);
      }
		}
		return std::make_pair(false, Model<TensorT>());
		//throw std::runtime_error("All modified models were broken!");
	}

	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::trainModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
		ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 4>& output,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes)
	{
		// std::vector<std::string> broken_model_names;

    // Make the list of workers
    std::vector<std::packaged_task<bool
      (std::vector<Model<TensorT>>&,
        ModelTrainer<TensorT, InterpreterT>&, InterpreterT&,
        ModelLogger<TensorT>&,
        const Eigen::Tensor<TensorT, 4>&,
        const Eigen::Tensor<TensorT, 4>&,
        const Eigen::Tensor<TensorT, 3>&,
        const std::vector<std::string>&
        )>> tasks;
    for (size_t i = 0; i < model_interpreters.size(); ++i) {
      std::packaged_task<bool
        (std::vector<Model<TensorT>>&,
          ModelTrainer<TensorT, InterpreterT>&, InterpreterT&,
          ModelLogger<TensorT>&,
          const Eigen::Tensor<TensorT, 4>&,
          const Eigen::Tensor<TensorT, 4>&,
          const Eigen::Tensor<TensorT, 3>&,
          const std::vector<std::string>&
          )> task(PopulationTrainer<TensorT, InterpreterT>::trainModels_);
      tasks.push_back(std::move(task));
    }
    
    // Launch the tasks asyncronously
    train_models_iter_ = 0;
    for (int i = 0; i < model_interpreters.size(); ++i) {

      // create a copy of the model logger
      ModelLogger<TensorT> model_logger_copy = model_logger;

      // launch the interpreter
      std::future<bool> task_result = tasks.at(i).get_future();
      std::thread task_thread(std::move(tasks.at(i)),
        std::ref(models), std::ref(model_trainer), std::ref(model_interpreters[i]), std::ref(model_logger_copy),
        std::ref(input), std::ref(output), std::ref(time_steps),
        std::ref(input_nodes));
      task_thread.detach();

      // retreive the results
      try {
        const bool result = task_result.get();
      }
      catch (std::exception & e) {
        printf("Exception: %s", e.what());
      }
    }

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

  template<typename TensorT, typename InterpreterT>
  inline bool PopulationTrainer<TensorT, InterpreterT>::trainModels_(std::vector<Model<TensorT>>& models, ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter, ModelLogger<TensorT>& model_logger, const Eigen::Tensor<TensorT, 4>& input, const Eigen::Tensor<TensorT, 4>& output, const Eigen::Tensor<TensorT, 3>& time_steps, const std::vector<std::string>& input_nodes)
  {
    bool status = false;
    while (true) {
      // get the next available thread
      const size_t train_models_iter = train_models_iter_.fetch_add(1);
      if (train_models_iter >= models.size()) {
        break;
      }

      // create the task and the future
      std::packaged_task<bool
        (Model<TensorT>&,
          ModelTrainer<TensorT, InterpreterT>&, InterpreterT&,
          ModelLogger<TensorT>&,
          const Eigen::Tensor<TensorT, 4>&,
          const Eigen::Tensor<TensorT, 4>&,
          const Eigen::Tensor<TensorT, 3>&,
          const std::vector<std::string>&
          )> task(PopulationTrainer<TensorT, InterpreterT>::trainModel_);
      std::future<bool> task_result = task.get_future();

      // launch the thread
      std::thread task_thread(std::move(task),
        std::ref(models[train_models_iter]), std::ref(model_trainer), std::ref(model_interpreter), std::ref(model_logger),
        std::ref(input), std::ref(output), std::ref(time_steps),
        std::ref(input_nodes));
      task_thread.detach();

      // retreive the results
      try {
        status = task_result.get();
      }
      catch (std::exception & e) {
        printf("Exception: %s", e.what());
      }
    }

    // Clear the interpreter cache before moving to the next model
    model_interpreter.clear_cache();
    return status;
  }

	template<typename TensorT, typename InterpreterT>
	bool PopulationTrainer<TensorT, InterpreterT>::trainModel_(
		Model<TensorT>& model,
		ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
		ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 4>& output,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes)
	{
		std::lock_guard<std::mutex> lock(trainModel_mutex);

		try
		{
			model_trainer.trainModel(
				model,
				input, output, time_steps,
				input_nodes, model_logger, model_interpreter);
			return true;
		}
		catch (std::exception& e)
		{
			printf("The model %s is broken.\n", model.getName().data());
			printf("Error: %s.\n", e.what());
      ModelInterpreterFile<TensorT, InterpreterT>::storeModelInterpreterCsv(model.getName() + "_interpreterOps.csv", model_interpreter);
			return false;
		}
	}

	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::evalModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
		ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes)
	{
		// std::vector<std::string> broken_model_names;
		std::vector<std::future<bool>> task_results;
		int thread_cnt = 0;

		// train the models
		for (int i = 0; i < models.size(); ++i)
		{
			std::packaged_task<bool // encapsulate in a packaged_task
			(Model<TensorT>&,
				ModelTrainer<TensorT, InterpreterT>&, InterpreterT&,
				ModelLogger<TensorT>&,
				const Eigen::Tensor<TensorT, 4>&,
				const Eigen::Tensor<TensorT, 3>&,
				const std::vector<std::string>&
				)> task(PopulationTrainer<TensorT, InterpreterT>::evalModel_); 
			
			// create a copy of the model trainer and logger
			ModelLogger<TensorT> model_logger_copy = model_logger;

			// launch the thread
			task_results.push_back(task.get_future());
      std::thread task_thread(std::move(task),
        std::ref(models[i]), std::ref(model_trainer), std::ref(model_interpreters[thread_cnt]), std::ref(model_logger_copy),
				std::ref(input), std::ref(time_steps),
				std::ref(input_nodes));
			task_thread.detach();

			// retreive the results
			if (thread_cnt == model_interpreters.size() - 1 || i == models.size() - 1)
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

	template<typename TensorT, typename InterpreterT>
	bool PopulationTrainer<TensorT, InterpreterT>::evalModel_(
		Model<TensorT>& model,
		ModelTrainer<TensorT, InterpreterT>& model_trainer, InterpreterT& model_interpreter,
		ModelLogger<TensorT>& model_logger,
		const Eigen::Tensor<TensorT, 4>& input,
		const Eigen::Tensor<TensorT, 3>& time_steps,
		const std::vector<std::string>& input_nodes)
	{
		std::lock_guard<std::mutex> lock(evalModel_mutex);

		try
		{
			model_trainer.evaluateModel(
				model, input, time_steps,
				input_nodes, model_logger, model_interpreter);
			return true;
		}
		catch (std::exception& e)
		{
			printf("The model %s is broken.\n", model.getName().data());
			printf("Error: %s.\n", e.what());
			return false;
		}
	}

	template<typename TensorT, typename InterpreterT>
	int PopulationTrainer<TensorT, InterpreterT>::getNextID()
	{
		return ++unique_id_;
	}

	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::setID(const int & id)
	{
		unique_id_ = id;
	}

	template<typename TensorT, typename InterpreterT>
	std::vector<std::vector<std::tuple<int, std::string, TensorT>>> PopulationTrainer<TensorT, InterpreterT>::evolveModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
		ModelReplicator<TensorT>& model_replicator,
		DataSimulator<TensorT> &data_simulator,
		ModelLogger<TensorT>& model_logger,
		PopulationLogger<TensorT>& population_logger,
		const std::vector<std::string>& input_nodes)
	{
		std::vector<std::vector<std::tuple<int, std::string, TensorT>>> models_validation_errors_per_generation;

    std::vector<std::string> output_nodes = model_trainer.getLossOutputNodesLinearized();

		// generate the input/output data for validation		
		std::cout << "Generating the input/output data for validation..." << std::endl;
		Eigen::Tensor<TensorT, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsValidation());
		Eigen::Tensor<TensorT, 4> output_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochsValidation());
		Eigen::Tensor<TensorT, 3> time_steps_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsValidation());
		data_simulator.simulateValidationData(input_data_validation, output_data_validation, time_steps_validation);

		// Population initial conditions
		setID(models.size());

		// Initialize the logger
		if (this->getLogTraining())
			population_logger.initLogs("Population");

		// Evolve the population
		for (int iter = 0; iter < getNGenerations(); ++iter)
		{
			char iter_char[128];
			sprintf(iter_char, "Iteration #: %d\n", iter);
			std::cout << iter_char;

			// update the population dynamics
			adaptivePopulationScheduler(iter, models, models_validation_errors_per_generation);
      updateNEpochsTraining(model_trainer);

			// Generate the input and output data for training [BUG FREE]
			std::cout << "Generating the input/output data for training..." << std::endl;
			Eigen::Tensor<TensorT, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochsTraining());
			Eigen::Tensor<TensorT, 4> output_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochsTraining());
			Eigen::Tensor<TensorT, 3> time_steps_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochsTraining());
			data_simulator.simulateTrainingData(input_data_training, output_data_training, time_steps_training);

			// train the population
			std::cout << "Training the models..." << std::endl;
			if (model_trainer.getNEpochsTraining() > 0) {
				trainModels(models, model_trainer, model_interpreters, model_logger,
					input_data_training, output_data_training, time_steps_training, input_nodes);
			}

			// select the top N from the population
			std::cout << "Selecting the models..." << std::endl;
      std::vector<std::tuple<int, std::string, TensorT>> models_validation_errors;
      if (select_models_) { 
        models_validation_errors = selectModels(
          models, model_trainer, model_interpreters, model_logger,
          input_data_validation, output_data_validation, time_steps_validation, input_nodes);
      }
      else { // TODO: add unit test coverage for this case
        for (Model<TensorT>& model : models) {
          models_validation_errors.push_back(std::make_tuple(model.getId(), model.getName(), TensorT(-1)));
        }
      }
			models_validation_errors_per_generation.push_back(models_validation_errors);

      // update the model replication attributes
      model_replicator.adaptiveReplicatorScheduler(iter, models, models_validation_errors_per_generation);

			// log generation
			if (this->getLogTraining()) {
				//if (this->getVerbosityLevel() >= 2)
				//	std::cout << "Logging..." << std::endl;
				this->trainingPopulationLogger(iter, models, population_logger, models_validation_errors);
			}

			if (iter < getNGenerations() - 1)
			{
				// replicate and modify models
				// [TODO: add options for verbosity]
				std::cout << "Replicating and modifying the models..." << std::endl;
				replicateModels(models, model_replicator, std::to_string(iter), model_interpreters.size());
				std::cout << "Population size of " << models.size() << std::endl;
			}
		}
		return models_validation_errors_per_generation;
	}

	template<typename TensorT, typename InterpreterT>
	void PopulationTrainer<TensorT, InterpreterT>::evaluateModels(
		std::vector<Model<TensorT>>& models,
		ModelTrainer<TensorT, InterpreterT>& model_trainer,  std::vector<InterpreterT>& model_interpreters,
		ModelReplicator<TensorT>& model_replicator,
		DataSimulator<TensorT>& data_simulator,
		ModelLogger<TensorT>& model_logger,
		const std::vector<std::string>& input_nodes)
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
		evalModels(models, model_trainer, model_interpreters, model_logger,
			input_data_evaluation, time_steps_evaluation, input_nodes);
	}

  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::adaptivePopulationScheduler(const int & n_generations, std::vector<Model<TensorT>>& models,
    std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_errors_per_generations)
  {
    // TODO user
  }

	template<typename TensorT, typename InterpreterT>
	inline void PopulationTrainer<TensorT, InterpreterT>::trainingPopulationLogger(const int & n_generations,
		std::vector<Model<TensorT>>& models,
		PopulationLogger<TensorT>& population_logger, 
		const std::vector<std::tuple<int, std::string, TensorT>>& models_validation_errors_per_generation)
	{
		// TODO user
	}

  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::setNEpochsTraining(const int & n_epochs)
  {
    n_epochs_training_ = n_epochs;
  }

  template<typename TensorT, typename InterpreterT>
  inline int PopulationTrainer<TensorT, InterpreterT>::getNEpochsTraining() const
  {
    return n_epochs_training_;
  }

  template<typename TensorT, typename InterpreterT>
  inline void PopulationTrainer<TensorT, InterpreterT>::updateNEpochsTraining(ModelTrainer<TensorT, InterpreterT>& model_trainer)
  {
    if (n_epochs_training_ >= 0)
      model_trainer.setNEpochsTraining(n_epochs_training_);
  }

	// TensorT PopulationTrainer<TensorT, InterpreterT>::calculateMean(std::vector<TensorT> values)
	// {
	//   if (values.empty())
	//     return 0;
	//   return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
	// }

	// TensorT PopulationTrainer<TensorT, InterpreterT>::calculateStdDev(std::vector<TensorT> values)
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

#endif //SMARTPEAK_POPULATIONTRAINER_H