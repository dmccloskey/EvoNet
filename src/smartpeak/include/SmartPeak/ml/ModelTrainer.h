/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINER_H
#define SMARTPEAK_MODELTRAINER_H

// .h
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/ModelLogger.h>
#include <vector>
#include <string>

// .cpp

namespace SmartPeak
{

  /**
    @brief Class to train a network model
  */
	template<typename TensorT, typename InterpreterT>
  class ModelTrainer
  {
public:
    ModelTrainer() = default; ///< Default constructor
    ~ModelTrainer() = default; ///< Default destructor

    void setBatchSize(const int& batch_size); ///< batch_size setter
    void setMemorySize(const int& memory_size); ///< memory_size setter
    void setNEpochsTraining(const int& n_epochs); ///< n_epochs setter
		void setNEpochsValidation(const int& n_epochs); ///< n_epochs setter
		void setNEpochsEvaluation(const int& n_epochs); ///< n_epochs setter
		void setVerbosityLevel(const int& verbosity_level); ///< verbosity_level setter
		void setLogging(bool log_training = false, bool log_validation = false, bool log_evaluation = false); ///< enable_logging setter
		void setLossFunctions(const std::vector<std::shared_ptr<LossFunctionOp<TensorT>>>& loss_functions); ///< loss_functions setter [TODO: tests]
		void setLossFunctionGrads(const std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>>& loss_function_grads); ///< loss_functions setter [TODO: tests]
		void setOutputNodes(const std::vector<std::vector<std::string>>& output_nodes); ///< output_nodes setter [TODO: tests]
		void setNTBPTTSteps(const int& n_TBPTT); ///< n_TBPTT setter
		void setNTETTSteps(const int& n_TETT); ///< n_TETT setter
		void setFindCycles(const bool& find_cycles); ///< find_cycles setter [TODO: tests]

    int getBatchSize() const; ///< batch_size setter
    int getMemorySize() const; ///< memory_size setter
    int getNEpochsTraining() const; ///< n_epochs setter
		int getNEpochsValidation() const; ///< n_epochs setter
		int getNEpochsEvaluation() const; ///< n_epochs setter
		int getVerbosityLevel() const; ///< verbosity_level setter
		bool getLogTraining() const;
		bool getLogValidation() const;
		bool getLogEvaluation() const;
		std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> getLossFunctions(); ///< loss_functions getter [TODO: tests]
		std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> getLossFunctionGrads(); ///< loss_functions getter [TODO: tests]
		std::vector<std::vector<std::string>> getOutputNodes(); ///< output_nodes getter [TODO: tests]
		int getNTBPTTSteps() const; ///< n_TBPTT setter
		int getNTETTSteps() const; ///< n_TETT setter
		bool getFindCycles(); ///< find_cycles getter [TODO: tests]
 
    /**
      @brief Check input dimensions.

      @param n_epochs The number of training epochs
      @param input The input data
      @param batch_size The batch size of the nodes
      @param memory_size The memory size of the nodes
      @param input_nodes The node names

      @returns True on success, False if not
    */ 
    bool checkInputData(const int& n_epochs,
      const Eigen::Tensor<TensorT, 4>& input,
      const int& batch_size,
      const int& memory_size,
      const std::vector<std::string>& input_nodes);
 
    /**
      @brief Check output dimensions.

      @param n_epochs The number of training epochs
      @param output The output data
      @param batch_size The batch size of the nodes
      @param output_nodes The node names

      @returns True on success, False if not
    */ 
    bool checkOutputData(const int& n_epochs,
      const Eigen::Tensor<TensorT, 4>& output,
      const int& batch_size,
			const int& memory_size,
      const std::vector<std::string>& output_nodes);
 
    /**
      @brief Check time step dimensions required for FPTT.

      @param n_epochs The number of training epochs
      @param time_steps The time step spacing
      @param batch_size The batch size of the nodes
      @param memory_size The memory size of the nodes

      @returns True on success, False if not
    */ 
    bool checkTimeSteps(const int& n_epochs,
      const Eigen::Tensor<TensorT, 3>& time_steps,
      const int& batch_size,
      const int& memory_size);
 
    /**
      @brief Entry point for users to code their script
        for model training

      @param[in, out] model The model to train
      @param[in] model_resources The hardware available for training the model
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names

      @returns vector of average model error scores
    */ 
		virtual std::vector<TensorT> trainModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 4>& output,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			InterpreterT& model_interpreter) = 0;
 
    /**
      @brief Entry point for users to code their script
        for model validation

      @param[in, out] model The model to train
      @param[in] model_resources The hardware available for training the model
      @param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
      @param[in] output Expected output data tensor of dimensions: batch_size, memory_size, output_nodes, n_epochs
      @param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
      @param[in] input_nodes Input node names

      @returns vector of average model error scores
    */ 
		virtual std::vector<TensorT> validateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 4>& output,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			InterpreterT& model_interpreter) = 0;

		/**
			@brief Entry point for users to code their script
				for model forward evaluations

			@param[in, out] model The model to train
      @param[in] model_resources The hardware available for training the model
			@param[in] input Input data tensor of dimensions: batch_size, memory_size, input_nodes, n_epochs
			@param[in] time_steps Time steps of the forward passes of dimensions: batch_size, memory_size, n_epochs
			@param[in] input_nodes Input node names

			@returns vector of vectors corresponding to output nodes
		*/
		virtual std::vector<std::vector<Eigen::Tensor<TensorT, 2>>> evaluateModel(Model<TensorT>& model,
			const Eigen::Tensor<TensorT, 4>& input,
			const Eigen::Tensor<TensorT, 3>& time_steps,
			const std::vector<std::string>& input_nodes,
			ModelLogger<TensorT>& model_logger,
			InterpreterT& model_interpreter) = 0;
 
    /**
      @brief Entry point for users to code their script
        to build the model

      @returns The constructed model
    */ 
    virtual Model<TensorT> makeModel() = 0;

		/**
		@brief Entry point for users to code their adaptive scheduler
		to modify training parameters based on a given trigger

		@param[in] n_generations The number of evolution generations
		@param[in] n_epochs The number of training/validation epochs
		@param[in, out] model The model
		@param[in, out] model_interpreter The model interpreter
		@param[in] model_errors The trace of model errors from training/validation

		*/
		virtual void adaptiveTrainerScheduler(
			const int& n_generations,
			const int& n_epochs,
			Model<TensorT>& model,
			InterpreterT& model_interpreter,
			const std::vector<TensorT>& model_errors) = 0;

protected:
		std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> loss_functions_;
		std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> loss_function_grads_;
		std::vector<std::vector<std::string>> output_nodes_;

private:
    int batch_size_;
    int memory_size_;
    int n_epochs_training_ = 0;
		int n_epochs_validation_ = 0;
		int n_epochs_evaluation_ = 0;

		int n_TBPTT_steps_ = -1; ///< the number of truncated back propogation through time steps
		int n_TETT_steps_ = -1; ///< the number of truncated error through time calculation steps

		int verbosity_level_ = 0; ///< level of verbosity (0=none, 1=test/validation errors, 2=test/validation node values
		bool log_training_ = false; ///< whether to log training epochs or not
		bool log_validation_ = false; ///< whether to log validation epochs or not
		bool log_evaluation_ = false; ///< whether to log evaluation epochs or not

		bool find_cycles_ = true; ///< whether to find cycles prior to interpreting the model
		bool fast_interpreter_ = false; ///< whether to skip certain checks when interpreting the model
  };
	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setBatchSize(const int& batch_size)
	{
		batch_size_ = batch_size;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setMemorySize(const int& memory_size)
	{
		memory_size_ = memory_size;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setNEpochsTraining(const int& n_epochs)
	{
		n_epochs_training_ = n_epochs;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setNEpochsValidation(const int & n_epochs)
	{
		n_epochs_validation_ = n_epochs;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setNEpochsEvaluation(const int & n_epochs)
	{
		n_epochs_evaluation_ = n_epochs;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setVerbosityLevel(const int & verbosity_level)
	{
		verbosity_level_ = verbosity_level;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setLogging(bool log_training, bool log_validation, bool log_evaluation)
	{
		log_training_ = log_training;
		log_validation_ = log_validation;
		log_evaluation_ = log_evaluation;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setLossFunctions(const std::vector<std::shared_ptr<LossFunctionOp<TensorT>>>& loss_functions)
	{
		loss_functions_ = loss_functions;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setLossFunctionGrads(const std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>>& loss_function_grads)
	{
		loss_function_grads_ = loss_function_grads;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setOutputNodes(const std::vector<std::vector<std::string>>& output_nodes)
	{
		output_nodes_ = output_nodes;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setNTBPTTSteps(const int & n_TBPTT)
	{
		n_TBPTT_steps_ = n_TBPTT;
	}

	template<typename TensorT, typename InterpreterT>
	void ModelTrainer<TensorT, InterpreterT>::setNTETTSteps(const int & n_TETT)
	{
		n_TETT_steps_ = n_TETT;
	}

	template<typename TensorT, typename InterpreterT>
	inline void ModelTrainer<TensorT, InterpreterT>::setFindCycles(const bool & find_cycles)
	{
		find_cycles_ = find_cycles;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getBatchSize() const
	{
		return batch_size_;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getMemorySize() const
	{
		return memory_size_;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getNEpochsTraining() const
	{
		return n_epochs_training_;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getNEpochsValidation() const
	{
		return n_epochs_validation_;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getNEpochsEvaluation() const
	{
		return n_epochs_evaluation_;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getVerbosityLevel() const
	{
		return verbosity_level_;
	}

	template<typename TensorT, typename InterpreterT>
	inline bool ModelTrainer<TensorT, InterpreterT>::getLogTraining() const
	{
		return log_training_;
	}

	template<typename TensorT, typename InterpreterT>
	inline bool ModelTrainer<TensorT, InterpreterT>::getLogValidation() const
	{
		return log_validation_;
	}

	template<typename TensorT, typename InterpreterT>
	inline bool ModelTrainer<TensorT, InterpreterT>::getLogEvaluation() const
	{
		return log_evaluation_;
	}

	template<typename TensorT, typename InterpreterT>
	std::vector<std::shared_ptr<LossFunctionOp<TensorT>>> ModelTrainer<TensorT, InterpreterT>::getLossFunctions()
	{
		return loss_functions_;
	}

	template<typename TensorT, typename InterpreterT>
	std::vector<std::shared_ptr<LossFunctionGradOp<TensorT>>> ModelTrainer<TensorT, InterpreterT>::getLossFunctionGrads()
	{
		return loss_function_grads_;
	}

	template<typename TensorT, typename InterpreterT>
	std::vector<std::vector<std::string>> ModelTrainer<TensorT, InterpreterT>::getOutputNodes()
	{
		return output_nodes_;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getNTBPTTSteps() const
	{
		return n_TBPTT_steps_;
	}

	template<typename TensorT, typename InterpreterT>
	int ModelTrainer<TensorT, InterpreterT>::getNTETTSteps() const
	{
		return n_TETT_steps_;
	}

	template<typename TensorT, typename InterpreterT>
	inline bool ModelTrainer<TensorT, InterpreterT>::getFindCycles()
	{
		return find_cycles_;
	}

	template<typename TensorT, typename InterpreterT>
	bool ModelTrainer<TensorT, InterpreterT>::checkInputData(const int& n_epochs,
		const Eigen::Tensor<TensorT, 4>& input,
		const int& batch_size,
		const int& memory_size,
		const std::vector<std::string>& input_nodes)
	{
		if (input.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the input dim 0 of %d\n", batch_size, (int)input.dimension(0));
			return false;
		}
		else if (input.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the input dim 1 of %d\n", memory_size, (int)input.dimension(1));
			return false;
		}
		else if (input.dimension(2) != input_nodes.size())
		{
			printf("input_nodes size of %d is not compatible with the input dim 2 of %d\n", input_nodes.size(), (int)input.dimension(2));
			return false;
		}
		else if (input.dimension(3) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the input dim 3 of %d\n", n_epochs, (int)input.dimension(3));
			return false;
		}
		else
		{
			return true;
		}
	}

	template<typename TensorT, typename InterpreterT>
	bool ModelTrainer<TensorT, InterpreterT>::checkOutputData(const int& n_epochs,
		const Eigen::Tensor<TensorT, 4>& output,
		const int& batch_size,
		const int& memory_size,
		const std::vector<std::string>& output_nodes)
	{
		if (output.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the output dim 0 of %d\n", batch_size, (int)output.dimension(0));
			return false;
		}
		else if (output.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the output dim 1 of %d\n", memory_size, (int)output.dimension(1));
			return false;
		}
		else if (output.dimension(2) != output_nodes.size())
		{
			printf("output_nodes size of %d is not compatible with the output dim 2 of %d\n", output_nodes.size(), (int)output.dimension(2));
			return false;
		}
		else if (output.dimension(3) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the output dim 3 of %d\n", n_epochs, (int)output.dimension(3));
			return false;
		}
		else
		{
			return true;
		}
	}

	template<typename TensorT, typename InterpreterT>
	bool ModelTrainer<TensorT, InterpreterT>::checkTimeSteps(const int & n_epochs, const Eigen::Tensor<TensorT, 3>& time_steps, const int & batch_size, const int & memory_size)
	{
		if (time_steps.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the time_steps dim 0 of %d\n", batch_size, (int)time_steps.dimension(0));
			return false;
		}
		else if (time_steps.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the time_steps dim 1 of %d\n", memory_size, (int)time_steps.dimension(1));
			return false;
		}
		else if (time_steps.dimension(2) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the time_steps dim 3 of %d\n", n_epochs, (int)time_steps.dimension(2));
			return false;
		}
		else
		{
			return true;
		}
	}
}

#endif //SMARTPEAK_MODELTRAINER_H