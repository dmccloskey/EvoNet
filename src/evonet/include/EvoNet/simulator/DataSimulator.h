/**TODO:  Add copyright*/

#ifndef EVONET_DATASIMULATOR_H
#define EVONET_DATASIMULATOR_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace EvoNet
{
  /**
    @brief Base class to implement a data generator or simulator
  */
	template<typename TensorT>
	class DataSimulator
	{
	public:
		DataSimulator() = default; ///< Default constructor
		~DataSimulator() = default; ///< Default destructor

		/**
			@brief Entry point to define the simulated data for training

      Overload creates the input and output data for the entire epoch

			@param[in, out] input Input Tensor for the model
			@param[in, out] output Output Tensor for the model
			@param[in, out] time_steps Time step tensor for the model
		*/
    virtual void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {};

    /**
      @brief Entry point to define the simulated data for training

      Overload creates the input and output data for a single epoch

      @param[in, out] input Input Tensor for the model
      @param[in, out] output Output Tensor for the model
      @param[in, out] time_steps Time step tensor for the model
    */
    virtual void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps) {};

    /**
      @brief Entry point to define the simulated data for training

      Overload creates the input and output data for a single epoch

      @param[in, out] input_data Input Tensor for the model
      @param[in, out] loss_output_data Output Tensor for the model used to compute the loss function
      @param[in, out] metric_output_data Output Tensor for the model used to compute the model metrics
      @param[in, out] time_steps Time step tensor for the model
    */
    virtual void simulateTrainingData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {};

		/**
		@brief Entry point to define the simulated data for testing/validation

      Overload creates the input and output data for the entire epoch

		@param[in, out] input Input Tensor for the model
		@param[in, out] output Output Tensor for the model
		@param[in, out] time_steps Time step tensor for the model
		*/
    virtual void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps) {};

    /**
    @brief Entry point to define the simulated data for testing/validation

      Overload creates the input and output data for a single epoch

    @param[in, out] input Input Tensor for the model
    @param[in, out] output Output Tensor for the model
    @param[in, out] time_steps Time step tensor for the model
    */
    virtual void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& output_data, Eigen::Tensor<TensorT, 2>& time_steps) {};
    
    /**
      @brief Entry point to define the simulated data for validation

      Overload creates the input and output data for a single epoch

      @param[in, out] input_data Input Tensor for the model
      @param[in, out] loss_output_data Output Tensor for the model used to compute the loss function
      @param[in, out] metric_output_data Output Tensor for the model used to compute the model metrics
      @param[in, out] time_steps Time step tensor for the model
    */
    virtual void simulateValidationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& loss_output_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {};

		/**
		@brief Entry point to define the simulation data for evaluation

      Overload creates the input and output data for the entire epoch

		@param[in, out] input Input Tensor for the model
		@param[in, out] output Output Tensor for the model
		@param[in, out] time_steps Time step tensor for the model
		*/
    virtual void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};

    /**
    @brief Entry point to define the simulation data for evaluation

      Overload creates the input and output data for the entire epoch

    @param[in, out] input Input Tensor for the model
    @param[in, out] output Output Tensor for the model
    @param[in, out] time_steps Time step tensor for the model
    */
    virtual void simulateEvaluationData(Eigen::Tensor<TensorT, 3>& input_data, Eigen::Tensor<TensorT, 3>& metric_output_data, Eigen::Tensor<TensorT, 2>& time_steps) {};
	};
}

#endif //EVONET_DATASIMULATOR_H