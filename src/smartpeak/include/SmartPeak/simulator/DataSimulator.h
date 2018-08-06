/**TODO:  Add copyright*/

#ifndef SMARTPEAK_DATASIMULATOR_H
#define SMARTPEAK_DATASIMULATOR_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
  /**
    @brief Base class to implement a data generator or simulator
  */
	class DataSimulator
	{
	public:
		DataSimulator() = default; ///< Default constructor
		~DataSimulator() = default; ///< Default destructor

		/**
			@brief Entry point to define the simulated data

			@param[in, out] input Input Tensor for the model
			@param[in, out] output Output Tensor for the model
			@param[in, out] time_steps Time step tensor for the model
		*/
		void simulateData(Eigen::Tensor<float, 4>& input_data, Eigen::Tensor<float, 4>& output_data, Eigen::Tensor<float, 3>& time_steps);
	};
}

#endif //SMARTPEAK_DATASIMULATOR_H