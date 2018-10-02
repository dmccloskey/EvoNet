/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINERFILE_H
#define SMARTPEAK_POPULATIONTRAINERFILE_H

#include <SmartPeak/ml/PopulationTrainer.h>

#include <iostream>
#include <fstream>
#include <vector>

namespace SmartPeak
{

  /**
    @brief PopulationTrainerFile
  */
  class PopulationTrainerFile
  {
public:
    PopulationTrainerFile(); ///< Default constructor
    ~PopulationTrainerFile(); ///< Default destructor

		/**
		@brief remove characters that cannot be included in a filename

		@param[in, out] model_name The name of the model
		*/
		static void sanitizeModelName(std::string& model_name);

		/**
		@brief write all models to file

		Files written include:
		- links
		- nodes
		- weights
		- graph representation of the model in DOT format for vis using GraphVIZ

		@param[in] models The vector (i.e., population) of models
		@param[in] filename The base string to use when writing out the data files

		@returns True if successful, false otherwise
		*/
		bool storeModels(const std::vector<Model>& models,
			const std::string& filename);

		/**
		@brief write Model average validation error to file

		@param[in] models_validation_errors Vector of model_name/average error pairs

		@returns True if successful, false otherwise
		*/
		bool storeModelValidations(
			const std::string& filename,
			const std::vector<std::pair<int, float>>& models_validation_errors);
	};
}

#endif //SMARTPEAK_POPULATIONTRAINERFILE_H