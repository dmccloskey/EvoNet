/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINERFILE_H
#define SMARTPEAK_POPULATIONTRAINERFILE_H

// .h
#include <SmartPeak/ml/PopulationTrainer.h>
#include <iostream>
#include <fstream>
#include <vector>

// .cpp
#include <SmartPeak/io/CSVWriter.h>
#include <SmartPeak/ml/Model.h> 
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/ModelFile.h>

namespace SmartPeak
{

  /**
    @brief PopulationTrainerFile
  */
	template<typename TensorT>
	class PopulationTrainerFile
	{
	public:
		PopulationTrainerFile() = default; ///< Default constructor
		~PopulationTrainerFile() = default; ///< Default destructor

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
		bool storeModels(std::vector<Model<TensorT>>& models,
			const std::string& filename);

		/**
		@brief write Model average validation error to file

		@param[in] models_validation_errors Vector of model_name/average error pairs

		@returns True if successful, false otherwise
		*/
		bool storeModelValidations(
			const std::string& filename,
			const std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_validation_errors);
	};
	template<typename TensorT>
	void PopulationTrainerFile<TensorT>::sanitizeModelName(std::string& model_name)
	{
		// sanitize the model name
		std::string illegalChars = "\\/:?\"<>|";
		for (std::string::iterator it = model_name.begin(); it < model_name.end(); ++it) {
			bool found = illegalChars.find(*it) != std::string::npos;
			if (found) {
				*it = ' ';
			}
		}
	}

	template<typename TensorT>
	bool PopulationTrainerFile<TensorT>::storeModels(std::vector<Model<TensorT>>& models,
		const std::string& filename)
	{
		std::fstream file;
		// Open the file in truncate mode

		file.open(filename + ".sh", std::ios::out | std::ios::trunc);

		for (Model<TensorT>& model : models)
		{
			// write the model to file
			//std::string model_name = model.getName();
			//sanitizeModelName(model_name);
			//std::string model_name_score = model_name + "_";
			int model_id = model.getId();
			std::string model_name_score = std::to_string(model_id) + "_";

			WeightFile<TensorT> weightfile;
			weightfile.storeWeightsCsv(model_name_score + filename + "_Weights.csv", model.weights_);
			LinkFile linkfile;
			linkfile.storeLinksCsv(model_name_score + filename + "_Links.csv", model.links_);
			NodeFile<TensorT> nodefile;
			nodefile.storeNodesCsv(model_name_score + filename + "_Nodes.csv", model.nodes_);
			ModelFile<TensorT> modelfile;
			std::string dot_filename = model_name_score + filename + "_Graph.gv";
			modelfile.storeModelDot(dot_filename, model);

			char sh_cmd_char[512];
			sprintf(sh_cmd_char, "dot -Tpng -o %s.png %s\n", dot_filename.data(), dot_filename.data());
			std::string sh_cmd(sh_cmd_char);
			file << sh_cmd;
		}
		file.close();

		return true;
	}

	template<typename TensorT>
	bool PopulationTrainerFile<TensorT>::storeModelValidations(
		const std::string& filename,
		const std::vector<std::vector<std::tuple<int, std::string, TensorT>>>& models_validation_errors)
	{
		CSVWriter csvwriter(filename);

		// write the headers to the first line
		const std::vector<std::string> headers = { "model_id", "model_name", "ave_validation_error", "generation" };
		csvwriter.writeDataInRow(headers.begin(), headers.end());

		int generation_iter = 0;
		for (const std::vector<std::tuple<int, std::string, TensorT>>& generation_errors : models_validation_errors) {
			for (const std::tuple<int, std::string, TensorT>& model_validation_error : generation_errors) {
				std::vector<std::string> row;
				row.push_back(std::to_string(std::get<0>(model_validation_error)));
				row.push_back(std::get<1>(model_validation_error));
				char error[512];
				sprintf(error, "%0.6f", std::get<2>(model_validation_error));
				std::string error_str(error);
				row.push_back(error_str);
				row.push_back(std::to_string(generation_iter));

				// write to file
				csvwriter.writeDataInRow(row.begin(), row.end());
			}
			++generation_iter;
		}
		return true;
	}
}

#endif //SMARTPEAK_POPULATIONTRAINERFILE_H