/**TODO:  Add copyright*/

#include <SmartPeak/io/PopulationTrainerFile.h>
#include <SmartPeak/io/CSVWriter.h>
#include <SmartPeak/ml/Model.h> 
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/ModelFile.h>

namespace SmartPeak
{

  PopulationTrainerFile::PopulationTrainerFile(){}
  PopulationTrainerFile::~PopulationTrainerFile(){}

	void PopulationTrainerFile::sanitizeModelName(std::string& model_name)
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

	bool PopulationTrainerFile::storeModels(const std::vector<Model>& models,
		const std::string& filename)
	{
		for (const Model& model : models)
		{
			// write the model to file
			std::string model_name = model.getName();
			sanitizeModelName(model_name);
			std::string model_name_score = model_name + "_";

			WeightFile weightfile;
			weightfile.storeWeightsCsv(model_name_score + filename + "_Weights.csv", model.getWeights());
			LinkFile linkfile;
			linkfile.storeLinksCsv(model_name_score + filename + "_Links.csv", model.getLinks());
			NodeFile nodefile;
			nodefile.storeNodesCsv(model_name_score + filename + "_Nodes.csv", model.getNodes());
			ModelFile modelfile;
			modelfile.storeModelDot(model_name_score + filename + "_Graph.gv", model);
		}

		return true;
	}

	bool PopulationTrainerFile::storeModelValidations(
		const std::string& filename,
		const std::vector<std::pair<std::string, float>>& models_validation_errors)
	{
		CSVWriter csvwriter(filename);

		// write the headers to the first line
		const std::vector<std::string> headers = { "model_name", "ave_validation_error" };
		csvwriter.writeDataInRow(headers.begin(), headers.end());

		for (const std::pair<std::string, float>& model_validation_error : models_validation_errors)
		{
			std::vector<std::string> row;
			row.push_back(model_validation_error.first);
			char error[512];
			sprintf(error, "%0.6f", model_validation_error.second);
			std::string error_str(error);
			row.push_back(error_str);

			// write to file
			csvwriter.writeDataInRow(row.begin(), row.end());
		}
		return true;
	}
}