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
	template<typename HDelT, typename DDelT, typename TensorT>
	void PopulationTrainerFile<HDelT, DDelT, TensorT>::sanitizeModelName(std::string& model_name)
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

	template<typename HDelT, typename DDelT, typename TensorT>
	bool PopulationTrainerFile<HDelT, DDelT, TensorT>::storeModels(const std::vector<Model<HDelT, DDelT, TensorT>>& models,
		const std::string& filename)
	{
		std::fstream file;
		// Open the file in truncate mode

		file.open(filename + ".sh", std::ios::out | std::ios::trunc);

		for (const Model<HDelT, DDelT, TensorT>& model : models)
		{
			// write the model to file
			//std::string model_name = model.getName();
			//sanitizeModelName(model_name);
			//std::string model_name_score = model_name + "_";
			int model_id = model.getId();
			std::string model_name_score = std::to_string(model_id) + "_";

			WeightFile weightfile;
			weightfile.storeWeightsCsv(model_name_score + filename + "_Weights.csv", model.getWeights());
			LinkFile linkfile;
			linkfile.storeLinksCsv(model_name_score + filename + "_Links.csv", model.getLinks());
			NodeFile nodefile;
			nodefile.storeNodesCsv(model_name_score + filename + "_Nodes.csv", model.getNodes());
			ModelFile modelfile;
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

	template<typename HDelT, typename DDelT, typename TensorT>
	bool PopulationTrainerFile<HDelT, DDelT, TensorT>::storeModelValidations(
		const std::string& filename,
		const std::vector<std::pair<int, TensorT>>& models_validation_errors)
	{
		CSVWriter csvwriter(filename);

		// write the headers to the first line
		const std::vector<std::string> headers = { "model_name", "ave_validation_error" };
		csvwriter.writeDataInRow(headers.begin(), headers.end());

		for (const std::pair<int, TensorT>& model_validation_error : models_validation_errors)
		{
			std::vector<std::string> row;
			row.push_back(std::to_string(model_validation_error.first));
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