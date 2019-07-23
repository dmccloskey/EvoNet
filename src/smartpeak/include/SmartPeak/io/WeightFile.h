/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHTFILE_H
#define SMARTPEAK_WEIGHTFILE_H

// .h
#include <SmartPeak/ml/Weight.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// .cpp
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>
#include <regex>
#include <SmartPeak/core/StringParsing.h>
#include <cereal/archives/binary.hpp>

namespace SmartPeak
{
  /**
    @brief WeightFile
  */
	template<typename TensorT>
  class WeightFile
  {
public:
    WeightFile() = default; ///< Default constructor
    ~WeightFile() = default; ///< Default destructor
 
    /**
      @brief Load weights from binary file

      @param filename The name of the weights file
      @param weights The weights to load data into

      @returns Status True on success, False if not
    */ 
    bool loadWeightsBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights);
 
    /**
      @brief Load weights from csv file

      @param filename The name of the weights file
      @param weights The weights to load data into

      @returns Status True on success, False if not
    */ 
    bool loadWeightsCsv(const std::string& filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights);
		bool loadWeightValuesCsv(const std::string& filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights);
 
    /**
      @brief Stores weights from binary file

      @param filename The name of the weights file
      @param weights The weights to sore

      @returns Status True on success, False if not
    */ 
    bool storeWeightsBinary(const std::string& filename, const std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights);
 
    /**
      @brief Stores weights from binary file

      @param filename The name of the weights file
      @param weights The weights to sore

      @returns Status True on success, False if not
    */ 
    bool storeWeightsCsv(const std::string& filename, const std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights);
		bool storeWeightValuesCsv(const std::string& filename, const std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights);

    std::map<std::string, TensorT> parseParameters(const std::string& parameters);
  };
	template<typename TensorT>
	bool WeightFile<TensorT>::loadWeightsBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights) {
		std::ifstream ifs(filename, std::ios::binary);
		if (ifs.is_open()) {
			cereal::BinaryInputArchive iarchive(ifs);
			iarchive(weights);
			ifs.close();
		}
		return true; 
	}

	template<typename TensorT>
	bool WeightFile<TensorT>::loadWeightsCsv(const std::string& filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights)
	{
		weights.clear();

		io::CSVReader<9> weights_in(filename);
		weights_in.read_header(io::ignore_extra_column,
			"weight_name", "weight_init_op", "weight_init_params", "solver_op", "solver_params", "weight_value", "module_name", "layer_name", "tensor_index");
		std::string weight_name, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str, weight_value_str, module_name_str, layer_name_str, tensor_index_str;

		while (weights_in.read_row(weight_name, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str, weight_value_str, module_name_str, layer_name_str, tensor_index_str))
		{
			// parse the weight_init_params
			std::map<std::string, TensorT> weight_init_params = parseParameters(weight_init_params_str);

			// parse the weight_init_op
			std::shared_ptr<WeightInitOp<TensorT>> weight_init;
			if (weight_init_op_str == "ConstWeightInitOp")
			{
				ConstWeightInitOp<TensorT>* ptr = nullptr;
				if (weight_init_params.count("n"))
					ptr = new ConstWeightInitOp<TensorT>(weight_init_params.at("n"));
				else
					ptr = new ConstWeightInitOp<TensorT>(1.0);
				weight_init.reset(ptr);
			}
			else if (weight_init_op_str == "RandWeightInitOp")
			{
				RandWeightInitOp<TensorT>* ptr = nullptr;
				if (weight_init_params.count("n"))
					ptr = new RandWeightInitOp<TensorT>(weight_init_params.at("n"));
				else
					ptr = new RandWeightInitOp<TensorT>(1.0);
				weight_init.reset(ptr);
			}
      else if (weight_init_op_str == "RangeWeightInitOp")
      {
        RangeWeightInitOp<TensorT>* ptr = nullptr;
        if (weight_init_params.count("lb") && weight_init_params.count("ub"))
          ptr = new RangeWeightInitOp<TensorT>(weight_init_params.at("lb"), weight_init_params.at("ub"));
        else
          ptr = new RangeWeightInitOp<TensorT>(0.0, 1.0);
        weight_init.reset(ptr);
      }
			else std::cout << "WeightInitOp " << weight_init_op_str << " for weight_name " << weight_name << " was not recognized." << std::endl;

			// parse the solver_params_str
			std::map<std::string, TensorT> solver_params;
			if (!solver_params_str.empty())
				solver_params = parseParameters(solver_params_str);

			// parse the solver_op
			std::shared_ptr<SolverOp<TensorT>> solver;
			if (solver_op_str == "SGDOp")
			{
				SGDOp<TensorT>* ptr = new SGDOp<TensorT>();
				ptr->setLearningRate(0.01);
				if (solver_params.count("learning_rate"))
					ptr->setLearningRate(solver_params.at("learning_rate"));
				ptr->setMomentum(0.9);
				if (solver_params.count("momentum"))
					ptr->setMomentum(solver_params.at("momentum"));
				ptr->setGradientThreshold(1e6);
				if (solver_params.count("gradient_threshold"))
					ptr->setGradientThreshold(solver_params.at("gradient_threshold"));
				ptr->setGradientNoiseSigma(0.0);
				if (solver_params.count("gradient_noise_sigma"))
					ptr->setGradientNoiseSigma(solver_params.at("gradient_noise_sigma"));
				ptr->setGradientNoiseGamma(0.0);
				if (solver_params.count("gradient_noise_gamma"))
					ptr->setGradientNoiseGamma(solver_params.at("gradient_noise_gamma"));
				solver.reset(ptr);
			}
			else if (solver_op_str == "AdamOp")
			{
				AdamOp<TensorT>* ptr = new AdamOp<TensorT>();
				if (solver_params.count("learning_rate"))
					ptr->setLearningRate(solver_params.at("learning_rate"));
				ptr->setMomentum(0.9);
				if (solver_params.count("momentum"))
					ptr->setMomentum(solver_params.at("momentum"));
				ptr->setMomentum2(0.999);
				if (solver_params.count("momentum2"))
					ptr->setMomentum2(solver_params.at("momentum2"));
				ptr->setDelta(1e-8);
				if (solver_params.count("delta"))
					ptr->setDelta(solver_params.at("delta"));
				ptr->setGradientThreshold(1e6);
				if (solver_params.count("gradient_threshold"))
					ptr->setGradientThreshold(solver_params.at("gradient_threshold"));
				ptr->setGradientNoiseSigma(0.0);
				if (solver_params.count("gradient_noise_sigma"))
					ptr->setGradientNoiseSigma(solver_params.at("gradient_noise_sigma"));
				ptr->setGradientNoiseGamma(0.0);
				if (solver_params.count("gradient_noise_gamma"))
					ptr->setGradientNoiseGamma(solver_params.at("gradient_noise_gamma"));
				solver.reset(ptr);
			}
			else if (solver_op_str == "DummySolverOp")
			{
				DummySolverOp<TensorT>* ptr = new DummySolverOp<TensorT>();
				solver.reset(ptr);
			}
			else std::cout << "SolverOp " << solver_op_str << " for weight_name " << weight_name << " was not recognized." << std::endl;

			std::shared_ptr<Weight<TensorT>> weight(new Weight<TensorT>(weight_name, weight_init, solver));

			// parse the weight value
			TensorT weight_value = 0;
			try
			{
				weight_value = std::stof(weight_value_str);
			}
			catch (std::exception& e)
			{
				printf("Exception: %s", e.what());
			}
			weight->setWeight(weight_value);
			weight->setInitWeight(false);
			
			// parse the tensor indexing
			weight->setModuleName(module_name_str);
			weight->setLayerName(layer_name_str);
			std::vector<std::string> tensor_indices = SplitString(tensor_index_str, "|");
			for (std::string& tensor_index : tensor_indices) {
				if (!tensor_index.empty()) {
					std::vector<std::string> tensor_indexes = SplitString(ReplaceTokens(tensor_index, { "[\{\}]", "\\s+" }, ""), ";");
					assert(tensor_indexes.size() == 3);
					int ind1 = -1, ind2 = -1, ind3 = -1;
					ind1 = std::stoi(tensor_indexes[0]);
					ind2 = std::stoi(tensor_indexes[1]);
					ind3 = std::stoi(tensor_indexes[2]);
					weight->addTensorIndex(std::make_tuple(ind1, ind2, ind3));
				}
			}

			weights.emplace(weight_name, weight);
		}
		return true;
	}

	template<typename TensorT>
	inline bool WeightFile<TensorT>::loadWeightValuesCsv(const std::string & filename, std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights)
	{

		io::CSVReader<2> weights_in(filename);
		weights_in.read_header(io::ignore_extra_column,
			"weight_name", "weight_value");
		std::string weight_name, weight_value_str = "";

		while (weights_in.read_row(weight_name, weight_value_str))
		{
			// parse the weight value
			TensorT weight_value = 0;
			try
			{
				weight_value = std::stof(weight_value_str);
			}
			catch (std::exception& e)
			{
				printf("Exception: %s", e.what());
			}

      if (weights.count(weight_name)) {
        weights.at(weight_name)->setWeight(weight_value);
        weights.at(weight_name)->setInitWeight(false);
      }
      else {
        //std::cout << "Weight " << weight_name << " was not found in the model and will be skipped." << std::endl;
      }
		}
		return true;
	}

	template<typename TensorT>
	std::map<std::string, TensorT> WeightFile<TensorT>::parseParameters(const std::string& parameters)
	{
		// parse the parameters
		std::regex re(";");
		std::vector<std::string> str_tokens;
		std::copy(
			std::sregex_token_iterator(parameters.begin(), parameters.end(), re, -1),
			std::sregex_token_iterator(),
			std::back_inserter(str_tokens));

		// break into parameter name and value
		std::map<std::string, TensorT> parameters_map;
		for (std::string str : str_tokens)
		{
			str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
			std::regex re1(":");
			std::vector<std::string> params;
			std::copy(
				std::sregex_token_iterator(str.begin(), str.end(), re1, -1),
				std::sregex_token_iterator(),
				std::back_inserter(params));
			std::string param_name = params[0];
			TensorT param_value = 0.0;
			try
			{
				param_value = std::stof(params[1]);
			}
			catch (std::exception& e)
			{
				printf("Exception: %s", e.what());
			}
			parameters_map.emplace(param_name, param_value);
		}

		return parameters_map;
	}


	template<typename TensorT>
	bool WeightFile<TensorT>::storeWeightsBinary(const std::string& filename, const std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights) {
		std::ofstream ofs(filename, std::ios::binary);
		//if (ofs.is_open() == false) { // Lines check to make sure the file is not already created
		cereal::BinaryOutputArchive oarchive(ofs);
		oarchive(weights);
		ofs.close();
		//} // Lines check to make sure the file is not already created
		return true; 
	}

	template<typename TensorT>
	bool WeightFile<TensorT>::storeWeightsCsv(const std::string& filename, const std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights)
	{
		CSVWriter csvwriter(filename);

		// write the headers to the first line
		std::vector<std::string> headers = { "weight_name", "weight_init_op", "weight_init_params", "solver_op", "solver_params", "weight_value", "module_name", "layer_name", "tensor_index"};
		int n_operations = 0;
		if (weights.size() != 0)
			n_operations = weights.begin()->second->getTensorIndex().size();
		csvwriter.writeDataInRow(headers.begin(), headers.end());

		for (const auto& weight : weights)
		{
			std::vector<std::string> row;

			row.push_back(weight.second->getName());

			// parse the weight_init_op
			const std::string weight_init_op_name = weight.second->getWeightInitOp()->getName();
			row.push_back(weight_init_op_name);

			// parse the weight_init_params
			row.push_back(weight.second->getWeightInitOp()->getParamsAsStr());

			// parse the solver_op
			const std::string solver_op_name = weight.second->getSolverOp()->getName();
			row.push_back(solver_op_name);

			// parse the solver_op_params
			row.push_back(weight.second->getSolverOp()->getParamsAsStr());

			// parse the weight value
			row.push_back(std::to_string(weight.second->getWeight()));

			// parse the module name
			row.push_back(weight.second->getModuleName());

			// parse the tensor indexing
			row.push_back(weight.second->getLayerName());
			std::string tensor_index = "";
			for (int i = 0; i < n_operations; ++i) {
				tensor_index += "{" + std::to_string(std::get<0>(weight.second->getTensorIndex()[i])) + ";"
					+ std::to_string(std::get<1>(weight.second->getTensorIndex()[i])) + ";"
					+ std::to_string(std::get<2>(weight.second->getTensorIndex()[i])) + "}";
				if (i < n_operations - 1)	tensor_index += "|";
			}
			row.push_back(tensor_index);

			// write to file
			csvwriter.writeDataInRow(row.begin(), row.end());
		}
		return true;
	}
	template<typename TensorT>
	inline bool WeightFile<TensorT>::storeWeightValuesCsv(const std::string & filename, const std::map<std::string, std::shared_ptr<Weight<TensorT>>>& weights)
	{
		CSVWriter csvwriter(filename);

		// write the headers to the first line
		const std::vector<std::string> headers = { "weight_name", "weight_value" };
		csvwriter.writeDataInRow(headers.begin(), headers.end());

		for (const auto& weight : weights)
		{
			std::vector<std::string> row;

			row.push_back(weight.second->getName());

			// parse the weight value
			row.push_back(std::to_string(weight.second->getWeight()));

			// write to file
			csvwriter.writeDataInRow(row.begin(), row.end());
		}
		return true;
	}
}

#endif //SMARTPEAK_WEIGHTFILE_H