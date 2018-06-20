/**TODO:  Add copyright*/

#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

#include <map>
#include <regex>

namespace SmartPeak
{

  WeightFile::WeightFile(){}
  WeightFile::~WeightFile(){}
 
  bool WeightFile::loadWeightsBinary(const std::string& filename, std::vector<Weight>& weights) { return true; }

  bool WeightFile::loadWeightsCsv(const std::string& filename, std::vector<Weight>& weights)
  {
    weights.clear();

    io::CSVReader<6> weights_in(filename);
    weights_in.read_header(io::ignore_extra_column, 
      "weight_name", "weight_init_op", "weight_init_params", "solver_op", "solver_params", "weight_value");
    std::string weight_name, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str, weight_value_str;

    while(weights_in.read_row(weight_name, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str, weight_value_str))
    {
      // parse the weight_init_params
      std::map<std::string, float> weight_init_params = parseParameters(weight_init_params_str);

      // parse the weight_init_op
      std::shared_ptr<WeightInitOp> weight_init;
      if (weight_init_op_str == "ConstWeightInitOp")
      {
        ConstWeightInitOp* ptr = nullptr;
        if (weight_init_params.count("n"))
          ptr = new ConstWeightInitOp(weight_init_params.at("n"));
        else
          ptr = new ConstWeightInitOp(1.0);
        weight_init.reset(ptr);
      }
      else if (weight_init_op_str == "RandWeightInitOp")
      {
        RandWeightInitOp* ptr = nullptr;
        if (weight_init_params.count("n"))
          ptr = new RandWeightInitOp(weight_init_params.at("n"));
        else
          ptr = new RandWeightInitOp(1.0);
        weight_init.reset(ptr);
      }
      else std::cout<<"WeightInitOp "<<weight_init_op_str<<" for weight_name "<<weight_name<<" was not recognized."<<std::endl;

      // parse the solver_params_str
      std::map<std::string, float> solver_params = parseParameters(solver_params_str);

      // parse the solver_op
      std::shared_ptr<SolverOp> solver;
      if (solver_op_str == "SGDOp")
      {
        SGDOp* ptr = new SGDOp();
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
        AdamOp* ptr = new AdamOp();
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
      else std::cout<<"SolverOp "<<solver_op_str<<" for weight_name "<<weight_name<<" was not recognized."<<std::endl;

      Weight weight(weight_name, weight_init, solver);

      // parse the weight value
      float weight_value = 0;
      try
      {
        weight_value = std::stof(weight_value_str);
      }
      catch (std::exception& e)
      {
        printf("Exception: %s", e.what());
      }

      weights.push_back(weight);
    }
	return true;
  }

  std::map<std::string, float> WeightFile::parseParameters(const std::string& parameters)
  {
    // parse the parameters
    std::regex re(";");
    std::vector<std::string> str_tokens;
    std::copy(
      std::sregex_token_iterator(parameters.begin(), parameters.end(), re, -1),
      std::sregex_token_iterator(),
      std::back_inserter(str_tokens));

    // break into parameter name and value
    std::map<std::string, float> parameters_map;
    for (std::string str: str_tokens)
    {
      str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
      std::regex re1(":");
      std::vector<std::string> params;
      std::copy(
        std::sregex_token_iterator(str.begin(), str.end(), re1, -1),
        std::sregex_token_iterator(),
        std::back_inserter(params));
      std::string param_name = params[0];
      float param_value = 0.0;
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
  

  bool WeightFile::storeWeightsBinary(const std::string& filename, const std::vector<Weight>& weights) { return true; }

  bool WeightFile::storeWeightsCsv(const std::string& filename, const std::vector<Weight>& weights)
  {
    CSVWriter csvwriter(filename);

    // write the headers to the first line
    const std::vector<std::string> headers = {"weight_name", "weight_init_op", "weight_init_params", "solver_op", "solver_params", "weight_value"};
    csvwriter.writeDataInRow(headers.begin(), headers.end());

    for (const Weight& weight: weights)
    {
      std::vector<std::string> row;

      row.push_back(weight.getName());
      
      // parse the weight_init_op
      const std::string weight_init_op_name = weight.getWeightInitOp()->getName();
      row.push_back(weight_init_op_name);

      // parse the weight_init_params
      row.push_back(weight.getWeightInitOp()->getParameters());

      // parse the solver_op
      const std::string solver_op_name = weight.getSolverOp()->getName();
      row.push_back(solver_op_name);

      // parse the solver_op_params
      row.push_back(weight.getSolverOp()->getParameters());

      // parse the weight value
      row.push_back(std::to_string(weight.getWeight()));

      // write to file
      csvwriter.writeDataInRow(row.begin(), row.end());
    }
	return true;
  }
}