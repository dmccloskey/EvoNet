/**TODO:  Add copyright*/

#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/csv.h>

#include <map>

namespace SmartPeak
{

  WeightFile::WeightFile(){}
  WeightFile::~WeightFile(){}
 
  bool WeightFile::loadWeightsBinary(const std::string& filename, std::vector<Weight>& weights){}

  bool WeightFile::loadWeightsCsv(const std::string& filename, std::vector<Weight>& weights)
  {
    io::CSVReader<5> weights_in(filename);
    weights_in.read_header(io::ignore_extra_column, 
      "weight_name", "weight_init_op", "weight_init_params", "solver_op", "solver_params");
    std::string weight_name, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str;

    while(weights_in.read_row(weight_name, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str))
    {
      // parse the weight_init_params
      std::map<std::string, float> weight_init_params;
      // TODO...

      // parse the weight_init_op
      std::shared_ptr<WeightInitOp> weight_init;
      if (weight_init_op_str == "ConstWeightInitOp")
      {
        weight_init.reset(new ConstWeightInitOp(1.0));
      }
      else if (weight_init_op_str == "RandWeightInitOp")
      {
        weight_init.reset(new RandWeightInitOp(1.0));
      }
      else std::cout<<"WeightInitOp for weight_name "<<weight_name<<" was not recognized."<<std::endl;

      // parse the solver_params_str
      std::map<std::string, float> weight_params;
      // TODO...

      // parse the solver_op
      std::shared_ptr<SolverOp> solver;
      if (weight_init_op_str == "SGDOp")
      {
        solver.reset(new SGDOp(0.01, 0.9));
      }
      else if (weight_init_op_str == "AdamOp")
      {
        solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
      }
      else std::cout<<"WeightInitOp for weight_name "<<weight_name<<" was not recognized."<<std::endl;

      Weight weight(weight_name, weight_init, solver);
      weights.push_back(weight);
    }
  }

  bool WeightFile::storeWeightsBinary(const std::string& filename, const std::vector<Weight>& weights){}

  bool WeightFile::storeWeightsCsv(const std::string& filename, const std::vector<Weight>& weights){}
}