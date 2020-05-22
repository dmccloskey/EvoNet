/**TODO:  Add copyright*/

#ifndef SMARTPEAK_PARAMETERS_H
#define SMARTPEAK_PARAMETERS_H

// .h
#include <tuple>
#include <string>

// .cpp
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

namespace SmartPeak
{
  /**
    @brief Parameter
  */
  template<typename T>
  struct Parameter {
    std::string name_;
    std::string s_;
    T value_;
    Parameter(const std::string& name, const T& value) : name_(name), value_(value) { std::stringstream ss; ss << value; ss >> s_; };
    void set() { if (!s_.empty()) { std::stringstream ss; ss << s_; ss >> value_; } }
    T get() { return value_; }
    friend std::ostream& operator<<(std::ostream& os, const Parameter& parameter) { os << parameter.name_ << ": " << parameter.value_; return os; }
  };

  /// List of all available parameters and their types
  struct ID : Parameter<int> { using Parameter::Parameter; };
  struct DataDir : Parameter<std::string> { using Parameter::Parameter; };
  struct NInterpreters : Parameter<int> { using Parameter::Parameter; };
  struct NGenerations : Parameter<int> { using Parameter::Parameter; };
  struct MakeModel : Parameter<bool> { using Parameter::Parameter; };
  struct TrainModel : Parameter<bool> { using Parameter::Parameter; };
  struct EvolveModel : Parameter<bool> { using Parameter::Parameter; };
  struct EvaluateModel : Parameter<bool> { using Parameter::Parameter; };
  struct SimulationType : Parameter<std::string> { using Parameter::Parameter; };
  struct BatchSize : Parameter<int> { using Parameter::Parameter; };
  struct MemorySize : Parameter<int> { using Parameter::Parameter; };
  struct NEpochsTraining : Parameter<int> { using Parameter::Parameter; };
  struct NEpochsValidation : Parameter<int> { using Parameter::Parameter; };
  struct NEpochsEvaluation : Parameter<int> { using Parameter::Parameter; };
  struct NTBTTSteps : Parameter<int> { using Parameter::Parameter; };
  struct DeviceId : Parameter<int> { using Parameter::Parameter; };
  struct ModelName : Parameter<std::string> { using Parameter::Parameter; };

  /// Helper method to statically deduce the size of a tuple
  template<typename Tuple>
  constexpr size_t sizeOfParameters(const Tuple& t) {
    return std::tuple_size<Tuple>::value;
  }

  /*
  @brief Helper method to parse the command line arguments

  @param[in] argc
  @param[in] argv
  @param[in,out] id
  @param[in,out] parameters_file
  */
  void parseCommandLineArguments(int argc, char** argv, int& id, std::string& parameters_file) {
    if (argc >= 2) {
      try {
        id_int = std::stoi(argv[1]);
      }
      catch (std::exception & e) {
        std::cout << e.what() << std::endl;
      }
    }
    if (argc >= 3) {
      parameters_file = std::string(argv[2]);
    }
  }

  /*
  @brief Load the parameters from file

  @param[in] argc
  @param[in] argv
  @param[in,out] id
  @param[in,out] parameters_file
  */
  template<class ...ParameterTypes>
  void loadParametersFromCsv(const int& id, const std::string& parameters_file, ParameterTypes&... args) {
    auto parameters = std::make_tuple(args...);

    // Read in the parameters
    io::CSVReader<sizeOfParameters(parameters)> parameters_in(parameters_file);
    std::apply([&parameters_in](auto&& ...args) { parameters_in.read_header(io::ignore_extra_column, args.name_ ...); }, parameters);
    while (std::apply([&parameters_in](auto&& ...args) { return parameters_in.read_row(args.s_ ...); }, parameters))
    {
      if (std::to_string(id_int) == id.s_) {
        std::apply([](auto&& ...args) {((args.set()), ...); }, parameters);
        break;
      }
    }

    // Print the read in parameters to the screen
    std::apply([](auto&&... args) {((std::cout << args << std::endl), ...); }, parameters);
  }
}

#endif //SMARTPEAK_PARAMETERS_H