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
  /// Implementation of `std::invoke` for C++11 on CUDA
  /// Referece: https://stackoverflow.com/questions/34668720/stdapply-may-not-be-properly-implemented
  template <typename F, typename Tuple, size_t... I>
  decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
    return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
  }
  template <typename F, typename Tuple>
  decltype(auto) apply(F&& f, Tuple&& t) {
    using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
    return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
  }

  /// List of all available parameters and their types
  namespace EvoNetParameters {
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

    /// General
    struct ID : Parameter<int> { using Parameter::Parameter; };
    struct DataDir : Parameter<std::string> { using Parameter::Parameter; };

    /// Example main
    struct MakeModel : Parameter<bool> { using Parameter::Parameter; };
    struct TrainModel : Parameter<bool> { using Parameter::Parameter; };
    struct EvolveModel : Parameter<bool> { using Parameter::Parameter; };
    struct EvaluateModel : Parameter<bool> { using Parameter::Parameter; };
    struct SimulationType : Parameter<std::string> { using Parameter::Parameter; };
    struct ModelName : Parameter<std::string> { using Parameter::Parameter; };
    struct DeviceId : Parameter<int> { using Parameter::Parameter; };

    /// PopulationTrainer
    struct PopulationName : Parameter<std::string> { using Parameter::Parameter; };
    struct NInterpreters : Parameter<int> { using Parameter::Parameter; };
    struct NGenerations : Parameter<int> { using Parameter::Parameter; };
    //TODO: all members
    //TODO: PopulationSizeFixed

    /// ModelReplicator
    //TODO: lb/ub for all model modifications
    //TODO: ModificationRateFixed bool

    /// ModelTrainer
    struct BatchSize : Parameter<int> { using Parameter::Parameter; };
    struct MemorySize : Parameter<int> { using Parameter::Parameter; };
    struct NEpochsTraining : Parameter<int> { using Parameter::Parameter; };
    struct NEpochsValidation : Parameter<int> { using Parameter::Parameter; };
    struct NEpochsEvaluation : Parameter<int> { using Parameter::Parameter; };
    struct NTBTTSteps : Parameter<int> { using Parameter::Parameter; };
    struct FindCycles : Parameter<bool> { using Parameter::Parameter; };
    struct FastInterpreter : Parameter<bool> { using Parameter::Parameter; };
    //TODO: all members
  }

  /// Helper method to statically deduce the size of a tuple
  template<class Tuple>
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
        id = std::stoi(argv[1]);
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
  @brief Struct to load parameters from csv
  */
  struct LoadParametersFromCsv {
    LoadParametersFromCsv(const int& id, const std::string& parameters_filename) :id_(id), parameters_filename_(parameters_filename) {}
    int id_;
    std::string parameters_filename_;
    /*
    @brief Load the parameters from file

    @param[in] argc
    @param[in] argv
    @param[in,out] id
    @param[in,out] parameters_file
    */
    template<class ...ParameterTypes>
    std::tuple<ParameterTypes...> operator()(ParameterTypes&... args) {
      auto parameters = std::make_tuple(args...);
      // Read in the parameters
      io::CSVReader<sizeOfParameters(parameters)> parameters_in(parameters_filename_);
      SmartPeak::apply([&parameters_in](auto&& ...args) { parameters_in.read_header(io::ignore_extra_column, args.name_ ...); }, parameters);
      while (SmartPeak::apply([&parameters_in](auto&& ...args) { return parameters_in.read_row(args.s_ ...); }, parameters))
      {
        if (std::to_string(id_) == std::get<EvoNetParameters::ID>(parameters).s_) {
          //SmartPeak::apply([](auto&& ...args) {((args.set()), ...); }, parameters); // C++17
          SmartPeak::apply([](auto&& ...args) { // C++11/CUDA
            using expander = int[];
            (void)expander {0, (void(args.set()), 0)...};
          }, parameters);
          break;
        }
      }
      // Print the read in parameters to the screen
      //SmartPeak::apply([](auto&&... args) {((std::cout << args << std::endl), ...); }, parameters); // C++17
      SmartPeak::apply([](auto&&... args) { // C++11/CUDA
        using expander = int[];
        (void)expander {
          0, (void(std::cout << args << std::endl), 0)...
        };
      }, parameters);
      return parameters;
    }
  };
}

#endif //SMARTPEAK_PARAMETERS_H