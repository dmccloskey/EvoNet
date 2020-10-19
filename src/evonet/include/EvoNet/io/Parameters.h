/**TODO:  Add copyright*/

#ifndef EVONET_PARAMETERS_H
#define EVONET_PARAMETERS_H

// .h
#include <tuple>
#include <string>

// .cpp
#include <EvoNet/io/csv.h>
#include <EvoNet/io/CSVWriter.h>

namespace EvoNet
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

    namespace General {
      struct ID : Parameter<int> { using Parameter::Parameter; };
      struct DataDir : Parameter<std::string> { using Parameter::Parameter; };
      struct OutputDir : Parameter<std::string> { using Parameter::Parameter; };
      struct InputDir : Parameter<std::string> { using Parameter::Parameter; };
    }

    namespace Main {
      struct MakeModel : Parameter<bool> { using Parameter::Parameter; };
      struct TrainModel : Parameter<bool> { using Parameter::Parameter; };
      struct EvolveModel : Parameter<bool> { using Parameter::Parameter; };
      struct EvaluateModel : Parameter<bool> { using Parameter::Parameter; };
      struct EvaluateModels : Parameter<bool> { using Parameter::Parameter; };
      struct LoadModelCsv : Parameter<bool> { using Parameter::Parameter; };
      struct LoadModelBinary : Parameter<bool> { using Parameter::Parameter; };
      struct ModelName : Parameter<std::string> { using Parameter::Parameter; };
      struct DeviceId : Parameter<int> { using Parameter::Parameter; };
    }

    namespace PopulationTrainer {
      struct PopulationName : Parameter<std::string> { using Parameter::Parameter; };
      struct PopulationSize : Parameter<int> { using Parameter::Parameter; };
      struct NInterpreters : Parameter<int> { using Parameter::Parameter; };
      struct NTop : Parameter<int> { using Parameter::Parameter; };
      struct NRandom : Parameter<int> { using Parameter::Parameter; };
      struct NReplicatesPerModel : Parameter<int> { using Parameter::Parameter; };
      struct Logging : Parameter<bool> { using Parameter::Parameter; };
      struct RemoveIsolatedNodes : Parameter<bool> { using Parameter::Parameter; };
      struct PruneModelNum : Parameter<int> { using Parameter::Parameter; };
      struct CheckCompleteModelInputToOutput : Parameter<bool> { using Parameter::Parameter; };
      //struct SelectModels : Parameter<bool> { using Parameter::Parameter; };
      struct ResetModelCopyWeights : Parameter<bool> { using Parameter::Parameter; };
      struct ResetModelTemplateWeights : Parameter<bool> { using Parameter::Parameter; };
      struct NGenerations : Parameter<int> { using Parameter::Parameter; };
      struct SetPopulationSizeFixed : Parameter<bool> { using Parameter::Parameter; };
      struct SetPopulationSizeDoubling : Parameter<bool> { using Parameter::Parameter; };
      struct SetTrainingStepsByModelSize : Parameter<bool> { using Parameter::Parameter; };
    }

    namespace ModelReplicator {
      struct NNodeDownAdditionsLB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeRightAdditionsLB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeDownCopiesLB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeRightCopiesLB : Parameter<int> { using Parameter::Parameter; };
      struct NLinkAdditionsLB : Parameter<int> { using Parameter::Parameter; };
      struct NLinkCopiesLB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeDeletionsLB : Parameter<int> { using Parameter::Parameter; };
      struct NLinkDeletionsLB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeActivationChangesLB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeIntegrationChangesLB : Parameter<int> { using Parameter::Parameter; };
      struct NModuleAdditionsLB : Parameter<int> { using Parameter::Parameter; };
      struct NModuleCopiesLB : Parameter<int> { using Parameter::Parameter; };
      struct NModuleDeletionsLB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeDownAdditionsUB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeRightAdditionsUB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeDownCopiesUB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeRightCopiesUB : Parameter<int> { using Parameter::Parameter; };
      struct NLinkAdditionsUB : Parameter<int> { using Parameter::Parameter; };
      struct NLinkCopiesUB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeDeletionsUB : Parameter<int> { using Parameter::Parameter; };
      struct NLinkDeletionsUB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeActivationChangesUB : Parameter<int> { using Parameter::Parameter; };
      struct NNodeIntegrationChangesUB : Parameter<int> { using Parameter::Parameter; };
      struct NModuleAdditionsUB : Parameter<int> { using Parameter::Parameter; };
      struct NModuleCopiesUB : Parameter<int> { using Parameter::Parameter; };
      struct NModuleDeletionsUB : Parameter<int> { using Parameter::Parameter; };
      struct SetModificationRateFixed : Parameter<bool> { using Parameter::Parameter; };
      struct SetModificationRateByPrevError : Parameter<bool> { using Parameter::Parameter; };
    }

    namespace ModelTrainer {
      struct BatchSize : Parameter<int> { using Parameter::Parameter; };
      struct MemorySize : Parameter<int> { using Parameter::Parameter; };
      struct NEpochsTraining : Parameter<int> { using Parameter::Parameter; };
      struct NEpochsValidation : Parameter<int> { using Parameter::Parameter; };
      struct NEpochsEvaluation : Parameter<int> { using Parameter::Parameter; };
      struct Verbosity : Parameter<int> { using Parameter::Parameter; };
      struct LoggingTraining : Parameter<bool> { using Parameter::Parameter; };
      struct LoggingValidation : Parameter<bool> { using Parameter::Parameter; };
      struct LoggingEvaluation : Parameter<bool> { using Parameter::Parameter; };
      struct NTBTTSteps : Parameter<int> { using Parameter::Parameter; };
      struct NTETTSteps : Parameter<int> { using Parameter::Parameter; };
      struct FindCycles : Parameter<bool> { using Parameter::Parameter; };
      struct FastInterpreter : Parameter<bool> { using Parameter::Parameter; };
      struct PreserveOoO : Parameter<bool> { using Parameter::Parameter; };
      struct InterpretModel : Parameter<bool> { using Parameter::Parameter; };
      struct ResetModel : Parameter<bool> { using Parameter::Parameter; };
      struct ResetInterpreter : Parameter<bool> { using Parameter::Parameter; };
      struct LossFunction : Parameter<std::string> { using Parameter::Parameter; };
      /// Model building
      struct NHidden0 : Parameter<int> { using Parameter::Parameter; };
      struct NHidden1 : Parameter<int> { using Parameter::Parameter; };
      struct NHidden2 : Parameter<int> { using Parameter::Parameter; };
      struct LossFncWeight0 : Parameter<float> { using Parameter::Parameter; };
      struct LossFncWeight1 : Parameter<float> { using Parameter::Parameter; };
      struct LossFncWeight2 : Parameter<float> { using Parameter::Parameter; };
      struct AddGaussian : Parameter<bool> { using Parameter::Parameter; };
      struct AddMixedGaussian : Parameter<bool> { using Parameter::Parameter; };
      struct AddCategorical : Parameter<bool> { using Parameter::Parameter; };
      struct LearningRate : Parameter<float> { using Parameter::Parameter; };
      struct GradientClipping : Parameter<float> { using Parameter::Parameter; };
      struct KLDivergenceWarmup : Parameter<bool> { using Parameter::Parameter; };
      struct NEncodingsContinuous : Parameter<int> { using Parameter::Parameter; };
      struct NEncodingsCategorical : Parameter<int> { using Parameter::Parameter; };
      struct Beta : Parameter<float> { using Parameter::Parameter; };
      struct CapacityC : Parameter<float> { using Parameter::Parameter; };
      struct CapacityD : Parameter<float> { using Parameter::Parameter; };
    }

    namespace Examples {
      struct NMask : Parameter<int> { using Parameter::Parameter; };
      struct SequenceLength : Parameter<int> { using Parameter::Parameter; };
      struct SimulationType : Parameter<std::string> { using Parameter::Parameter; };
      struct ModelType : Parameter<std::string> { using Parameter::Parameter; };
      struct BiochemicalRxnsFilename : Parameter<std::string> { using Parameter::Parameter; };
      struct SupervisionWarmup : Parameter<bool> { using Parameter::Parameter; };
      struct SupervisionPercent : Parameter<float> { using Parameter::Parameter; };
    }
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
      EvoNet::apply([&parameters_in](auto&& ...args) { parameters_in.read_header(io::ignore_extra_column, args.name_ ...); }, parameters);
      while (EvoNet::apply([&parameters_in](auto&& ...args) { return parameters_in.read_row(args.s_ ...); }, parameters))
      {
        if (std::to_string(id_) == std::get<EvoNetParameters::General::ID>(parameters).s_) {
          //EvoNet::apply([](auto&& ...args) {((args.set()), ...); }, parameters); // C++17
          EvoNet::apply([](auto&& ...args) { // C++11/CUDA
            using expander = int[];
            (void)expander {0, (void(args.set()), 0)...};
          }, parameters);
          break;
        }
      }
      // Print the read in parameters to the screen
      //EvoNet::apply([](auto&&... args) {((std::cout << args << std::endl), ...); }, parameters); // C++17
      EvoNet::apply([](auto&&... args) { // C++11/CUDA
        using expander = int[];
        (void)expander {
          0, (void(std::cout << args << std::endl), 0)...
        };
      }, parameters);
      return parameters;
    }
  };

  /// Helper method to set the PopulationTrainer parameters
  template<typename PopulationTrainerT, class ...ParameterTypes>
  void setPopulationTrainerParameters(PopulationTrainerT& population_trainer, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    // set the population trainer parameters
    population_trainer.setNGenerations(std::get<EvoNetParameters::PopulationTrainer::NGenerations>(parameters).get());
    population_trainer.setPopulationSize(std::get<EvoNetParameters::PopulationTrainer::PopulationSize>(parameters).get());
    population_trainer.setNReplicatesPerModel(std::get<EvoNetParameters::PopulationTrainer::NReplicatesPerModel>(parameters).get());
    population_trainer.setNTop(std::get<EvoNetParameters::PopulationTrainer::NTop>(parameters).get());
    population_trainer.setNRandom(std::get<EvoNetParameters::PopulationTrainer::NRandom>(parameters).get());
    population_trainer.setLogging(std::get<EvoNetParameters::PopulationTrainer::Logging>(parameters).get());
    population_trainer.setRemoveIsolatedNodes(std::get<EvoNetParameters::PopulationTrainer::RemoveIsolatedNodes>(parameters).get());
    population_trainer.setPruneModelNum(std::get<EvoNetParameters::PopulationTrainer::PruneModelNum>(parameters).get());
    population_trainer.setCheckCompleteModelInputToOutput(std::get<EvoNetParameters::PopulationTrainer::CheckCompleteModelInputToOutput>(parameters).get());
    population_trainer.setResetModelCopyWeights(std::get<EvoNetParameters::PopulationTrainer::ResetModelCopyWeights>(parameters).get());
    population_trainer.setResetModelTemplateWeights(std::get<EvoNetParameters::PopulationTrainer::ResetModelTemplateWeights>(parameters).get());
    population_trainer.set_population_size_fixed_ = std::get<EvoNetParameters::PopulationTrainer::SetPopulationSizeFixed>(parameters).get();
    population_trainer.set_population_size_doubling_ = std::get<EvoNetParameters::PopulationTrainer::SetPopulationSizeDoubling>(parameters).get();
    population_trainer.set_training_steps_by_model_size_ = std::get<EvoNetParameters::PopulationTrainer::SetTrainingStepsByModelSize>(parameters).get();
  }

  /// Helper method to set the ModelReplicator parameters
  template<typename ModelReplicatorT, class ...ParameterTypes>
  void setModelReplicatorParameters(ModelReplicatorT& model_replicator, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    // set the model replicator parameters
    model_replicator.setNodeActivations({ std::make_pair(std::make_shared<ReLUOp<float>>(ReLUOp<float>()), std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>())),
      std::make_pair(std::make_shared<LinearOp<float>>(LinearOp<float>()), std::make_shared<LinearGradOp<float>>(LinearGradOp<float>())),
      std::make_pair(std::make_shared<ELUOp<float>>(ELUOp<float>()), std::make_shared<ELUGradOp<float>>(ELUGradOp<float>())),
      std::make_pair(std::make_shared<SigmoidOp<float>>(SigmoidOp<float>()), std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>())),
      std::make_pair(std::make_shared<TanHOp<float>>(TanHOp<float>()), std::make_shared<TanHGradOp<float>>(TanHGradOp<float>()))//,
      //std::make_pair(std::make_shared<ExponentialOp<float>>(ExponentialOp<float>()), std::make_shared<ExponentialGradOp<float>>(ExponentialGradOp<float>())),
      //std::make_pair(std::make_shared<LogOp<float>>(LogOp<float>()), std::make_shared<LogGradOp<float>>(LogGradOp<float>())),
      //std::make_pair(std::shared_ptr<ActivationOp<float>>(new InverseOp<float>()), std::shared_ptr<ActivationOp<float>>(new InverseGradOp<float>()))
      });
    model_replicator.setNodeIntegrations({ std::make_tuple(std::make_shared<ProdOp<float>>(ProdOp<float>()), std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>()), std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>())),
      std::make_tuple(std::make_shared<SumOp<float>>(SumOp<float>()), std::make_shared<SumErrorOp<float>>(SumErrorOp<float>()), std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>())),
      //std::make_tuple(std::make_shared<MeanOp<float>>(MeanOp<float>()), std::make_shared<MeanErrorOp<float>>(MeanErrorOp<float>()), std::make_shared<MeanWeightGradO<float>>(MeanWeightGradOp<float>())),
      //std::make_tuple(std::make_shared<VarModOp<float>>(VarModOp<float>()), std::make_shared<VarModErrorOp<float>>(VarModErrorOp<float>()), std::make_shared<VarModWeightGradOp<float>>(VarModWeightGradOp<float>())),
      //std::make_tuple(std::make_shared<CountOp<float>>(CountOp<float>()), std::make_shared<CountErrorOp<float>>(CountErrorOp<float>()), std::make_shared<CountWeightGradOp<float>>(CountWeightGradOp<float>()))
      });
    model_replicator.set_modification_rate_by_prev_error_ = std::get<EvoNetParameters::ModelReplicator::SetModificationRateByPrevError>(parameters).get();
    model_replicator.set_modification_rate_fixed_ = std::get<EvoNetParameters::ModelReplicator::SetModificationRateFixed>(parameters).get();
    model_replicator.setRandomModifications(
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeDownAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeDownAdditionsUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeRightAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeRightAdditionsUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeDownCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeDownCopiesUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeRightCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeRightCopiesUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NLinkAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NLinkAdditionsUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NLinkCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NLinkCopiesUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeDeletionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeDeletionsUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NLinkDeletionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NLinkDeletionsUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeActivationChangesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeActivationChangesUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NNodeIntegrationChangesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NNodeIntegrationChangesUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NModuleAdditionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NModuleAdditionsUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NModuleCopiesLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NModuleCopiesUB>(parameters).get()),
      std::make_pair(std::get<EvoNetParameters::ModelReplicator::NModuleDeletionsLB>(parameters).get(), std::get<EvoNetParameters::ModelReplicator::NModuleDeletionsUB>(parameters).get()));
  }

  /// Helper method to set the number of threads and assign the resources for the model interpreters
  template<typename ModelInterpreterT, class ...ParameterTypes>
  void setModelInterpreterParameters(std::vector<ModelInterpreterT>& model_interpreters, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    // define the multithreading parameters
    const int n_hard_threads = std::thread::hardware_concurrency();
    const int n_threads = (std::get<EvoNetParameters::PopulationTrainer::NInterpreters>(parameters).get() > n_hard_threads) ? n_hard_threads : std::get<EvoNetParameters::PopulationTrainer::NInterpreters>(parameters).get(); // the number of threads
    // define the model trainers and resources for the trainers
    for (size_t i = 0; i < n_threads; ++i) {
      ModelResources model_resources = { ModelDevice(std::get<EvoNetParameters::Main::DeviceId>(parameters).get(), 1) };
      ModelInterpreterT model_interpreter(model_resources);
      model_interpreters.push_back(model_interpreter);
    }
  }

  /// Helper method to set the ModelTrainer parameters
  template<typename ModelTrainerT, class ...ParameterTypes>
  void setModelTrainerParameters(ModelTrainerT& model_trainer, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    // set the model trainer
    model_trainer.setBatchSize(std::get<EvoNetParameters::ModelTrainer::BatchSize>(parameters).get());
    model_trainer.setMemorySize(std::get<EvoNetParameters::ModelTrainer::MemorySize>(parameters).get());
    model_trainer.setNEpochsTraining(std::get<EvoNetParameters::ModelTrainer::NEpochsTraining>(parameters).get());
    model_trainer.setNEpochsValidation(std::get<EvoNetParameters::ModelTrainer::NEpochsValidation>(parameters).get());
    model_trainer.setNEpochsEvaluation(std::get<EvoNetParameters::ModelTrainer::NEpochsEvaluation>(parameters).get());
    model_trainer.setNTBPTTSteps(std::get<EvoNetParameters::ModelTrainer::NTBTTSteps>(parameters).get());
    model_trainer.setNTETTSteps(std::get<EvoNetParameters::ModelTrainer::NTETTSteps>(parameters).get());
    model_trainer.setVerbosityLevel(std::get<EvoNetParameters::ModelTrainer::Verbosity>(parameters).get());
    model_trainer.setLogging(std::get<EvoNetParameters::ModelTrainer::LoggingTraining>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::LoggingValidation>(parameters).get(),
      std::get<EvoNetParameters::ModelTrainer::LoggingEvaluation>(parameters).get());
    model_trainer.setFindCycles(std::get<EvoNetParameters::ModelTrainer::FindCycles>(parameters).get());
    model_trainer.setFastInterpreter(std::get<EvoNetParameters::ModelTrainer::FastInterpreter>(parameters).get());
    model_trainer.setPreserveOoO(std::get<EvoNetParameters::ModelTrainer::PreserveOoO>(parameters).get());
    model_trainer.setInterpretModel(std::get<EvoNetParameters::ModelTrainer::InterpretModel>(parameters).get());
    model_trainer.setResetModel(std::get<EvoNetParameters::ModelTrainer::ResetModel>(parameters).get());
    model_trainer.setResetInterpreter(std::get<EvoNetParameters::ModelTrainer::ResetInterpreter>(parameters).get());
  }

  /// Helper method to read in a trained model
  template<typename ModelT, typename InterpreterT, typename ModelFileT, typename InterpreterFileT, class ...ParameterTypes>
  void loadModelFromParameters(ModelT& model, InterpreterT& interpreter, ModelFileT& model_file, InterpreterFileT& interpreter_file, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    // read in the trained model
    if (std::get<EvoNetParameters::Main::LoadModelBinary>(parameters).get()) {
      std::cout << "Reading in the model from binary..." << std::endl;
      model_file.loadModelBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_model.binary", model);
      model.setId(1);
      interpreter_file.loadModelInterpreterBinary(std::get<EvoNetParameters::General::OutputDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_interpreter.binary", interpreter);
    }
    else if (std::get<EvoNetParameters::Main::LoadModelCsv>(parameters).get()) {
      // read in the trained model
      std::cout << "Reading in the model from csv..." << std::endl;
      model_file.loadModelCsv(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_nodes.csv", std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_links.csv", std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::Main::ModelName>(parameters).get() + "_weights.csv", model, true, true, true);
      model.setId(1);
    }
  }

  /// Helper method to train, evaluate, or evolve from parameters
  template<typename TensorT, typename ModelT, typename InterpreterT, typename ModelTrainerT, typename PopulationTrainerT, typename ModelReplicatorT, typename DataSimulatorT, typename ModelLoggerT, typename PopulationLoggerT, class ...ParameterTypes>
  void runTrainEvalEvoFromParameters(ModelT& model, std::vector<InterpreterT>& model_interpreters, ModelTrainerT& model_trainer, PopulationTrainerT& population_trainer, ModelReplicatorT& model_replicator, DataSimulatorT& data_simulator, ModelLoggerT& model_logger, PopulationLoggerT& population_logger, const std::vector<std::string>& input_nodes, const ParameterTypes& ...args) {
    auto parameters = std::make_tuple(args...);
    if (std::get<EvoNetParameters::Main::TrainModel>(parameters).get()) {
      // Train the model
      model.setName(model.getName() + "_train");
      std::pair<std::vector<TensorT>, std::vector<TensorT>> model_errors = model_trainer.trainModel(model, data_simulator, input_nodes, model_logger, model_interpreters.front());
    }
    else if (std::get<EvoNetParameters::Main::EvolveModel>(parameters).get()) {
      // Evolve the population
      std::vector<ModelT> population = { model };
      std::vector<std::vector<std::tuple<int, std::string, TensorT>>> models_validation_errors_per_generation = population_trainer.evolveModels(
        population, std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get(), //So that all output will be written to a specific directory
        model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, population_logger, input_nodes);
      //// Write the evolved population to disk
      //PopulationTrainerFile<float> population_trainer_file;
      //population_trainer_file.storeModels(population, std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get());
      //population_trainer_file.storeModelValidations(std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get() + "Errors.csv", models_validation_errors_per_generation);
    }
    else if (std::get<EvoNetParameters::Main::EvaluateModel>(parameters).get()) {
      // Evaluate the model
      model.setName(model.getName() + "_evaluation");
      Eigen::Tensor<TensorT, 4> model_output = model_trainer.evaluateModel(model, data_simulator, input_nodes, model_logger, model_interpreters.front());
    }
    else if (std::get<EvoNetParameters::Main::EvaluateModels>(parameters).get()) {
      // Evaluate the population
      std::vector<Model<TensorT>> population = { model };
      population_trainer.evaluateModels(population, std::get<EvoNetParameters::General::DataDir>(parameters).get() + std::get<EvoNetParameters::PopulationTrainer::PopulationName>(parameters).get(), 
        model_trainer, model_interpreters, model_replicator, data_simulator, model_logger, input_nodes);
    }
  }
}
#endif //EVONET_PARAMETERS_H