/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>

#include <random> // random number generator

#include <ctime> // time format
#include <chrono> // current time

namespace SmartPeak
{
  PopulationTrainer::PopulationTrainer(){};
  PopulationTrainer::~PopulationTrainer(){};

  std::vector<Model> PopulationTrainer::selectModels(
    const int& n_top,
    const int& n_random,
    const std::vector<Model>& models)
  {
    // [TODO: add method body]
  }

  std::vector<Model> PopulationTrainer::copyModels(const std::vector<Model>& models)
  {
    // [TODO: add method body]
  }

  void PopulationTrainer::modifyModels(
    std::vector<Model>& models,
    const ModelReplicator& model_replicator)
  {
    // [TODO: add method body]  
  }

  void PopulationTrainer::trainModels(std::vector<Model>& models,
    const ModelTrainer& model_trainer)
  {
    // [TODO: add method body]    
  }

  void PopulationTrainer::validateModels(std::vector<Model>& models,
    const ModelTrainer& model_trainer)
  {
    // [TODO: add method body] 
  }
}