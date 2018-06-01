/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PopulationTrainer test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/PopulationTrainer.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <ctime> // time format
#include <chrono> // current time

#include <set> // std::map sort
#include <algorithm> // std::map sort
#include <functional> // std::map sort

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(populationTrainer)

BOOST_AUTO_TEST_CASE(constructor) 
{
  PopulationTrainer* ptr = nullptr;
  PopulationTrainer* nullPointer = nullptr;
	ptr = new PopulationTrainer();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  PopulationTrainer* ptr = nullptr;
	ptr = new PopulationTrainer();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(DELETEAfterTesting) 
{
  PopulationTrainer population_trainer;

  // define the model trainer
  class ModelTrainerTest: public ModelTrainer
  {
  public:
    Model makeModel(){};
    void trainModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes)
    {
      printf("Training the model\n");

      // Check input and output data
      if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
      {
        return;
      }
      if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
      {
        return;
      }
      printf("Data checks passed\n");
      
      // Initialize the model
      model.initNodes(getBatchSize(), getMemorySize());
      model.initWeights();
      printf("Initialized the model\n");

      for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
      {
        printf("Training epoch: %d\t", iter);

        // forward propogate
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2)); 

        // calculate the model error and node output error
        model.calculateError(output.chip(iter, 2), output_nodes);
        std::cout<<"Model error: "<<model.getError().sum()<<std::endl;

        // back propogate
        model.TBPTT(getMemorySize()-1);

        // update the weights
        model.updateWeights(getMemorySize());   

        // reinitialize the model
        model.reInitializeNodeStatuses();
        model.initNodes(getBatchSize(), getMemorySize());
      }
    }
    std::vector<float> validateModel(Model& model,
      const Eigen::Tensor<float, 4>& input,
      const Eigen::Tensor<float, 3>& output,
      const Eigen::Tensor<float, 3>& time_steps,
      const std::vector<std::string>& input_nodes,
      const std::vector<std::string>& output_nodes)
    {
      printf("Validating the model\n");

      std::vector<float> model_error;

      // Check input and output data
      if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
      {
        return model_error;
      }
      if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
      {
        return model_error;
      }
      printf("Data checks passed\n");
      
      // Initialize the model
      model.initNodes(getBatchSize(), getMemorySize());
      model.initWeights();
      printf("Initialized the model\n");

      for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
      {
        printf("validation epoch: %d\t", iter);

        // forward propogate
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2)); 

        // calculate the model error and node output error
        model.calculateError(output.chip(iter, 2), output_nodes); 
        const Eigen::Tensor<float, 0> total_error = model.getError().sum();
        model_error.push_back(total_error(0));  
        std::cout<<"Model error: "<<total_error(0)<<std::endl;

        // reinitialize the model
        model.reInitializeNodeStatuses();
        model.initNodes(getBatchSize(), getMemorySize());
      }

      return model_error;
    }
  };
  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(5);
  model_trainer.setMemorySize(8);
  model_trainer.setNEpochs(20);

  // define the model replicator for growth mode
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // define the initial population of 10 baseline models
  std::cout<<"Making the initial population..."<<std::endl;
  std::vector<Model> population; 
  std::shared_ptr<WeightInitOp> weight_init;
  std::shared_ptr<SolverOp> solver;
  for (int i=0; i<10; ++i)
  {
    // baseline model
    std::cout<<"Making the baseline model "<<i<<"..."<<std::endl;
    weight_init.reset(new RandWeightInitOp(1.0));
    solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
    Model model = model_replicator.makeBaselineModel(
      1, 0, 1,
      NodeActivation::ReLU, NodeActivation::ReLU,
      weight_init, solver,
      ModelLossFunction::MSE, std::to_string(i));
    
    // modify the models
    std::cout<<"Modifying the baseline model "<<i<<"..."<<std::endl;
    model_replicator.modifyModel(model);

    population.push_back(model);
  }

  // Make the input data
  std::cout<<"Making the input data..."<<std::endl;
  const std::vector<std::string> input_nodes = {"Input_0"}; // true inputs + biases
  Eigen::Tensor<float, 4> input_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), input_nodes.size(), model_trainer.getNEpochs());
  Eigen::Tensor<float, 3> input_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), input_nodes.size()); 
  input_tmp.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  
  // Make the output data
  std::cout<<"Making the output data..."<<std::endl;
  const std::vector<std::string> output_nodes = {"Output_0"};
  Eigen::Tensor<float, 3> output_data(model_trainer.getBatchSize(), output_nodes.size(), model_trainer.getNEpochs());
  Eigen::Tensor<float, 2> output_tmp(model_trainer.getBatchSize(), output_nodes.size()); 
  output_tmp.setValues({{2.5}, {3}, {3.5}, {4}, {4.5}});
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
    for (int nodes_iter=0; nodes_iter<output_nodes.size(); ++nodes_iter)
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
        output_data(batch_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, nodes_iter);

  // Make the simulation time_steps
  std::cout<<"Making the time steps..."<<std::endl;
  Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochs());
  Eigen::Tensor<float, 2> time_steps_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize()); 
  time_steps_tmp.setValues({
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}}
  );
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
        time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

  // train the population
  std::cout<<"Training the population..."<<std::endl;
  for (int i=0; i<population.size(); ++i)
  {
    std::cout<<"Training the model "<<i<<"..."<<std::endl;
    try
    {
      model_trainer.trainModel(
        population[i], input_data, output_data, time_steps,
        input_nodes, output_nodes);
    }
    catch (std::exception& e)
    {
      printf("The model %s is broken.\n", population[i].getName().data());
      // need to remove the model somehow...
    }

    const Eigen::Tensor<float, 0> total_error = population[i].getError().sum();
    std::cout<<"Total error for Model "<<population[i].getName()<<" is "<<total_error(0)<<std::endl;
  }

  // select the top N from the population
  // NOTES: will need to deal with cases where there are less models in the population than N
  std::cout<<"Select the top N models from the population..."<<std::endl;

  // score each model on the validation data
  std::map<std::string, float> population_errors_map;
  for (int i=0; i<population.size(); ++i)
  {
    std::cout<<"Validating the model "<<i<<"..."<<std::endl;    
    try
    {
      std::vector<float> model_errors = model_trainer.validateModel(
        population[i], input_data, output_data, time_steps,
        input_nodes, output_nodes);
      float model_ave_error = accumulate(model_errors.begin(), model_errors.end(), 0.0)/model_errors.size();
      population_errors_map.emplace(population[i].getName(), model_ave_error);
    }
    catch (std::exception& e)
    {
      printf("The model %s is broken.\n", population[i].getName().data());
      population_errors_map.emplace(population[i].getName(), 1e6f);
    }
  }

  // sort each model based on their scores in ascending order
  std::vector<std::pair<std::string, float>> pairs;
  for (auto itr = population_errors_map.begin(); itr != population_errors_map.end(); ++itr)
      pairs.push_back(*itr);

  std::sort(
    pairs.begin(), pairs.end(), 
    [=](std::pair<std::string, float>& a, std::pair<std::string, float>& b)
    {
      return a.second < b.second;
    }
  );

  // select the top N from the population
  int n_top = 2;  // move into function arguments
  std::vector<std::string> top_n_model_names;
  for (int i=0; i<n_top; ++i) {top_n_model_names.push_back(pairs[i].first);}
  std::vector<Model> top_n_models;
  for (int i=0; i<n_top; ++i)
    for (int j=0; j<population.size(); ++j)
      if (population[j].getName() == top_n_model_names[i])
        top_n_models.push_back(population[j]);

  // replicate and modify
  int n_replicates_per_model = 4;
  std::vector<Model> population_new;
  for (const auto& model: top_n_models)
  {
    for (int i=0; i<n_replicates_per_model; ++i)
    {
      Model model_copy = model;
      model_replicator.modifyModel(model_copy);
      population_new.push_back(model_copy);
    }

    population_new.push_back(model); // persist the original model
  }
}

BOOST_AUTO_TEST_CASE(selectModels) 
{
  PopulationTrainer population_trainer;

  // [TODO: add tests]
}

BOOST_AUTO_TEST_CASE(copyModels) 
{
  PopulationTrainer population_trainer;

  // [TODO: add tests]
}

BOOST_AUTO_TEST_CASE(modifyModels) 
{
  PopulationTrainer population_trainer;

  // [TODO: add tests]
}

BOOST_AUTO_TEST_CASE(trainModels) 
{
  PopulationTrainer population_trainer;

  // [TODO: add tests]
}

BOOST_AUTO_TEST_CASE(validateModels) 
{
  PopulationTrainer population_trainer;

  // [TODO: add tests]
}


BOOST_AUTO_TEST_SUITE_END()