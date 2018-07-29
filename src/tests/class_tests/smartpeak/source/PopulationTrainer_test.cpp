/**TODO:  Add copyright*/
/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PopulationTrainer test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/PopulationTrainer.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/ModelFile.h>

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

BOOST_AUTO_TEST_CASE(removeDuplicateModels) 
{
  PopulationTrainer population_trainer;

  // make a vector of models to use for testing
  std::vector<Model> models;
  for (int i=0; i<2; ++i)
  {
    for (int j=0; j<4; ++j)
    {
      Model model;
      model.setName(std::to_string(j));
			model.setId(i*j+j);
      models.push_back(model);
    }
  }

  population_trainer.removeDuplicateModels(models);
  BOOST_CHECK_EQUAL(models.size(), 4);
  for (int i=0; i<4; ++i)
    BOOST_CHECK_EQUAL(models[i].getName(), std::to_string(i));
}

BOOST_AUTO_TEST_CASE(getTopNModels_) 
{
  PopulationTrainer population_trainer;

  // make dummy data
  std::vector<std::pair<int, float>> models_validation_errors;
  const int n_models = 4;
  for (int i=0; i<n_models; ++i)
    models_validation_errors.push_back(std::make_pair(i+1, (float)(n_models-i)));

  const int n_top_models = 2;
  std::vector<std::pair<int, float>> top_n_models = population_trainer.getTopNModels_(
    models_validation_errors, n_top_models);
  
  for (int i=0; i<n_top_models; ++i)
  {
    BOOST_CHECK_EQUAL(top_n_models[i].first, n_models-i);
    BOOST_CHECK_EQUAL(top_n_models[i].second, (float)(i+1));
  }
}

BOOST_AUTO_TEST_CASE(getRandomNModels_) 
{
  PopulationTrainer population_trainer;

  // make dummy data
  std::vector<std::pair<int, float>> models_validation_errors;
  const int n_models = 4;
  for (int i=0; i<n_models; ++i)
    models_validation_errors.push_back(std::make_pair(i+1, (float)(n_models-i)));
  
  const int n_random_models = 2;
  std::vector<std::pair<int, float>> random_n_models = population_trainer.getRandomNModels_(
    models_validation_errors, n_random_models);
  
  BOOST_CHECK_EQUAL(random_n_models.size(), 2);  
  // for (int i=0; i<n_random_models; ++i)
  // {
  //   printf("model name %s error %.2f", random_n_models[i].first.data(), random_n_models[i].second);
  // }
}

// Toy ModelTrainer used for all tests
class ModelTrainerTest: public ModelTrainer
{
public:
  Model makeModel()
  {
    Model model;
    return model;
  };
  void trainModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 4>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // printf("Training the model\n");

    // Check input and output data
    if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
    {
      return;
    }
    if (!checkOutputData(getNEpochs(), output, getBatchSize(), getMemorySize(), output_nodes))
    {
      return;
    }
    if (!model.checkNodeNames(input_nodes))
    {
      return;
    }
    if (!model.checkNodeNames(output_nodes))
    {
      return;
    }
    // printf("Data checks passed\n");
    
    // Initialize the model
		model.initError(getBatchSize(), getMemorySize());
    model.clearCache();
    model.initNodes(getBatchSize(), getMemorySize());
    // printf("Initialized the model\n");

    for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
    {
      // printf("Training epoch: %d\t", iter);

      // forward propogate 
      if (iter == 0)
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, 2); 
      else      
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, 2); 

      // calculate the model error and node output error
			model.CETT(output.chip(iter,3), output_nodes, getMemorySize());
      std::cout<<"Model error: "<<model.getError().sum()<<std::endl;

      // back propogate
      // model.TBPTT(getMemorySize()-1);
      if (iter == 0)
        model.TBPTT(getMemorySize()-1, true, true, 2);
      else
        model.TBPTT(getMemorySize()-1, false, true, 2);

      // update the weights
      model.updateWeights(getMemorySize());   

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
    }
		model.clearCache();
  }
  std::vector<float> validateModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 4>& output,
    const Eigen::Tensor<float, 3>& time_steps,
    const std::vector<std::string>& input_nodes,
    const std::vector<std::string>& output_nodes)
  {
    // printf("Validating model %s\n", model.getName().data());

    std::vector<float> model_error;

    // Check input and output data
    if (!checkInputData(getNEpochs(), input, getBatchSize(), getMemorySize(), input_nodes))
    {
      return model_error;
    }
    if (!checkOutputData(getNEpochs(), output, getBatchSize(), getMemorySize(), output_nodes))
    {
      return model_error;
    }
    if (!model.checkNodeNames(input_nodes))
    {
      return model_error;
    }
    if (!model.checkNodeNames(output_nodes))
    {
      return model_error;
    }
    // printf("Data checks passed\n");
    
    // Initialize the model
		model.initError(getBatchSize(), getMemorySize());
    model.clearCache();
    model.initNodes(getBatchSize(), getMemorySize());
    // printf("Initialized the model\n");

    for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
    {
      // printf("validation epoch: %d\t", iter);

      // forward propogate
      model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2)); 

      // calculate the model error and node output error
			model.CETT(output.chip(iter, 3), output_nodes, getMemorySize());
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      model_error.push_back(total_error(0));  
      // std::cout<<"Model error: "<<total_error(0)<<std::endl;

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
			model.initError(getBatchSize(), getMemorySize());
    }
		model.clearCache();
    return model_error;
  }
};

BOOST_AUTO_TEST_CASE(validateModels_) 
{
  // PopulationTrainer population_trainer;
  
  // model_trainer_validateModels_.setBatchSize(5);
  // model_trainer_validateModels_.setMemorySize(8);
  // model_trainer_validateModels_.setNEpochs(100);

  // // make a vector of models to use for testing
  // std::vector<Model> models;
  // Eigen::Tensor<float, 1> model_error(model_trainer_validateModels_.setBatchSize(5));
  // for (int i=0; i<4; ++i)
  // {
  //   Model model;
  //   model.setName(std::to_string(i));
  //   float values = (float)(4-i);
  //   model_error.setValues({values, values, values, values, values});
  //   model.setError(model_error);
  // }

  // [TODO: complete]
}

BOOST_AUTO_TEST_CASE(selectModels) 
{
  PopulationTrainer population_trainer;
  ModelTrainerTest model_trainer;

  // [TODO: add tests]
}

BOOST_AUTO_TEST_CASE(replicateModels) 
{
  PopulationTrainer population_trainer;
  ModelTrainerTest model_trainer;

  ModelReplicator model_replicator;

  // create an initial population
  std::vector<Model> population1, population2, population3;
  for (int i=0; i<2; ++i)
  {
    // baseline model
    std::shared_ptr<WeightInitOp> weight_init;
    std::shared_ptr<SolverOp> solver;
    weight_init.reset(new ConstWeightInitOp(1.0));
    solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
    Model model = model_replicator.makeBaselineModel(
      1, 1, 1,
      NodeActivation::ELU, NodeActivation::ELU,
      weight_init, solver,
      ModelLossFunction::MSE, std::to_string(i));
    model.initWeights();
		model.initNodes(4, 4);
		model.initError(4, 4);
    
    // modify the models
    model_replicator.modifyModel(model, std::to_string(i));

		Model model1(model), model2(model), model3(model); // copy the models
    population1.push_back(model1); // push the copies to the different test populations
		population2.push_back(model2);
		population3.push_back(model3);
  }
	std::vector<std::string> input_nodes = { "Input_0" };
	std::vector<std::string> output_nodes = { "Output_0" };

	// control (no modifications)
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
  population_trainer.replicateModels(population1, model_replicator, input_nodes, output_nodes, 2);

	// check for the expected size
	BOOST_CHECK_EQUAL(population1.size(), 6);

	// control (additions only)
	model_replicator.setRandomModifications(
		std::make_pair(1, 1),
		std::make_pair(1, 1),
		std::make_pair(0, 0),
		std::make_pair(0, 0));
	population_trainer.replicateModels(population2, model_replicator, input_nodes, output_nodes, 2);

  // check for the expected size
  BOOST_CHECK_EQUAL(population2.size(), 6);

	// break the new replicates (deletions only)
	model_replicator.setRandomModifications(
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(1, 1),
		std::make_pair(1, 1));
	population_trainer.replicateModels(population3, model_replicator, input_nodes, output_nodes, 2);

	// check for the expected size
	BOOST_CHECK_EQUAL(population3.size(), 2);

  // // check for the expected tags
  // int cnt = 0;
  // for (const Model& model: population)
  // {    
  //   std::regex re("@replicateModel:");
  //   std::vector<std::string> str_tokens;
  //   std::copy(
  //     std::sregex_token_iterator(model.getName().begin(), model.getName().end(), re, -1),
  //     std::sregex_token_iterator(),
  //     std::back_inserter(str_tokens));
  //   if (cnt < 2)
  //     BOOST_CHECK_EQUAL(str_tokens.size(), 1); // original model, no tag
  //   else
  //     BOOST_CHECK_EQUAL(str_tokens.size(), 2); // replicaed moel, tag
  //   cnt += 1;
  // }


}

BOOST_AUTO_TEST_CASE(trainModels) 
{
  PopulationTrainer population_trainer;

  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(5);
  model_trainer.setMemorySize(8);
  model_trainer.setNEpochs(5);

  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // create an initial population
  std::vector<Model> population;
  for (int i=0; i<4; ++i)
  {
    // baseline model
    std::shared_ptr<WeightInitOp> weight_init;
    std::shared_ptr<SolverOp> solver;
    weight_init.reset(new ConstWeightInitOp(1.0));
    solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
    Model model = model_replicator.makeBaselineModel(
      1, 0, 1,
      NodeActivation::ReLU, NodeActivation::ReLU,
      weight_init, solver,
      ModelLossFunction::MSE, std::to_string(i));
    model.initWeights();
    
    // modify the models
    model_replicator.modifyModel(model, std::to_string(i));

    population.push_back(model);
  }

  // Break two of the models
  for (int i=0; i<2; ++i)
  {
    model_replicator.deleteLink(population[i], 1e6);
    model_replicator.deleteLink(population[i], 1e6);  
    model_replicator.deleteLink(population[i], 1e6);
  }

  // Toy data set used for all tests
  // Make the input data
  const std::vector<std::string> input_nodes = {"Input_0"}; // true inputs + biases
  Eigen::Tensor<float, 4> input_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
  Eigen::Tensor<float, 3> input_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size()); 
  input_tmp.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the output data
  const std::vector<std::string> output_nodes = {"Output_0"};
	Eigen::Tensor<float, 4> output_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
	Eigen::Tensor<float, 3> output_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size());
	output_tmp.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
	for (int batch_iter = 0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
			for (int nodes_iter = 0; nodes_iter<(int)output_nodes.size(); ++nodes_iter)
				for (int epochs_iter = 0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
					output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the simulation time_steps
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

  population_trainer.trainModels(population, model_trainer,
    input_data, output_data, time_steps, input_nodes, output_nodes);

  BOOST_CHECK_EQUAL(population.size(), 4); // broken models should still be there

  for (int i=0; i<population.size(); ++i)
  {
    if (i<2)
      BOOST_CHECK_EQUAL(population[i].getError().size(), 0); // error has not been calculated
    else
      BOOST_CHECK_EQUAL(population[i].getError().size(), model_trainer.getBatchSize()*model_trainer.getMemorySize()); // error has been calculated
  }
}

BOOST_AUTO_TEST_CASE(exampleUsage) 
{
  PopulationTrainer population_trainer;

  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(5);
  model_trainer.setMemorySize(8);
  model_trainer.setNEpochs(500);

  // Toy data set used for all tests
  // Make the input data
  const std::vector<std::string> input_nodes = {"Input_0"}; // true inputs + biases
  Eigen::Tensor<float, 4> input_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
  Eigen::Tensor<float, 3> input_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size()); 
  input_tmp.setValues(
    {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
    {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}},
    {{3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}},
    {{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}},
    {{5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}}
  );
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
    for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
      for (int nodes_iter=0; nodes_iter<(int)input_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
          input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the output data
  const std::vector<std::string> output_nodes = {"Output_0"};
  Eigen::Tensor<float, 4> output_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
  Eigen::Tensor<float, 3> output_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size());
  output_tmp.setValues(
		{ { { 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 },{ 1 } },
		{ { 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 },{ 1 } },
		{ { 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 },{ 2 } },
		{ { 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 },{ 2 } },
		{ { 6 },{ 6 },{ 5 },{ 5 },{ 4 },{ 4 },{ 3 },{ 3 } } });
  for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
			for (int nodes_iter=0; nodes_iter<(int)output_nodes.size(); ++nodes_iter)
				for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
					output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = output_tmp(batch_iter, memory_iter, nodes_iter);
  // Make the simulation time_steps
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

  // define the model replicator for growth mode
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // Evolve the population
  std::vector<Model> population; 
  const int population_size = 8;
  const int n_top = 2;
  const int n_random = 2;
  const int n_replicates_per_model = 3;
  const int iterations = 5;
  for (int iter=0; iter<iterations; ++iter)
  {
    if (iter == 0)
    {
      // define the initial population of 10 baseline models
      std::cout<<"Making the initial population..."<<std::endl;
      for (int i=0; i<population_size; ++i)
      {
        // baseline model
        std::shared_ptr<WeightInitOp> weight_init;
        std::shared_ptr<SolverOp> solver;
        weight_init.reset(new RandWeightInitOp(1.0));
        solver.reset(new AdamOp(0.1, 0.9, 0.999, 1e-8));
        Model model = model_replicator.makeBaselineModel(
          (int)input_nodes.size(), 1, (int)output_nodes.size(),
          NodeActivation::ReLU,
          NodeActivation::ReLU,
          weight_init, solver,
          ModelLossFunction::MSE, std::to_string(i));
        model.initWeights();
        
        // modify the models
        model_replicator.modifyModel(model, std::to_string(i));

        population.push_back(model);
      }
    }

		model_replicator.setRandomModifications(
			std::make_pair(0, 0),
			std::make_pair(1, 1),
			std::make_pair(0, 0),
			std::make_pair(1, 1));

    // train the population
    std::cout<<"Training the population..."<<std::endl;
    population_trainer.trainModels(population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes, 2);

    // select the top N from the population
    std::cout<<"Select the top N models from the population..."<<std::endl;
		std::vector<std::pair<int, float>> models_validation_errors = population_trainer.selectModels(
      n_top, n_random, population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes, 2);

    //for (const Model& model: population)
    //{
    //  const Eigen::Tensor<float, 0> total_error = model.getError().sum();
    //  printf("Model %s (Nodes: %d, Links: %d) error: %.2f\n", model.getName().data(), model.getNodes().size(), model.getLinks().size(), total_error.data()[0]);
    //  for (auto link: model.getLinks())
    //    printf("Links %s\n", link.getName().data());
    //}

    if (iter < iterations - 1)  
    {
      // replicate and modify models
      std::cout<<"Replicate and modify the top N models from the population..."<<std::endl; 
      population_trainer.replicateModels(population, model_replicator, input_nodes, output_nodes, n_replicates_per_model, std::to_string(iter), 2);
    }
    else
    {
			models_validation_errors = population_trainer.selectModels(
        1, 1, population, model_trainer,
        input_data, output_data, time_steps, input_nodes, output_nodes, 2);
    }
  }

  // write the model to file
  WeightFile weightfile;
  weightfile.storeWeightsCsv("populationTrainerWeights.csv", population[0].getWeights());
  LinkFile linkfile;
  linkfile.storeLinksCsv("populationTrainerLinks.csv", population[0].getLinks());
  NodeFile nodefile;
  nodefile.storeNodesCsv("populationTrainerNodes.csv", population[0].getNodes());
	ModelFile modelfile;
	modelfile.storeModelDot("populationTrainerGraph.gv", population[0]);

  // [TODO: check that one of the models has a 0.0 error
  //        i.e., correct structure and weights]
}


BOOST_AUTO_TEST_SUITE_END()