/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h> 
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>

#include <random>
#include <fstream>
#include <thread>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/*
  @brief implementation of the add problem that
    has been used to test sequence prediction in 
    RNNS

  References:
    [TODO]

  @input[in] sequence_length
  @input[in, out] random_sequence
  @input[in, out] mask_sequence

  @returns the result of the two random numbers in the sequence
**/
static float AddProb(
  Eigen::Tensor<float, 1>& random_sequence,
  Eigen::Tensor<float, 1>& mask_sequence)
{
  float result = 0.0;
  const int sequence_length = random_sequence.size();
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> zero_to_one(0, 1);
  std::uniform_int_distribution<> zero_to_length(0, sequence_length-1);

  // generate 2 random and unique indexes between 
  // [0, sequence_length) for the mask
  int mask_index_1 = zero_to_length(gen);
  int mask_index_2 = 0;
  do {
    mask_index_2 = zero_to_length(gen);
  } while (mask_index_1 == mask_index_2);

  // generate the random sequence
  // and the mask sequence
  for (int i=0; i<sequence_length; ++i)
  {
    // the random sequence
    random_sequence[i] = zero_to_one(gen);
    // the mask
    if (i == mask_index_1 || i == mask_index_2)
      mask_sequence[i] = 1.0;
    else
      mask_sequence[i] = 0.0;

    // result update
    result += mask_sequence[i] * random_sequence[i];
  }

  return result;
};

// ModelTrainer used for all tests
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
    const Eigen::Tensor<float, 3>& output,
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
    if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
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
    const int n_threads = 2;
    model.clearCache();
    model.initNodes(getBatchSize(), getMemorySize());
    // printf("Initialized the model\n");

    for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
    {
      // printf("Training epoch: %d\t", iter);

      // forward propogate
      if (iter == 0)
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, n_threads); 
      else      
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, n_threads); 

      // calculate the model error and node output error
      model.calculateError(output.chip(iter, 2), output_nodes);
      // std::cout<<"Model "<<model.getName()<<" error: "<<model.getError().sum()<<std::endl;

      // // Print some details that are useful for debugging
      // printf("Node ID:\t");
      // for (const std::string& node : output_nodes)
      //   printf("%s\t", node.data());
      // printf("\nExpected:\n");
      // const Eigen::Tensor<float, 2> output_chip = output.chip(iter, 2);
      // for (int j=0; j<getBatchSize(); ++j) 
      // {
      //   printf("Batch %d:\t", j);
      //   for (int i=0; i<output_nodes.size(); ++i)
      //     printf("%0.2f\t", (float)output_chip(j, i));
      //   printf("\n");
      // }
      // printf("\nCalculated:\n");
      // for (int j=0; j<getBatchSize(); ++j) 
      // {
      //   printf("Batch %d:\t", j);
      //   for (int i=0; i<output_nodes.size(); ++i)
      //     printf("%0.2f\t", model.getNode(output_nodes[i]).getOutput()(j,0));
      //   printf("\n");
      // }
      // printf("\n"); 

      // back propogate
      if (iter == 0)
        model.TBPTT(getMemorySize()-1, true, true, n_threads);
      else
        model.TBPTT(getMemorySize()-1, false, true, n_threads);

      // update the weights
      model.updateWeights(getMemorySize());   

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
    }    
    model.clearCache();
  }
  std::vector<float> validateModel(Model& model,
    const Eigen::Tensor<float, 4>& input,
    const Eigen::Tensor<float, 3>& output,
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
    if (!checkOutputData(getNEpochs(), output, getBatchSize(), output_nodes))
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
    const int n_threads = 2;
    model.clearCache();
    model.initNodes(getBatchSize(), getMemorySize());
    // printf("Initialized the model\n");

    for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
    {
      // printf("validation epoch: %d\t", iter);

      // forward propogate
      if (iter == 0)
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), true, true, n_threads); 
      else      
        model.FPTT(getMemorySize(), input.chip(iter, 3), input_nodes, time_steps.chip(iter, 2), false, true, n_threads);

      // calculate the model error and node output error
      model.calculateError(output.chip(iter, 2), output_nodes); 
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      model_error.push_back(total_error(0));  
      // std::cout<<"Model error: "<<total_error(0)<<std::endl;

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
    }
    model.clearCache();
    return model_error;
  }
};

// Main
int main(int argc, char** argv)
{
  PopulationTrainer population_trainer;

  // Add problem parameters
  const std::size_t input_size = 2;  // random number from the sequence and 0 or 1 from the mask
  const std::size_t output_size = 1;  // result of random number addition
  const int sequence_length = 2; // test sequence length
  const std::size_t training_data_size = 100000; //60000;
  const std::size_t validation_data_size = 10000; //10000;

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = n_hard_threads/2; // the number of threads
  char threads_cout[512];
  sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
    n_hard_threads, 2);
  std::cout<<threads_cout;

  // Make the input nodes 
  // [TODO: refactor into a convenience function]
  std::vector<std::string> input_nodes;
  for (int i=0; i<input_size; ++i)
    input_nodes.push_back("Input_" + std::to_string(i));

  // Make the output nodes
  // [TODO: refactor into a convenience function]
  std::vector<std::string> output_nodes;
  for (int i=0; i<output_size; ++i)
    output_nodes.push_back("Output_" + std::to_string(i));

  // define the model replicator for growth mode
  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(4);
  model_trainer.setMemorySize(sequence_length);
  model_trainer.setNEpochs(100);

  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochs());
  time_steps.setConstant(1.0f);

  // define the model replicator for growth mode
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // Population initial conditions
  const int population_size = 1;
  int n_top = 1;
  int n_random = 1;
  int n_replicates_per_model = 0;
  
  // random generator for model modifications
  std::random_device rd;
  std::mt19937 gen(rd());

  // Evolve the population
  std::vector<Model> population; 
  const int iterations = 100;
  for (int iter=0; iter<iterations; ++iter)
  {
    printf("Iteration #: %d\n", iter);

    if (iter == 0)
    {
      std::cout<<"Initializing the population..."<<std::endl;  
      // define the initial population [BUG FREE]
      for (int i=0; i<population_size; ++i)
      {
        // baseline model
        std::shared_ptr<WeightInitOp> weight_init;
        std::shared_ptr<SolverOp> solver;
        weight_init.reset(new RandWeightInitOp(input_nodes.size()));
        solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
        Model model = model_replicator.makeBaselineModel(
          input_nodes.size(), 2, output_nodes.size(),
          NodeActivation::TanH,
          NodeActivation::TanH,
          weight_init, solver,
          ModelLossFunction::MSE, std::to_string(i));
        model.initWeights();
        
        // modify the models
        model_replicator.modifyModel(model, std::to_string(i));

        char cout_char[512];
        sprintf(cout_char, "Model %s (Nodes: %d, Links: %d)\n", model.getName().data(), model.getNodes().size(), model.getLinks().size());
        std::cout<<cout_char;
        population.push_back(model);
      }
    }
  
    // Generate the input and output data for training [BUG FREE]
    std::cout<<"Generating the input/output data for training..."<<std::endl;  
    Eigen::Tensor<float, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
    Eigen::Tensor<float, 3> output_data_training(model_trainer.getBatchSize(), (int)output_nodes.size(), model_trainer.getNEpochs());  
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter) {
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter) {

        // generate a new sequence
        Eigen::Tensor<float, 1> random_sequence(sequence_length);
        Eigen::Tensor<float, 1> mask_sequence(sequence_length);
        float result = AddProb(random_sequence, mask_sequence);

        // assign the output
        output_data_training(batch_iter, 0, epochs_iter) = result;
        
        // assign the input sequences
        for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter) {
          input_data_training(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_iter); // random sequence
          input_data_training(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_iter); // mask sequence
        }
      }
    }

    // model modification scheduling
    // if (iter > 100)
    // {
    //   model_replicator.setNNodeAdditions(1);
    //   model_replicator.setNLinkAdditions(2);
    //   model_replicator.setNNodeDeletions(1);
    //   model_replicator.setNLinkDeletions(2);
    // }
    // else if (iter > 5 && iter < 100)
    // {
    //   model_replicator.setNNodeAdditions(2);
    //   model_replicator.setNLinkAdditions(4);
    //   model_replicator.setNNodeDeletions(1);
    //   model_replicator.setNLinkDeletions(2);
    // }
    // else if (iter >= 0)
    // {      
    //   model_replicator.setNNodeAdditions(1);
    //   model_replicator.setNLinkAdditions(2);
    //   model_replicator.setNNodeDeletions(0);
    //   model_replicator.setNLinkDeletions(0);
    // }

    // generate a random number of model modifications
    if (iter>0)
    {
      std::uniform_int_distribution<> zero_to_one(0, 1);
      std::uniform_int_distribution<> zero_to_two(0, 2);
      model_replicator.setNNodeAdditions(zero_to_one(gen));
      model_replicator.setNLinkAdditions(zero_to_two(gen));
      model_replicator.setNNodeDeletions(zero_to_one(gen));
      model_replicator.setNLinkDeletions(zero_to_two(gen));
    }

    // train the population
    std::cout<<"Training the models..."<<std::endl;
    population_trainer.trainModels(population, model_trainer,
      input_data_training, output_data_training, time_steps, input_nodes, output_nodes, n_threads);

    // generate the input/output data for validation
    std::cout<<"Generating the input/output data for validation..."<<std::endl;      
    model_trainer.setNEpochs(20);  // lower the number of epochs for validation

    Eigen::Tensor<float, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
    Eigen::Tensor<float, 3> output_data_validation(model_trainer.getBatchSize(), (int)output_nodes.size(), model_trainer.getNEpochs());  
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter) {
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter) {

        // generate a new sequence
        Eigen::Tensor<float, 1> random_sequence(sequence_length);
        Eigen::Tensor<float, 1> mask_sequence(sequence_length);
        float result = AddProb(random_sequence, mask_sequence);

        // assign the output
        output_data_validation(batch_iter, 0, epochs_iter) = result;
        
        // assign the input sequences
        for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter) {
          input_data_validation(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_iter); // random sequence
          input_data_validation(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_iter); // mask sequence
        }
      }
    }

    // select the top N from the population
    std::cout<<"Selecting the models..."<<std::endl;    
    population_trainer.selectModels(
      n_top, n_random, population, model_trainer,
      input_data_validation, output_data_validation, time_steps, input_nodes, output_nodes, n_threads);

    model_trainer.setNEpochs(100);  // restore the number of epochs for training

    // for (const Model& model: population)
    // {
    //   const Eigen::Tensor<float, 0> total_error = model.getError().sum();
    //   char cout_char[512];
    //   sprintf(cout_char, "Model %s (Nodes: %d, Links: %d) error: %.2f\n", model.getName().data(), model.getNodes().size(), model.getLinks().size(), total_error.data()[0]);
    //   std::cout<<cout_char;
    //   // for (auto link: model.getLinks())
    //   //   printf("Links %s\n", link.getName().data());
    // }

    if (iter < iterations - 1)  
    {
      // Population size of 8
      if (iter == 0)
      {
        n_top = 2;
        n_random = 2;
        n_replicates_per_model = 7;
      }
      else
      {
        n_top = 2;
        n_random = 2;
        n_replicates_per_model = 3;
      }

      // // Binary selection with a total population size of 2
      // if (iter == 0)
      // {
      //   n_top = 1;
      //   n_random = 1;
      //   n_replicates_per_model = 1;
      // } n_threads

      // replicate and modify models
      std::cout<<"Replicating and modifying the models..."<<std::endl;
      population_trainer.replicateModels(population, model_replicator, n_replicates_per_model, std::to_string(iter), n_threads);
      std::cout<<"Population size of "<<population.size()<<std::endl;
    }
  }


  // write the model to file
  WeightFile weightfile;
  weightfile.storeWeightsCsv("AddProbExampleWeights.csv", population[0].getWeights());
  LinkFile linkfile;
  linkfile.storeLinksCsv("AddProbExampleLinks.csv", population[0].getLinks());
  NodeFile nodefile;
  nodefile.storeNodesCsv("AddProbExampleNodes.csv", population[0].getNodes());
  
  return 0;
}