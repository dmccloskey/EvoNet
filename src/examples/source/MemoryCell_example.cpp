/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h>
#include <SmartPeak/io/PopulationTrainerFile.h>

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
  std::uniform_real_distribution<> zero_to_one(-0.5, 0.5);
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
    random_sequence(i) = zero_to_one(gen);
    // the mask
    if (i == mask_index_1 || i == mask_index_2)
      mask_sequence(i) = 1.0;
    else
      mask_sequence(i) = 0.0;

    // result update
    result += mask_sequence(i) * random_sequence(i);
  }

  return result;
};

// ModelTrainer used for all testsLink_m_to_o
class ModelTrainerTest: public ModelTrainer
{
public:
	Model makeModel()
	{
		Node i_rand, i_mask, h, m, o,
			h_bias, m_bias, o_bias;
		Link Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_m,
			Link_h_bias_to_h,
			Link_m_bias_to_m, Link_o_bias_to_o;
		Weight Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_m,
			Weight_h_bias_to_h,
			Weight_m_bias_to_m, Weight_o_bias_to_o;
		Model model;
		// Nodes
		i_rand = Node("i_rand", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
		i_mask = Node("i_mask", NodeType::input, NodeStatus::activated, NodeActivation::Linear);
		//h = Node("h", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
		//m = Node("m", NodeType::hidden, NodeStatus::deactivated, NodeActivation::ReLU);
		//o = Node("o", NodeType::output, NodeStatus::deactivated, NodeActivation::ReLU);
		h = Node("h", NodeType::hidden, NodeStatus::deactivated, NodeActivation::TanH);  // works well in range 0-1
		m = Node("m", NodeType::hidden, NodeStatus::deactivated, NodeActivation::TanH);
		o = Node("o", NodeType::output, NodeStatus::deactivated, NodeActivation::TanH);
		h_bias = Node("h_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
		m_bias = Node("m_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
		o_bias = Node("o_bias", NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
		// weights  
		std::shared_ptr<WeightInitOp> weight_init;
		std::shared_ptr<SolverOp> solver;
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_rand_to_h = Weight("Weight_i_rand_to_h", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(100.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_i_mask_to_h = Weight("Weight_i_mask_to_h", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_h_to_m = Weight("Weight_h_to_m", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_to_m = Weight("Weight_m_to_m", weight_init, solver);
		weight_init.reset(new RandWeightInitOp(2.0));
		//weight_init.reset(new ConstWeightInitOp(1.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_to_o = Weight("Weight_m_to_o", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		//weight_init.reset(new ConstWeightInitOp(-100.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_h_bias_to_h = Weight("Weight_h_bias_to_h", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		//weight_init.reset(new ConstWeightInitOp(0.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_m_bias_to_m = Weight("Weight_m_bias_to_m", weight_init, solver);
		weight_init.reset(new ConstWeightInitOp(1.0));
		//weight_init.reset(new ConstWeightInitOp(0.0)); //solution
		//solver.reset(new SGDOp(0.01, 0.9));
		solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
		solver->setGradientThreshold(1000.0f);
		Weight_o_bias_to_o = Weight("Weight_o_bias_to_o", weight_init, solver);
		weight_init.reset();
		solver.reset();
		// links
		Link_i_rand_to_h = Link("Link_i_rand_to_h", "i_rand", "h", "Weight_i_rand_to_h");
		Link_i_mask_to_h = Link("Link_i_mask_to_h", "i_mask", "h", "Weight_i_mask_to_h");
		Link_h_to_m = Link("Link_h_to_m", "h", "m", "Weight_h_to_m");
		Link_m_to_o = Link("Link_m_to_o", "m", "o", "Weight_m_to_o");
		Link_m_to_m = Link("Link_m_to_m", "m", "m", "Weight_m_to_m");
		Link_h_bias_to_h = Link("Link_h_bias_to_h", "h_bias", "h", "Weight_h_bias_to_h");
		Link_m_bias_to_m = Link("Link_m_bias_to_m", "m_bias", "m", "Weight_m_bias_to_m");
		Link_o_bias_to_o = Link("Link_o_bias_to_o", "o_bias", "o", "Weight_o_bias_to_o");
		// add nodes, links, and weights to the model
		model.setName("MemoryCell");
		model.addNodes({ i_rand, i_mask, h, m, o,
			h_bias, m_bias, o_bias });
		model.addWeights({ Weight_i_rand_to_h, Weight_i_mask_to_h,
			Weight_h_to_m,
			Weight_m_to_o, Weight_m_to_m,
			Weight_h_bias_to_h,
			Weight_m_bias_to_m, Weight_o_bias_to_o });
		model.addLinks({ Link_i_rand_to_h, Link_i_mask_to_h,
			Link_h_to_m,
			Link_m_to_o, Link_m_to_m,
			Link_h_bias_to_h,
			Link_m_bias_to_m, Link_o_bias_to_o });
		model.setLossFunction(ModelLossFunction::MSE);
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
    const int n_threads = 2;
		model.initError(getBatchSize(), getMemorySize());
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
			model.CETT(output.chip(iter, 3), output_nodes, 1);  // just the last result
      //model.CETT(output.chip(iter, 3), output_nodes, getMemorySize());

      //std::cout<<"Model "<<model.getName()<<" error: "<<model.getError().sum()<<std::endl;

      // back propogate
      if (iter == 0)
        model.TBPTT(getMemorySize()-1, true, true, n_threads);
      else
        model.TBPTT(getMemorySize()-1, false, true, n_threads);

			//for (const Node& node : model.getNodes())
			//{
			//	std::cout << node.getName() << " Output: " << node.getOutput() << std::endl;
			//	std::cout << node.getName() << " Error: " << node.getError() << std::endl;
			//}
			//for (const Weight& weight : model.getWeights())
			//	std::cout << weight.getName() << " Weight: " << weight.getWeight() << std::endl;

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
    const int n_threads = 2;
		model.initError(getBatchSize(), getMemorySize());
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
			model.CETT(output.chip(iter, 3), output_nodes, 1); // just the last predicted result
			//model.CETT(output.chip(iter, 3), output_nodes, getMemorySize()); // just the last predicted result
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      model_error.push_back(total_error(0));  
      //std::cout<<"Model error: "<<total_error(0)<<std::endl;

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
  const int sequence_length = 10; // test sequence length
	const int n_epochs = 500;
	const int n_epochs_validation = 10;

  const int n_hard_threads = std::thread::hardware_concurrency();
  const int n_threads = n_hard_threads/2; // the number of threads
  char threads_cout[512];
  sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
    n_hard_threads, 2);
  std::cout<<threads_cout;

  // Make the input nodes 
	//std::vector<std::string> input_nodes = {"i_rand", "i_mask", "h_bias", "m_bias", "o_bias" };
	std::vector<std::string> input_nodes = { "i_rand", "i_mask"};

  // Make the output nodes
	std::vector<std::string> output_nodes = {"o"};

  // define the model replicator for growth mode
  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(8);
  model_trainer.setMemorySize(sequence_length);
  model_trainer.setNEpochs(n_epochs);

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

  // Evolve the population
  std::vector<Model> population; 
  const int iterations = 12;
  for (int iter=0; iter<iterations; ++iter)
  {
    printf("Iteration #: %d\n", iter);

    if (iter == 0)
    {
      std::cout<<"Initializing the population..."<<std::endl;  
      // define the initial population [BUG FREE]
      for (int i=0; i<population_size; ++i)
      {
        // make the model name
        Model model = model_trainer.makeModel();
				model.initWeights(); // initialize the weights

        char model_name_char[512];
        sprintf(model_name_char, "%s_%d", model.getName().data(), i);
        std::string model_name(model_name_char);
				model.setName(model_name);
        population.push_back(model);
      }
    }
  
    // Generate the input and output data for training [BUG FREE]
    std::cout<<"Generating the input/output data for training..."<<std::endl;  
    Eigen::Tensor<float, 4> input_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
    Eigen::Tensor<float, 4> output_data_training(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter) {
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter) {

        // generate a new sequence
        Eigen::Tensor<float, 1> random_sequence(sequence_length);
        Eigen::Tensor<float, 1> mask_sequence(sequence_length);
        float result = AddProb(random_sequence, mask_sequence);

				float result_cumulative = 0.0;
        
        for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter) {
					// assign the input sequences
          input_data_training(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_iter); // random sequence
          input_data_training(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_iter); // mask sequence
					//input_data_training(batch_iter, memory_iter, 2, epochs_iter) = 0.0f; // h bias
					//input_data_training(batch_iter, memory_iter, 3, epochs_iter) = 0.0f; // m bias
					//input_data_training(batch_iter, memory_iter, 4, epochs_iter) = 0.0f; // 0 bias

					// assign the output
					//result_cumulative += random_sequence(memory_iter) * mask_sequence(memory_iter);
					//output_data_training(batch_iter, memory_iter, 0, epochs_iter) = result_cumulative;
					if (memory_iter == 0)
						output_data_training(batch_iter, memory_iter, 0, epochs_iter) = result;
					else
						output_data_training(batch_iter, memory_iter, 0, epochs_iter) = 0.0;
        }
      }
    }

    // generate a random number of model modifications
    if (iter>0)
    {
			model_replicator.setRandomModifications(
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2));
    }

    // train the population
    std::cout<<"Training the models..."<<std::endl;
    population_trainer.trainModels(population, model_trainer,
      input_data_training, output_data_training, time_steps, input_nodes, output_nodes, n_threads);

    // generate the input/output data for validation
    std::cout<<"Generating the input/output data for validation..."<<std::endl;      
    model_trainer.setNEpochs(n_epochs_validation);  // lower the number of epochs for validation

    Eigen::Tensor<float, 4> input_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
    Eigen::Tensor<float, 4> output_data_validation(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter) {
      for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter) {

        // generate a new sequence
        Eigen::Tensor<float, 1> random_sequence(sequence_length);
        Eigen::Tensor<float, 1> mask_sequence(sequence_length);
        float result = AddProb(random_sequence, mask_sequence);    

				float result_cumulative = 0.0;
        
        for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter) {
					// assign the input sequences
          input_data_validation(batch_iter, memory_iter, 0, epochs_iter) = random_sequence(memory_iter); // random sequence
          input_data_validation(batch_iter, memory_iter, 1, epochs_iter) = mask_sequence(memory_iter); // mask sequence
					//input_data_validation(batch_iter, memory_iter, 2, epochs_iter) = 0.0f; // h bias
					//input_data_validation(batch_iter, memory_iter, 3, epochs_iter) = 0.0f; // m bias
					//input_data_validation(batch_iter, memory_iter, 4, epochs_iter) = 0.0f; // o bias

					// assign the output
					//result_cumulative += random_sequence(memory_iter) * mask_sequence(memory_iter);
					//output_data_validation(batch_iter, memory_iter, 0, epochs_iter) = result_cumulative;
					if (memory_iter == 0)
						output_data_validation(batch_iter, memory_iter, 0, epochs_iter) = result;
					else
						output_data_validation(batch_iter, memory_iter, 0, epochs_iter) = 0.0;
        }
      }
    }

    // select the top N from the population
    std::cout<<"Selecting the models..."<<std::endl;    
		std::vector<std::pair<std::string, float>> models_validation_errors = population_trainer.selectModels(
      n_top, n_random, population, model_trainer,
      input_data_validation, output_data_validation, time_steps, input_nodes, output_nodes, n_threads);

    model_trainer.setNEpochs(n_epochs);  // restore the number of epochs for training

    if (iter < iterations - 1)  
    {
			// Population size of 16
			if (iter == 0)
			{
				n_top = 3;
				n_random = 3;
				n_replicates_per_model = 15;
			}
			else
			{
				n_top = 3;
				n_random = 3;
				n_replicates_per_model = 3;
			}
      // replicate and modify models
      std::cout<<"Replicating and modifying the models..."<<std::endl;
      population_trainer.replicateModels(population, model_replicator, n_replicates_per_model, std::to_string(iter), n_threads);
      std::cout<<"Population size of "<<population.size()<<std::endl;
    }
		else
		{
			PopulationTrainerFile population_trainer_file;
			population_trainer_file.storeModels(population, "MemoryCell");
			population_trainer_file.storeModelValidations("MemoryCellValidationErrors.csv", models_validation_errors);
		}
  }

  system("pause");
  return 0;
}