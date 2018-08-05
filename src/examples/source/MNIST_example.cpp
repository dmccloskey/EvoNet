/**TODO:  Add copyright*/

#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h> 
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/NodeFile.h>

#include <SmartPeak/core/Preprocessing.h>

#include <fstream>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

/**
 * EXAMPLES using the MNIST data set
 * 
 * EXAMPLE1:
 * - classification on MNIST using DAG
 * - whole image pixels (linearized) 28x28 normalized to 0 to 1
 * - classifier (1 hot vector from 0 to 9)
 * 
 * EXAMPLE2:
 * - classification on MNIST using DCG
 * - scan of pixel 8x8 pixel subset over time
 * - classifier (1 hot vector from 0 to 9)
 * 
 * ISSUES:
 * 1. problem: Forward propogation and backward propogation are slow
 *    fix: need to implement GPU device in tensor library
 *    steps: 1) install CUDA toolkit, 2) modify cmake to build with nvcc, 3) modify code to use GpuDevice
 */

// Toy ModelTrainer used for all tests
class ModelTrainerTest: public ModelTrainer
{
};

int ReverseInt(int i)
{
  unsigned char ch1, ch2, ch3, ch4;
  ch1=i&255;
  ch2=(i>>8)&255;
  ch3=(i>>16)&255;
  ch4=(i>>24)&255;
  return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

template<typename T>
void ReadMNIST(const std::string& filename, Eigen::Tensor<T, 2>& data, const bool& is_labels)
{
  // dims: sample, pixel intensity or sample, label
  // e.g., pixel data dims: 1000 x (28x28)
  // e.g., label data dims: 1000 x 1

  // open up the file
  std::ifstream file (filename, std::ios::binary);
  if (file.is_open())
  {
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;

    // get the magic number
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= ReverseInt(magic_number);

    // get the number of images
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= ReverseInt(number_of_images);
    if (number_of_images > data.dimension(0))
      number_of_images = data.dimension(0);

    // get the number of rows and cols
    if (!is_labels)
    {
      file.read((char*)&n_rows,sizeof(n_rows));
      n_rows= ReverseInt(n_rows);
      file.read((char*)&n_cols,sizeof(n_cols));
      n_cols= ReverseInt(n_cols);
    }
    else
    {
      n_rows=1;
      n_cols=1;
    }

    // get the actual data
    for(int i=0;i<number_of_images;++i)
    {
      for(int r=0;r<n_rows;++r)
      {
        for(int c=0;c<n_cols;++c)
        {
          unsigned char temp=0;
          file.read((char*)&temp,sizeof(temp));
          data(i, (n_rows*r)+c) = (T)temp;
        }
      }
    }
  }
}

int main(int argc, char** argv)
{

  PopulationTrainer population_trainer;

  const std::size_t input_size = 784;
  const std::size_t training_data_size = 1000; //60000;
  const std::size_t validation_data_size = 100; //10000;
  const std::vector<float> mnist_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int n_threads = 8;

  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i=0; i<input_size; ++i)
    input_nodes.push_back("Input_" + std::to_string(i));

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i=0; i<mnist_labels.size(); ++i)
    output_nodes.push_back("Output_" + std::to_string(i));

  // Read input images [BUG FREE]
  std::cout<<"Reading in the training and validation data..."<<std::endl;
  // const std::string training_data_filename = "C:/Users/domccl/GitHub/mnist/train-images.idx3-ubyte";
  const std::string training_data_filename = "/home/user/data/train-images-idx3-ubyte";
  // const std::string training_data_filename = "/home/user/data/train-images.idx3-ubyte";
  Eigen::Tensor<float, 2> training_data(training_data_size, input_size);
  ReadMNIST<float>(training_data_filename, training_data, false);

  // const std::string validation_data_filename = "C:/Users/domccl/GitHub/mnist/t10k-images.idx3-ubyte";
  const std::string validation_data_filename = "/home/user/data/t10k-images-idx3-ubyte";
  // const std::string validation_data_filename = "/home/user/data/t10k-images.idx3-ubyte";
  Eigen::Tensor<float, 2> validation_data(validation_data_size, input_size);
  ReadMNIST<float>(validation_data_filename, validation_data, false);

  // Normalize images [BUG FREE]
  training_data = training_data.unaryExpr(UnitScale<float>(training_data));
  validation_data = validation_data.unaryExpr(UnitScale<float>(validation_data));

  // Read input label [BUG FREE]
  std::cout<<"Reading in the training and validation labels..."<<std::endl;  
  // const std::string training_labels_filename = "C:/Users/domccl/GitHub/mnist/train-labels.idx1-ubyte";
  const std::string training_labels_filename = "/home/user/data/train-labels-idx1-ubyte";
  // const std::string training_labels_filename = "/home/user/data/train-labels.idx1-ubyte";
  Eigen::Tensor<float, 2> training_labels(training_data_size, 1);
  ReadMNIST<float>(training_labels_filename, training_labels, true);
  
  // const std::string validation_labels_filename = "C:/Users/domccl/GitHub/mnist/t10k-labels.idx1-ubyte";
  const std::string validation_labels_filename = "/home/user/data/t10k-labels-idx1-ubyte";
  // const std::string validation_labels_filename = "/home/user/data/t10k-labels.idx1-ubyte";
  Eigen::Tensor<float, 2> validation_labels(validation_data_size, 1);
  ReadMNIST<float>(validation_labels_filename, validation_labels, true);

  // Convert labels to 1 hot encoding [BUG FREE]
  Eigen::Tensor<int, 2> training_labels_encoded = OneHotEncoder<float>(training_labels, mnist_labels);
  Eigen::Tensor<int, 2> validation_labels_encoded = OneHotEncoder<float>(validation_labels, mnist_labels);

	// define the model replicator for growth mode
	ModelTrainerTest model_trainer;
	model_trainer.setBatchSize(8);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochs(50);

  // Make the simulation time_steps
	Eigen::Tensor<float, 3> time_steps(model_trainer.getBatchSize(), model_trainer.getMemorySize(), model_trainer.getNEpochs());
	Eigen::Tensor<float, 2> time_steps_tmp(model_trainer.getBatchSize(), model_trainer.getMemorySize());
	time_steps_tmp.setValues({
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 } }
	);
	for (int batch_iter = 0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
		for (int memory_iter = 0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
			for (int epochs_iter = 0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
				time_steps(batch_iter, memory_iter, epochs_iter) = time_steps_tmp(batch_iter, memory_iter);

  // define the model replicator for growth mode
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(0);
  model_replicator.setNLinkAdditions(0);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // Evolve the population
  std::vector<Model> population; 
  const int population_size = 1;
  int n_top = 1;
  int n_random = 1;
  int n_replicates_per_model = 0;
  int mnist_sample_start = 0;
  int mnist_sample_end = 0;
  const int iterations = 5;
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
          input_nodes.size(), 100, output_nodes.size(),
          NodeActivation::ELU, NodeIntegration::Sum,
          NodeActivation::ELU, NodeIntegration::Sum,
          weight_init, solver,
          ModelLossFunction::MSE, std::to_string(i));
        model.initWeights();
        
        // modify the models
        model_replicator.modifyModel(model, std::to_string(i));

        char cout_char[512];
        sprintf(cout_char, "Model %s (Nodes: %d, Links: %d)\n", model.getName().data(), model.getNodes().size(), model.getLinks().size());
        std::cout<<cout_char;
        // for (auto link: model.getLinks())
        // {
        //   memset(cout_char, 0, sizeof(cout_char));
        //   sprintf(cout_char, "Links %s\n", link.getName().data());
        //   std::cout<<cout_char;
        // }
        // for (auto node: model.getNodes())
        // {
        //   memset(cout_char, 0, sizeof(cout_char));
        //   sprintf(cout_char, "Nodes %s\n", node.getName().data());
        //   std::cout<<cout_char;
        // }
        population.push_back(model);
      }
    }

    // make the start and end sample indices [BUG FREE]
    mnist_sample_start = mnist_sample_end;
    mnist_sample_end = mnist_sample_start + model_trainer.getBatchSize()*model_trainer.getNEpochs();
    if (mnist_sample_end > training_data_size - 1)
      mnist_sample_end = mnist_sample_end - model_trainer.getBatchSize()*model_trainer.getNEpochs(); 

    // make a vector of sample_indices [BUG FREE]
    std::vector<int> sample_indices;
    for (int i=0; i<model_trainer.getBatchSize()*model_trainer.getNEpochs(); ++i)
    {
      int sample_index = i + mnist_sample_start;
      if (sample_index > training_data_size - 1)
      {
        sample_index = sample_index - model_trainer.getBatchSize()*model_trainer.getNEpochs();
      }
      sample_indices.push_back(sample_index);
    }   
  
    // Reformat the input data for training [BUG FREE]
    std::cout<<"Reformatting the input data..."<<std::endl;  
    Eigen::Tensor<float, 4> input_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)input_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
      for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
        for (int nodes_iter=0; nodes_iter<input_nodes.size(); ++nodes_iter)
          for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = training_data(sample_indices[epochs_iter*model_trainer.getBatchSize() + batch_iter], nodes_iter);

    // reformat the output data for training [BUG FREE]
    std::cout<<"Reformatting the output data..."<<std::endl;  
    Eigen::Tensor<float, 4> output_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), (int)output_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
			for (int memory_iter = 0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
				for (int nodes_iter=0; nodes_iter<output_nodes.size(); ++nodes_iter)
					for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = (float)training_labels_encoded(sample_indices[epochs_iter*model_trainer.getBatchSize() + batch_iter], nodes_iter);

    // model modification scheduling
    if (iter > 100)
    {
      model_replicator.setNNodeAdditions(1);
      model_replicator.setNLinkAdditions(2);
      model_replicator.setNNodeDeletions(1);
      model_replicator.setNLinkDeletions(2);
    }
    else if (iter > 1 && iter < 100)
    {
      model_replicator.setNNodeAdditions(1);
      model_replicator.setNLinkAdditions(2);
      model_replicator.setNNodeDeletions(1);
      model_replicator.setNLinkDeletions(2);
    }
    else if (iter == 0)
    {      
      model_replicator.setNNodeAdditions(10);
      model_replicator.setNLinkAdditions(20);
      model_replicator.setNNodeDeletions(0);
      model_replicator.setNLinkDeletions(0);
    }

    // train the population
    std::cout<<"Training the models..."<<std::endl;
    population_trainer.trainModels(population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes, n_threads);

    // reformat the input data for validation

    // reformat the output data for validation

    // select the top N from the population
    std::cout<<"Selecting the models..."<<std::endl;    
    model_trainer.setNEpochs(4);  // lower the number of epochs for validation
		std::vector<std::pair<int, float>> models_validation_errors = population_trainer.selectModels(
      n_top, n_random, population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes, n_threads);
    model_trainer.setNEpochs(50);  // restor the number of epochs for training

    for (const Model& model: population)
    {
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      char cout_char[512];
      sprintf(cout_char, "Model %s (Nodes: %d, Links: %d) error: %.2f\n", model.getName().data(), model.getNodes().size(), model.getLinks().size(), total_error.data()[0]);
      std::cout<<cout_char;
      // for (auto link: model.getLinks())
      //   printf("Links %s\n", link.getName().data());
    }

    if (iter < iterations - 1)  
    {
      if (iter == 0)
      {
        n_top = 5;
        n_random = 5;
        n_replicates_per_model = 10;
      }
      else
      {
        n_top = 5;
        n_random = 5;
        n_replicates_per_model = 2;
      }
      // replicate and modify models
      std::cout<<"Replicating and modifying the models..."<<std::endl;
      population_trainer.replicateModels(population, model_replicator, input_nodes, output_nodes, 
				n_replicates_per_model, std::to_string(iter), n_threads);
      std::cout<<"Population size of "<<population.size()<<std::endl;
    }
  }

  // write the model to file
  WeightFile weightfile;
  weightfile.storeWeightsCsv("MNISTExampleWeights.csv", population[0].getWeights());
  LinkFile linkfile;
  linkfile.storeLinksCsv("MNISTExampleLinks.csv", population[0].getLinks());
  NodeFile nodefile;
  nodefile.storeNodesCsv("MNISTExampleNodes.csv", population[0].getNodes());

  return 0;
}