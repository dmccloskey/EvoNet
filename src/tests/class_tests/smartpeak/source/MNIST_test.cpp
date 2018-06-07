/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MNIST test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/PopulationTrainer.h>

#include <SmartPeak/ml/Model.h>
#include <fstream>

using namespace SmartPeak;

BOOST_AUTO_TEST_SUITE(mnist)

// Toy ModelTrainer used for all tests
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
      // assign the input data
      model.mapValuesToNodes(input.chip(iter, 3), input_nodes, NodeStatus::activated, "output"); 

      // forward propogate
      model.forwardPropogate(0);

      // calculate the model error and node output error
      model.calculateError(output.chip(iter, 2), output_nodes);
      std::cout<<"Model error: "<<model.getError().sum()<<std::endl;

      // back propogate
      model.backPropogate(0);

      // update the weights
      model.updateWeights(1);   

      // reinitialize the model
      model.reInitializeNodeStatuses();
    }
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
    // printf("Data checks passed\n");
    
    // Initialize the model
    model.initNodes(getBatchSize(), getMemorySize());
    // printf("Initialized the model\n");

    for (int iter = 0; iter < getNEpochs(); ++iter) // use n_epochs here
    {
      // printf("validation epoch: %d\t", iter);

      // assign the input data
      model.mapValuesToNodes(input.chip(iter, 3), input_nodes, NodeStatus::activated, "output"); 

      // forward propogate
      model.forwardPropogate(0);

      // calculate the model error and node output error
      model.calculateError(output.chip(iter, 2), output_nodes); 
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      model_error.push_back(total_error(0));  
      // std::cout<<"Model error: "<<total_error(0)<<std::endl;

      // reinitialize the model
      model.reInitializeNodeStatuses();
      model.initNodes(getBatchSize(), getMemorySize());
    }
    return model_error;
  }
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
void ReadMNIST(const std::string& filename, Eigen::Tensor<T, 2> data)
{
  // dims: sample, pixel intensity or sample, label
  // e.g., pixel data dims: 1000 x (28x28)
  // e.g., label data dims: 1000 x 1

  std::ifstream file (filename, std::ios::binary);
  if (file.is_open())
  {
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= ReverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= ReverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= ReverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= ReverseInt(n_cols);
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

template<typename T>
class UnitScale
{
public: 
  UnitScale(){}; 
  UnitScale(const Eigen::Tensor<T, 2>& data){setUnitScale(data);}; 
  ~UnitScale(){};
  void setUnitScale(const Eigen::Tensor<T, 2>& data)
  {
    const Eigen::Tensor<T,0> max_value = data.maximum();
    const Eigen::Tensor<T,0> min_value = data.minimum();
    unit_scale_ = 1/sqrt(pow(max_value(0) - min_value(0), 2));
  }  
  T operator()(const T& x_I) const { return x_I/unit_scale_; };
private:
  T unit_scale_;
};

template<typename T>
Eigen::Tensor<int, 2> OneHotEncoder(Eigen::Tensor<T, 2>& data, const std::vector<T>& all_possible_values)
{
  // integer encode input data
  std::map<T, int> T_to_int;
  for (int i=0; i<all_possible_values.size(); ++i)
    T_to_int.emplace(all_possible_values[i], i);

  // convert to 1 hot vector
  Eigen::Tensor<int, 2> onehot_encoded(data.dimension(0), T_to_int.size());
  onehot_encoded.setConstant(0);
  for (int i=0; i<data.dimension(0); ++i)
    onehot_encoded(i, T_to_int.at(data(i,0)))=1;
  
  return onehot_encoded;
}

BOOST_AUTO_TEST_CASE(mnistTest) 
{
  PopulationTrainer population_trainer;

  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(64);
  model_trainer.setMemorySize(0);
  model_trainer.setNEpochs(100);

  const int mnist_pixels = 784;
  const int mnist_training_size = 1000;
  const std::vector<float> mnist_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};


  // Make the input nodes
  std::vector<std::string> input_nodes;
  for (int i=0; i<mnist_pixels; ++i)
    input_nodes.push_back("Input_" + std::to_string(i));

  // Make the output nodes
  std::vector<std::string> output_nodes;
  for (int i=0; i<mnist_labels.size(); ++i)
    output_nodes.push_back("output_" + std::to_string(i));

  // Read input images
  const std::string training_data = ""; //TODO
  Eigen::Tensor<float, 2> input_data_mnist(mnist_training_size, mnist_pixels);
  ReadMNIST<float>(training_data, input_data_mnist);

  // Normalize images
  input_data_mnist = input_data_mnist.unaryExpr(UnitScale<float>(input_data_mnist));

  // Read input labels
  const std::string training_labels = ""; //TODO
  Eigen::Tensor<float, 2> output_data_mnist(mnist_training_size, 1);
  ReadMNIST<float>(training_labels, output_data_mnist);

  // Convert labels to 1 hot encoding
  Eigen::Tensor<int, 2> output_data_encoded = OneHotEncoder<float>(output_data_mnist, mnist_labels);

  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps;

  // define the model replicator for growth mode
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // Evolve the population
  std::vector<Model> population; 
  const int iterations = 10;
  const int population_size = 12;
  int mnist_sample_start = 0;
  int mnist_sample_end = 0;
  for (int iter=0; iter<iterations; ++iter)
  {
    printf("Iteration #: %d\n", iter);

    if (iter == 0)
    {
      // define the initial population
      for (int i=0; i<population_size; ++i)
      {
        // baseline model
        std::shared_ptr<WeightInitOp> weight_init;
        std::shared_ptr<SolverOp> solver;
        weight_init.reset(new RandWeightInitOp(1.0));
        solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
        Model model = model_replicator.makeBaselineModel(
          input_nodes.size(), 0, output_nodes.size(),
          NodeActivation::ELU,
          NodeActivation::ELU,
          weight_init, solver,
          ModelLossFunction::MSE, std::to_string(i));
        model.initWeights();
        
        // modify the models
        model_replicator.modifyModel(model, std::to_string(i));

        population.push_back(model);
      }
    }

    // make the start and end sample indices
    mnist_sample_start = mnist_sample_end;
    mnist_sample_end = mnist_sample_start + model_trainer.getBatchSize()*model_trainer.getNEpochs();
    if (mnist_sample_end > mnist_training_size - 1)
      mnist_sample_end = mnist_sample_end - model_trainer.getBatchSize()*model_trainer.getNEpochs(); 

    // make a vector of sample_indices
    std::vector<int> sample_indices;
    for (int i=0; i<model_trainer.getBatchSize()*model_trainer.getNEpochs(); ++i)
    {
      int sample_index = i + mnist_sample_start;
      if (sample_index > mnist_training_size - 1);
        sample_index = sample_index - model_trainer.getBatchSize()*model_trainer.getNEpochs();
      sample_indices.push_back(sample_index);
    }   
  
    // Reformat the input data for training
    Eigen::Tensor<float, 4> input_data(model_trainer.getBatchSize(), model_trainer.getMemorySize(), input_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
      for (int memory_iter=0; memory_iter<model_trainer.getMemorySize(); ++memory_iter)
        for (int nodes_iter=0; nodes_iter<input_nodes.size(); ++nodes_iter)
          for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
            input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = input_data_mnist(sample_indices[epochs_iter*model_trainer.getBatchSize() + batch_iter], nodes_iter);

    // reformat the output data for training
    Eigen::Tensor<float, 3> output_data(model_trainer.getBatchSize(), output_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
      for (int nodes_iter=0; nodes_iter<output_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
          output_data(batch_iter, nodes_iter, epochs_iter) = (float)output_data_encoded(sample_indices[epochs_iter*model_trainer.getBatchSize() + batch_iter], nodes_iter);

    // // model modification scheduling
    // if (iter > 100)
    // {
    //   model_replicator.setNNodeAdditions(1);
    //   model_replicator.setNLinkAdditions(1);
    //   model_replicator.setNNodeDeletions(1);
    //   model_replicator.setNLinkDeletions(1);
    // }
    // else
    // {
    //   model_replicator.setNNodeAdditions(0);
    //   model_replicator.setNLinkAdditions(3);
    //   model_replicator.setNNodeDeletions(0);
    //   model_replicator.setNLinkDeletions(3);
    // }

    // train the population
    population_trainer.trainModels(population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes);

    // reformat the input data for validation

    // reformat the output data for validation

    // select the top N from the population
    population_trainer.selectModels(
      3, 3, population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes);

    for (const Model& model: population)
    {
      const Eigen::Tensor<float, 0> total_error = model.getError().sum();
      printf("Model %s (Nodes: %d, Links: %d) error: %.2f\n", model.getName().data(), model.getNodes().size(), model.getLinks().size(), total_error.data()[0]);
      // for (auto link: model.getLinks())
      //   printf("Links %s\n", link.getName().data());
    }

    if (iter < iterations - 1)  
    {
      // replicate and modify models
      population_trainer.replicateModels(population, model_replicator, 3, std::to_string(iter));
    }
  }
}


BOOST_AUTO_TEST_SUITE_END()