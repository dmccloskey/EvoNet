/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MNIST test suite 

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/ml/Model.h>

#include <unsupported/Eigen/CXX11/Tensor>


#include <fstream>
#include <cuda.h>
#include <cuda_runtime_api.h>

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

BOOST_AUTO_TEST_SUITE(mnist)

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
    if (!model.checkNodeNames(input_nodes))
    {
      return;
    }
    if (!model.checkNodeNames(output_nodes))
    {
      return;
    }
    printf("Data checks passed\n");
    
    // Initialize the model
    model.initNodes(getBatchSize(), getMemorySize());
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

//  BOOST_AUTO_TEST_CASE(cuda) 
//  {
//    const int tensor_dim = 2;
//    Eigen::Tensor<float, 1> in1(tensor_dim);
//    Eigen::Tensor<float, 1> in2(tensor_dim);
//    Eigen::Tensor<float, 1> out(tensor_dim);
//    in1.setRandom();
//    in2.setRandom();

//    std::size_t in1_bytes = in1.size() * sizeof(float);
//    std::size_t in2_bytes = in2.size() * sizeof(float);
//    std::size_t out_bytes = out.size() * sizeof(float);

//    float* d_in1;
//    float* d_in2;
//    float* d_out;
//    cudaMalloc((void**)(&d_in1), in1_bytes);
//    cudaMalloc((void**)(&d_in2), in2_bytes);
//    cudaMalloc((void**)(&d_out), out_bytes);

//    cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);

//    Eigen::CudaStreamDevice stream;
//    Eigen::GpuDevice gpu_device(&stream);

//    Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
//        d_in1, tensor_dim);
//    Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
//        d_in2, tensor_dim);
//    Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
//        d_out, tensor_dim);

//    gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;

//    //BOOST_CHECK_EQUAL(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
//    //                       gpu_device.stream()) == cudaSuccess);
//    //BOOST_CHECK_EQUAL(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
//  }

BOOST_AUTO_TEST_CASE(mnistTest) 
{
  PopulationTrainer population_trainer;

  const std::size_t input_size = 784;
  const std::size_t training_data_size = 1000; //60000;
  const std::size_t validation_data_size = 100; //10000;
  const std::vector<float> mnist_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

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
  const std::string training_data_filename = "/home/user/data/train-images.idx3-ubyte";
  Eigen::Tensor<float, 2> training_data(training_data_size, input_size);
  ReadMNIST<float>(training_data_filename, training_data, false);

  const std::string validation_data_filename = "/home/user/data/t10k-images.idx3-ubyte";
  Eigen::Tensor<float, 2> validation_data(validation_data_size, input_size);
  ReadMNIST<float>(validation_data_filename, validation_data, false);

  // Normalize images [BUG FREE]
  training_data = training_data.unaryExpr(UnitScale<float>(training_data));
  validation_data = validation_data.unaryExpr(UnitScale<float>(validation_data));

  // Read input label [BUG FREE]
  std::cout<<"Reading in the training and validation labels..."<<std::endl;  
  const std::string training_labels_filename = "/home/user/data/train-labels.idx1-ubyte";
  Eigen::Tensor<float, 2> training_labels(training_data_size, 1);
  ReadMNIST<float>(training_labels_filename, training_labels, true);
  
  const std::string validation_labels_filename = "/home/user/data/t10k-labels.idx1-ubyte";
  Eigen::Tensor<float, 2> validation_labels(validation_data_size, 1);
  ReadMNIST<float>(validation_labels_filename, validation_labels, true);

  // Convert labels to 1 hot encoding [BUG FREE]
  Eigen::Tensor<int, 2> training_labels_encoded = OneHotEncoder<float>(training_labels, mnist_labels);
  Eigen::Tensor<int, 2> validation_labels_encoded = OneHotEncoder<float>(validation_labels, mnist_labels);

  // Make the simulation time_steps
  Eigen::Tensor<float, 3> time_steps;

  // define the model replicator for growth mode
  ModelTrainerTest model_trainer;
  model_trainer.setBatchSize(4);
  model_trainer.setMemorySize(1);
  model_trainer.setNEpochs(100);

  // define the model replicator for growth mode
  ModelReplicator model_replicator;
  model_replicator.setNNodeAdditions(1);
  model_replicator.setNLinkAdditions(1);
  model_replicator.setNNodeDeletions(0);
  model_replicator.setNLinkDeletions(0);

  // Evolve the population
  std::vector<Model> population; 
  const int population_size = 1;
  const int n_top = 1;
  const int n_random = 1;
  const int n_replicates_per_model = 0;
  int mnist_sample_start = 0;
  int mnist_sample_end = 0;
  const int iterations = 2;
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
          NodeActivation::ELU,
          NodeActivation::ELU,
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
    Eigen::Tensor<float, 3> output_data(model_trainer.getBatchSize(), (int)output_nodes.size(), model_trainer.getNEpochs());
    for (int batch_iter=0; batch_iter<model_trainer.getBatchSize(); ++batch_iter)
      for (int nodes_iter=0; nodes_iter<output_nodes.size(); ++nodes_iter)
        for (int epochs_iter=0; epochs_iter<model_trainer.getNEpochs(); ++epochs_iter)
          output_data(batch_iter, nodes_iter, epochs_iter) = (float)training_labels_encoded(sample_indices[epochs_iter*model_trainer.getBatchSize() + batch_iter], nodes_iter);

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
    std::cout<<"Training the models..."<<std::endl;
    population_trainer.trainModels(population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes);

    // reformat the input data for validation

    // reformat the output data for validation

    // select the top N from the population
    std::cout<<"Selecting the models..."<<std::endl;
    population_trainer.selectModels(
      n_top, n_random, population, model_trainer,
      input_data, output_data, time_steps, input_nodes, output_nodes);

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
      // replicate and modify models
      std::cout<<"Replicating and modifying the models..."<<std::endl;
      population_trainer.replicateModels(population, model_replicator, n_replicates_per_model, std::to_string(iter));
    }
  }
}


BOOST_AUTO_TEST_SUITE_END()