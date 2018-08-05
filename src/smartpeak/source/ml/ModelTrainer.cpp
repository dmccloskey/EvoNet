/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/io/csv.h>


namespace SmartPeak
{
  ModelTrainer::ModelTrainer(){};
  ModelTrainer::~ModelTrainer(){};

  void ModelTrainer::setBatchSize(const int& batch_size)
  {
    batch_size_ = batch_size;
  }

  void ModelTrainer::setMemorySize(const int& memory_size)
  {
    memory_size_ = memory_size;    
  }

  void ModelTrainer::setNEpochs(const int& n_epochs)
  {
    n_epochs_ = n_epochs;    
  }

  int ModelTrainer::getBatchSize() const
  {
    return batch_size_;
  }

  int ModelTrainer::getMemorySize() const
  {
    return memory_size_;
  }

  int ModelTrainer::getNEpochs() const
  {
    return n_epochs_;
  }

  bool ModelTrainer::loadModel(const std::string& filename_nodes,
    const std::string& filename_links,
    const std::string& filename_weights,
    Model& model)
  {
    // Read in the nodes
    std::vector<Node> nodes;

    // Read in the links
    std::vector<Link> links;

    // Read in the weights
    std::vector<Weight> weights;

    // Make the model
    model.addNodes(nodes);
    model.addLinks(links);
    model.addWeights(weights);

	return true;
  }

  bool ModelTrainer::loadNodeStates(const std::string& filename, Model& model)
  {
	return true;
  }

  bool ModelTrainer::loadWeights(const std::string& filename, Model& model)
  {
	return true;
  }

  bool ModelTrainer::storeModel(const std::string& filename_nodes,
    const std::string& filename_links,
    const std::string& filename_weights,
    const Model& model)
  {
	return true;
  }

  bool ModelTrainer::storeNodeStates(const std::string& filename, const Model& model)
  {
	return true;    
  }

  bool ModelTrainer::storeWeights(const std::string& filename, const Model& model)
  {
	return true;
  }

  bool ModelTrainer::loadInputData(const std::string& filename, Eigen::Tensor<float, 4>& input)
  {
	return true;
  }

  bool ModelTrainer::loadOutputData(const std::string& filename, Eigen::Tensor<float, 3>& output)
  {
	return true;
  }

  bool ModelTrainer::checkInputData(const int& n_epochs,
    const Eigen::Tensor<float, 4>& input,
    const int& batch_size,
    const int& memory_size,
    const std::vector<std::string>& input_nodes)
  {
    if (input.dimension(0) != batch_size)
    {
      printf("batch_size of %d is not compatible with the input dim 0 of %d\n", batch_size, (int)input.dimension(0));
      return false;
    }
    else if (input.dimension(1) != memory_size)
    {
      printf("memory_size of %d is not compatible with the input dim 1 of %d\n", memory_size, (int)input.dimension(1));
      return false;
    }
    else if (input.dimension(2) != input_nodes.size())
    {
      printf("input_nodes size of %d is not compatible with the input dim 2 of %d\n", input_nodes.size(), (int)input.dimension(2));
      return false;
    }
    else if (input.dimension(3) != n_epochs)
    {
      printf("n_epochs of %d is not compatible with the input dim 3 of %d\n", n_epochs, (int)input.dimension(3));
      return false;
    }
    else 
    {
      return true;
    }
  }

  bool ModelTrainer::checkOutputData(const int& n_epochs,
    const Eigen::Tensor<float, 4>& output,
    const int& batch_size,
		const int& memory_size,
    const std::vector<std::string>& output_nodes)
  {
		if (output.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the output dim 0 of %d\n", batch_size, (int)output.dimension(0));
			return false;
		}
		else if (output.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the output dim 1 of %d\n", memory_size, (int)output.dimension(1));
			return false;
		}
		else if (output.dimension(2) != output_nodes.size())
		{
			printf("output_nodes size of %d is not compatible with the output dim 2 of %d\n", output_nodes.size(), (int)output.dimension(2));
			return false;
		}
		else if (output.dimension(3) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the output dim 3 of %d\n", n_epochs, (int)output.dimension(3));
			return false;
		}
		else
		{
			return true;
		}
  }
	bool ModelTrainer::checkTimeSteps(const int & n_epochs, const Eigen::Tensor<float, 3>& time_steps, const int & batch_size, const int & memory_size)
	{
		if (time_steps.dimension(0) != batch_size)
		{
			printf("batch_size of %d is not compatible with the time_steps dim 0 of %d\n", batch_size, (int)time_steps.dimension(0));
			return false;
		}
		else if (time_steps.dimension(1) != memory_size)
		{
			printf("memory_size of %d is not compatible with the time_steps dim 1 of %d\n", memory_size, (int)time_steps.dimension(1));
			return false;
		}
		else if (time_steps.dimension(2) != n_epochs)
		{
			printf("n_epochs of %d is not compatible with the time_steps dim 3 of %d\n", n_epochs, (int)time_steps.dimension(2));
			return false;
		}
		else
		{
			return true;
		}
	}
}