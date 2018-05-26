/**TODO:  Add copyright*/

#include <SmartPeak/ml/Trainer.h>
#include <SmartPeak/io/csv.h>


namespace SmartPeak
{
  Trainer::Trainer(){};
  Trainer::~Trainer(){};

  void Trainer::setBatchSize(const int& batch_size)
  {
    batch_size_ = batch_size;
  }

  void Trainer::setMemorySize(const int& memory_size)
  {
    memory_size_ = memory_size;    
  }

  void Trainer::setNEpochs(const int& n_epochs)
  {
    n_epochs_ = n_epochs;    
  }

  int Trainer::getBatchSize() const
  {
    return batch_size_;
  }

  int Trainer::getMemorySize() const
  {
    return memory_size_;
  }

  int Trainer::getNEpochs() const
  {
    return n_epochs_;
  }

  bool Trainer::loadModel(const std::string& filename_nodes,
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
  }

  bool Trainer::loadNodeStates(const std::string& filename, Model& model)
  {
    
  }

  bool Trainer::loadWeights(const std::string& filename, Model& model)
  {
    
  }

  bool Trainer::storeModel(const std::string& filename_nodes,
    const std::string& filename_links,
    const std::string& filename_weights,
    const Model& model)
  {
    
  }

  bool Trainer::storeNodeStates(const std::string& filename, const Model& model)
  {
    
  }

  bool Trainer::storeWeights(const std::string& filename, const Model& model)
  {
    
  }

  bool Trainer::loadInputData(const std::string& filename, Eigen::Tensor<float, 4>& input)
  {
    
  }

  bool Trainer::loadOutputData(const std::string& filename, Eigen::Tensor<float, 3>& output)
  {
    
  }

  bool Trainer::checkInputData(const int& n_epochs,
    const Eigen::Tensor<float, 4>& input,
    const int& batch_size,
    const int& memory_size,
    const std::vector<std::string>& input_nodes)
  {
    
  }

  bool Trainer::checkOutputData(const int& n_epochs,
    const Eigen::Tensor<float, 3>& output,
    const int& batch_size,
    const std::vector<std::string>& output_nodes)
  {
    
  }
}