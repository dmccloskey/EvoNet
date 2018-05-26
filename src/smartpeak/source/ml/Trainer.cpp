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
    io::CSVReader<3> nodes_in(filename_nodes);
    nodes_in.read_header(io::ignore_extra_column, 
      "node_name", "node_type", "node_status");
    std::string node_name, node_type_str, node_status_str;

    std::vector<Node> nodes;
    while(nodes_in.read_row(node_name, node_type_str, node_status_str))
    {
      // parse the node_type
      NodeType node_type;
      if (node_type_str == "ReLU") node_status = NodeType::deactivated;
      else if (node_type_str == "ELU") node_type = NodeType::ELU;
      else if (node_type_str == "input") node_type = NodeType::input;
      else if (node_type_str == "bias") node_type = NodeType::bias;
      else if (node_type_str == "Sigmoid") node_type = NodeType::Sigmoid;
      else if (node_type_str == "TanH") node_type = NodeType::TanH;
      else std::cout<<"NodeType for node_name "<<node_name<<" was not recognized."<<std::endl;

      // parse the node_status
      NodeStatus node_status;
      if (node_status_str == "deactivated") node_status = NodeStatus::deactivated;
      else if (node_status_str == "initialized") node_status = NodeStatus::initialized;
      else if (node_status_str == "activated") node_status = NodeStatus::activated;
      else if (node_status_str == "corrected") node_status = NodeStatus::corrected;
      else std::cout<<"NodeStatus for node_name "<<node_name<<" was not recognized."<<std::endl;
      
      Node node(node_name, node_type, node_status);
      nodes.push_back(node);
    }

    // Read in the links
    io::CSVReader<4> links_in(filename_links);
    links_in.read_header(io::ignore_extra_column, 
      "link_name", "source_node_name", "sink_node_name", "weight_name");
    std::string link_name, source_node_name, sink_node_name, weight_name;

    std::vector<Link> links;
    while(links_in.read_row(link_name, source_node_name, sink_node_name, weight_name))
    {
      Link link(link_name, source_node_name, sink_node_name, weight_name);
      links.push_back(link);
    }

    // Read in the weights
    io::CSVReader<5> weights_in(filename_weights);
    weights_in.read_header(io::ignore_extra_column, 
      "weight_name", "weight_init_op", "weight_init_params", "solver_op", "solver_params");
    std::string weight_name, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str;

    std::vector<Weight> weights;
    while(weights_in.read_row(weight_name_str, weight_init_op_str, weight_init_params_str, solver_op_str, solver_params_str))
    {
      // parse the weight_init_params
      std::map<std::str, float> weight_init_params;
      // TODO...

      // parse the weight_init_op
      std::shared_ptr<WeightInitOp> weight_init;
      if (weight_init_op_str == "ConstWeightInitOp")
      {
        weight_init.reset(new ConstWeightInitOp(1.0));
      }
      else if (weight_init_op_str == "RandWeightInitOp")
      {
        weight_init.reset(new RandWeightInitOp(1.0));
      }
      else std::cout<<"WeightInitOp for weight_name "<<weight_name<<" was not recognized."<<std::endl;

      // parse the solver_params_str
      std::map<std::str, float> weight_params;
      // TODO...

      // parse the solver_op
      std::shared_ptr<SolverOp> solver;
      if (weight_init_op_str == "SGDOp")
      {
        solver.reset(new SGDOp(0.01, 0.9));
      }
      else if (weight_init_op_str == "AdamOp")
      {
        solver.reset(new AdamOp(0.01, 0.9, 0.999, 1e-8));
      }
      else std::cout<<"WeightInitOp for weight_name "<<weight_name<<" was not recognized."<<std::endl;

      Weight weight(weight_name, weight_init, solver);
      weights.push_back(weight);
    }

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