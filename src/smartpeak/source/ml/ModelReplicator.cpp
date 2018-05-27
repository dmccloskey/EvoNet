/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelReplicator.h>


namespace SmartPeak
{
  ModelReplicator::ModelReplicator(){};
  ModelReplicator::~ModelReplicator(){};

  void ModelReplicator::setBatchSize(const int& n_node_additions)
  {
    n_node_additions_ = n_node_additions;
  }

  void ModelReplicator::setMemorySize(const int& n_link_additions)
  {
    n_link_additions_ = n_link_additions;    
  }

  void ModelReplicator::setNEpochs(const int& n_node_deletions)
  {
    n_node_deletions_ = n_node_deletions;    
  }

  void ModelReplicator::setNLinkDeletions(const int& n_link_deletions)
  {
    n_link_deletions_ = n_link_deletions;
  }

  void ModelReplicator::setNWeightChanges(const int& n_weight_changes)
  {
    n_weight_changes_ = n_weight_changes;    
  }

  void ModelReplicator::setWeightChangeStDev(const float& weight_change_stdev)
  {
    weight_change_stdev_ = weight_change_stdev;    
  }

  int ModelReplicator::getBatchSize() const
  {
    return n_node_additions_;
  }

  int ModelReplicator::getMemorySize() const
  {
    return n_link_additions_;
  }

  int ModelReplicator::getNEpochs() const
  {
    return n_node_deletions_;
  }

  int ModelReplicator::getNLinkDeletions() const
  {
    return n_link_deletions_;
  }

  int ModelReplicator::getNWeightChanges() const
  {
    return n_weight_changes_;
  }

  float ModelReplicator::getWeightChangeStDev() const
  {
    return weight_change_stdev_;
  }

  Model ModelReplicator::makeBaselineModel(const int& n_input_nodes, const int& n_hidden_nodes, const int& n_output_nodes,
    const NodeType& hidden_node_type, const NodeType& output_node_type,
    const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver)
  {
    Model model;

    // Create the input nodes
    for (int i=0; i<n_input_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Input_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, NodeType::input, NodeStatus::deactivated);
      model.addNodes({node});
    }
    // Create the hidden nodes + biases
    for (int i=0; i<n_hidden_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Hidden_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, hidden_node_type, NodeStatus::deactivated);

      char bias_name_char[64];
      sprintf(bias_name_char, "Hidden_bias_%d", i);
      std::string bias_name(bias_name_char);
      Node bias(bias_name, NodeType::bias, NodeStatus::deactivated);
      model.addNodes({node, bias});
    }
    // Create the output nodes + biases
    for (int i=0; i<n_output_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Output_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, output_node_type, NodeStatus::deactivated);
      
      char bias_name_char[64];
      sprintf(bias_name_char, "Output_bias_%d", i);
      std::string bias_name(bias_name_char);
      Node bias(bias_name, NodeType::bias, NodeStatus::deactivated);
      model.addNodes({node, bias});
    }

    // Create the weights and links for input to hidden
    for (int i=0; i<n_input_nodes; ++i)
    {
      char input_name_char[64];
      sprintf(input_name_char, "Input_%d", i);
      std::string input_name(input_name_char);

      for (int j=0; j<n_hidden_nodes; ++j)
      {
        char hidden_name_char[64];
        sprintf(hidden_name_char, "Hidden_%d", j);
        std::string hidden_name(hidden_name_char);

        char link_name_char[64];
        sprintf(link_name_char, "Input_%d_to_Hidden_%d", i, j);
        std::string link_name(link_name_char);

        char weight_name_char[64];
        sprintf(weight_name_char, "Weight_%d_to_%d", i, j);
        std::string weight_name(weight_name_char);

        std::shared_ptr<WeightInitOp> hidden_weight_init = weight_init;
        std::shared_ptr<SolverOp> hidden_solver = solver;
        Weight weight(weight_name_char, hidden_weight_init, hidden_solver);
        Link link(link_name, input_name, hidden_name, weight_name);

        char bias_name_char[64];
        sprintf(bias_name_char, "Hidden_bias_%d", j);
        std::string bias_name(bias_name_char);

        char weight_bias_name_char[64];
        sprintf(weight_bias_name_char, "Weight_bias_%d_to_%d", i, j);
        std::string weight_bias_name(weight_bias_name_char);

        char link_bias_name_char[64];
        sprintf(link_bias_name_char, "Bias_%d_to_Hidden_%d", i, j);
        std::string link_bias_name(link_bias_name_char);

        std::shared_ptr<WeightInitOp> bias_weight_init = weight_init;
        std::shared_ptr<SolverOp> bias_solver = solver;
        Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
        Link link_bias(link_bias_name, bias_name, hidden_name, weight_bias_name);

        model.addWeights({weight, weight_bias});
        model.addLinks({link, link_bias});
      }
    }

    // Create the weights and links for hidden to output
    for (int i=0; i<n_hidden_nodes; ++i)
    {
      char hidden_name_char[64];
      sprintf(hidden_name_char, "Hidden_%d", i);
      std::string hidden_name(hidden_name_char);

      for (int j=0; j<n_output_nodes; ++j)
      {
        char output_name_char[64];
        sprintf(output_name_char, "Output_%d", j);
        std::string output_name(output_name_char);

        char link_name_char[64];
        sprintf(link_name_char, "Input_%d_to_Output_%d", i, j);
        std::string link_name(link_name_char);

        char weight_name_char[64];
        sprintf(weight_name_char, "Weight_%d_to_%d", i, j);
        std::string weight_name(weight_name_char);

        std::shared_ptr<WeightInitOp> output_weight_init = weight_init;
        std::shared_ptr<SolverOp> output_solver = solver;
        Weight weight(weight_name_char, output_weight_init, output_solver);
        Link link(link_name, hidden_name, output_name, weight_name);

        char bias_name_char[64];
        sprintf(bias_name_char, "Output_bias_%d", j);
        std::string bias_name(bias_name_char);

        char weight_bias_name_char[64];
        sprintf(weight_bias_name_char, "Weight_bias_%d_to_%d", i, j);
        std::string weight_bias_name(weight_bias_name_char);

        char link_bias_name_char[64];
        sprintf(link_bias_name_char, "Bias_%d_to_Output_%d", i, j);
        std::string link_bias_name(link_bias_name_char);

        std::shared_ptr<WeightInitOp> bias_weight_init = weight_init;
        std::shared_ptr<SolverOp> bias_solver = solver;
        Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
        Link link_bias(link_bias_name, bias_name, output_name, weight_bias_name);

        model.addWeights({weight, weight_bias});
        model.addLinks({link, link_bias});
      }
    }
    return model;
  }

  void ModelReplicator::modifyModel(Model& model)
  {

  }

  void ModelReplicator::addNode(Model& model)
  {
    // pick a random node from the model
    // that is not an input or bias

    // copy the node, its links, and its bias

  }

  void ModelReplicator::addLink(Model& model)
  {    
    // pick a random source and sink node from the model
    // that are not inputs or biases
    // enforce ACG link direcitonality if required
    // [TODO: any other rules?]

    // add link to the model
  }

  void ModelReplicator::deleteNode(Model& model)
  {

  }

  void ModelReplicator::deleteLink(Model& model)
  {

  }

  void ModelReplicator::modifyWeight(Model& model)
  {
    
  }
}