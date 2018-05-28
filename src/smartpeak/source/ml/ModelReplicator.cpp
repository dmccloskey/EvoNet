/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelReplicator.h>

#include <random>

namespace SmartPeak
{
  ModelReplicator::ModelReplicator(){};
  ModelReplicator::~ModelReplicator(){};

  void ModelReplicator::setNNodeCopies(const int& n_node_copies)
  {
    n_node_copies_ = n_node_copies;
  }

  void ModelReplicator::setNNodeAdditions(const int& n_node_additions)
  {
    n_node_additions_ = n_node_additions;
  }

  void ModelReplicator::setNLinkAdditions(const int& n_link_additions)
  {
    n_link_additions_ = n_link_additions;    
  }

  void ModelReplicator::setNNodeDeletions(const int& n_node_deletions)
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

  int ModelReplicator::getNNodeCopies() const
  {
    return n_node_copies_;
  }

  int ModelReplicator::getNNodeAdditions() const
  {
    return n_node_additions_;
  }

  int ModelReplicator::getNLinkAdditions() const
  {
    return n_link_additions_;
  }

  int ModelReplicator::getNNodeDeletions() const
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
        sprintf(weight_name_char, "Input_%d_to_Hidden_%d", i, j);
        std::string weight_name(weight_name_char);

        std::shared_ptr<WeightInitOp> hidden_weight_init = weight_init;
        std::shared_ptr<SolverOp> hidden_solver = solver;
        Weight weight(weight_name_char, hidden_weight_init, hidden_solver);
        Link link(link_name, input_name, hidden_name, weight_name);

        char bias_name_char[64];
        sprintf(bias_name_char, "Hidden_bias_%d", j);
        std::string bias_name(bias_name_char);

        char weight_bias_name_char[64];
        sprintf(weight_bias_name_char, "Bias_%d_to_Hidden_%d", j, j);
        std::string weight_bias_name(weight_bias_name_char);

        char link_bias_name_char[64];
        sprintf(link_bias_name_char, "Bias_%d_to_Hidden_%d", j, j);
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
        sprintf(link_name_char, "Hidden_%d_to_Output_%d", i, j);
        std::string link_name(link_name_char);

        char weight_name_char[64];
        sprintf(weight_name_char, "Hidden_%d_to_Output_%d", i, j);
        std::string weight_name(weight_name_char);

        std::shared_ptr<WeightInitOp> output_weight_init = weight_init;
        std::shared_ptr<SolverOp> output_solver = solver;
        Weight weight(weight_name_char, output_weight_init, output_solver);
        Link link(link_name, hidden_name, output_name, weight_name);

        char bias_name_char[64];
        sprintf(bias_name_char, "Output_bias_%d", j);
        std::string bias_name(bias_name_char);

        char weight_bias_name_char[64];
        sprintf(weight_bias_name_char, "Bias_%d_to_Output_%d", j, j);
        std::string weight_bias_name(weight_bias_name_char);

        char link_bias_name_char[64];
        sprintf(link_bias_name_char, "Bias_%d_to_Output_%d", j, j);
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
    // [TODO: add method body]
  }

  Model ModelReplicator::copyModel(const Model& model)
  {
    // [TODO: add method body]
    // [TODO: implement copy and copy assignment operators
    //  in model, node, link, and weight classes
    //  see: https://www.geeksforgeeks.org/copy-constructor-vs-assignment-operator-in-c/
    //  but with "rhs" instead of "t"]
  }

  void ModelReplicator::addNode(Model& model)
  {
    // [TODO: add method body]

    // pick a random node from the model
    // that is not an input or bias

    // copy the node, its links, and its bias

  }

  void ModelReplicator::addLink(Model& model)
  {    
    // [TODO: add method body]

    // pick a random source and sink node from the model

    // add link to the model
  }

  void ModelReplicator::deleteNode(Model& model)
  {
    // [TODO: add method body]

  }

  void ModelReplicator::deleteLink(Model& model)
  {
    // [TODO: add method body]

  }

  void ModelReplicator::modifyWeight(Model& model)
  {
    // [TODO: add method body]
    
  }
  
  std::vector<std::string> ModelReplicator::selectNodes(
    const Model& model,
    const std::vector<NodeType>& node_type_exclude,
    const std::vector<NodeType>& node_type_include)
  {
    // populate our list of nodes to select from
    std::vector<std::string> node_ids;
    for (const Node& node : model.getNodes())
    {
      // check the exclusion list
      bool exclude_node = false;
      for (const NodeType& node_type: node_type_exclude)
      {
        if (node_type == node.getType())
        {
          exclude_node = true;
          break;
        }
      }

      // check the inclusion list
      bool include_node = true;
      if (node_type_include.size()>0)
      {
        include_node = false;
        for (const NodeType& node_type: node_type_include)
        {
          if (node_type == node.getType())
          {
            include_node = true;
            break;
          }
        }
      }

      // add the node name to the list
      if (include_node && !exclude_node)
        node_ids.push_back(node.getName());
    }
    return node_ids;
  }

  template<typename T>
  T ModelReplicator::selectRandomElement(std::vector<T> elements)
  {
    // select a random node
    // based on https://www.rosettacode.org/wiki/Pick_random_element
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_int_distribution<int> choose(0, elements.size() - 1);
    return elements[choose(engine)];
  }

  std::string ModelReplicator::selectRandomNode(
    const Model& model,
    const std::vector<NodeType>& node_type_exclude,
    const std::vector<NodeType>& node_type_include,
    const Node& node, 
    const float& distance_weight,
    const std::string& direction)
  {
    // [TODO: add method body]    
  }

  std::string ModelReplicator::selectRandomNode(
    const Model& model,
    const std::vector<NodeType>& node_type_exclude,
    const std::vector<NodeType>& node_type_include)
  {
    std::vector<std::string> node_ids = selectNodes(Model, node_type_exclude, node_type_include);

    if (node_ids.size()>0)
      return selectRandomElement<std::string>(node_ids);
    else
      printf("No nodes were found that matched the inclusion/exclusion criteria");
  }

  std::string ModelReplicator::selectRandomLink(
    const Model& model,
    const std::vector<NodeType>& source_node_type_exclude,
    const std::vector<NodeType>& source_node_type_include,
    const std::vector<NodeType>& sink_node_type_exclude,
    const std::vector<NodeType>& sink_node_type_include)
  {
    // select all source and sink nodes that meet the inclusion/exclusion criteria
    std::vector<std::string> source_node_ids = selectNodes(Model, source_node_type_exclude, source_node_type_include);
    if (source_node_ids.size() == 0)
    {
      printf("No source nodes were found that matched the inclusion/exclusion criteria");
      return;
    }
    std::vector<std::string> sink_node_ids = selectNodes(Model, sink_node_type_exclude, sink_node_type_include);
    if (sink_node_ids.size() == 0)
    {
      printf("No sink nodes were found that matched the inclusion/exclusion criteria");
      return;
    }

    // find all links that have an existing connection with the source and sink node candidates
    std::vector<std::string> link_ids;
    for (const Link& link : model.getLinks())
    {
      if (std::count(source_node_ids.begin(), source_node_ids.end(), link.getSourceNodeName()) != 0)
        if (std::count(sink_node_ids.begin(), sink_node_ids.end(), link.getSinkNodeName()) != 0)
          link_ids.push_back(link.getName());
    }
    
    if (link_ids.size()>0)
      return selectRandomElement<std::string>(link_ids);
    else
      printf("No links were found that matched the node inclusion/exclusion criteria"); 
  }

  void ModelReplicator::addLink(
    const Model& model)
  {
    // define the inclusion/exclusion nodes    
    const std::vector<NodeType> source_node_type_exclude = {};
    const std::vector<NodeType> source_node_type_include = {};
    const std::vector<NodeType> sink_node_type_exclude = {NodeType::bias, NodeType::input};
    const std::vector<NodeType> sink_node_type_include = {};

    // select candidate source and sink nodes
    std::vector<std::string> source_node_ids = selectNodes(Model, source_node_type_exclude, source_node_type_include);
    if (source_node_ids.size() == 0)
    {
      printf("No source nodes were found that matched the inclusion/exclusion criteria"); 
      return;
    }
    std::vector<std::string> sink_node_ids = selectNodes(Model, sink_node_type_exclude, sink_node_type_include);
    if (sink_node_ids.size() == 0)
    {
      printf("No sink nodes were found that matched the inclusion/exclusion criteria"); 
      return;
    }

    // select a random source and sink node
    std::string source_node_name = selectRandomElement<std::string>(source_node_ids);
    std::string sink_node_name = selectRandomElement<std::string>(sink_node_ids);

    // create the new link [TODO: finish...]

    // add the link to the model [TODO: finish...]
  }
}