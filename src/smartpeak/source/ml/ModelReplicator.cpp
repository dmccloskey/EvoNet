/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelReplicator.h>

#include <random> // random number geenrator

#include <ctime> // time format
#include <chrono> // current time

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
    const NodeActivation& hidden_node_activation, const NodeActivation& output_node_activation,
    const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver)
  {
    Model model;

    // Create the input nodes
    for (int i=0; i<n_input_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Input_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, NodeType::input, NodeStatus::activated, NodeActivation::Linear);
      model.addNodes({node});
    }
    // Create the hidden nodes + biases
    for (int i=0; i<n_hidden_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Hidden_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, NodeType::hidden, NodeStatus::deactivated, hidden_node_activation);

      char bias_name_char[64];
      sprintf(bias_name_char, "Hidden_bias_%d", i);
      std::string bias_name(bias_name_char);
      Node bias(bias_name, NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
      model.addNodes({node, bias});
    }
    // Create the output nodes + biases
    for (int i=0; i<n_output_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Output_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, NodeType::output, NodeStatus::deactivated, output_node_activation);
      
      char bias_name_char[64];
      sprintf(bias_name_char, "Output_bias_%d", i);
      std::string bias_name(bias_name_char);
      Node bias(bias_name, NodeType::bias, NodeStatus::activated, NodeActivation::Linear);
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
    std::vector<std::string> node_ids = selectNodes(model, node_type_exclude, node_type_include);

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
    std::vector<std::string> source_node_ids = selectNodes(model, source_node_type_exclude, source_node_type_include);
    if (source_node_ids.size() == 0)
    {
      printf("No source nodes were found that matched the inclusion/exclusion criteria");
    }
    std::vector<std::string> sink_node_ids = selectNodes(model, sink_node_type_exclude, sink_node_type_include);
    if (sink_node_ids.size() == 0)
    {
      printf("No sink nodes were found that matched the inclusion/exclusion criteria");
    }

    // find all links that have an existing connection with the source and sink node candidates
    std::vector<std::string> link_ids;
    for (const Link& link : model.getLinks())
    {
      if (std::count(source_node_ids.begin(), source_node_ids.end(), link.getSourceNodeName()) != 0)
        if (std::count(sink_node_ids.begin(), sink_node_ids.end(), link.getSinkNodeName()) != 0)
          link_ids.push_back(link.getName());
    }

    // [TODO: break into seperate method here for testing purposes]
    
    if (link_ids.size()>0)
      return selectRandomElement<std::string>(link_ids);
    else
      printf("No links were found that matched the node inclusion/exclusion criteria"); 
  }

  void ModelReplicator::addLink(
    Model& model)
  {
    // define the inclusion/exclusion nodes    
    const std::vector<NodeType> source_node_type_exclude = {NodeType::bias};
    const std::vector<NodeType> source_node_type_include = {};
    const std::vector<NodeType> sink_node_type_exclude = {NodeType::bias, NodeType::input};
    const std::vector<NodeType> sink_node_type_include = {};

    // select candidate source and sink nodes
    std::vector<std::string> source_node_ids = selectNodes(model, source_node_type_exclude, source_node_type_include);
    if (source_node_ids.size() == 0)
    {
      printf("No source nodes were found that matched the inclusion/exclusion criteria"); 
    }
    std::vector<std::string> sink_node_ids = selectNodes(model, sink_node_type_exclude, sink_node_type_include);
    if (sink_node_ids.size() == 0)
    {
      printf("No sink nodes were found that matched the inclusion/exclusion criteria"); 
    }

    // select a random source and sink node
    std::string source_node_name = selectRandomElement<std::string>(source_node_ids);
    std::string sink_node_name = selectRandomElement<std::string>(sink_node_ids);

    // [TODO: Need a check if the link already exists...]

    // generate a current time-stamp to avoid duplicate name additions
    // [TODO: refactor to its own method for testing purposes]
    std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
    std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
    std::tm now_tm = *std::localtime(&time_now_t);
    char timestamp[64];
    std::strftime(timestamp, 64, "%Y-%m-%d-%H-%M-%S", &now_tm);

    // create the new weight based on a random link (this can probably be optmized...)
    std::string random_link = selectRandomLink(model, source_node_type_exclude, source_node_type_include, sink_node_type_exclude, sink_node_type_include);

    Weight weight = model.getWeight(model.getLink(random_link).getWeightName()); // copy assignment
    char weight_name_char[128];
    sprintf(weight_name_char, "Weight_%s_to_%s@addLink:%s", source_node_name.data(), sink_node_name.data(), timestamp);
    std::string weight_name(weight_name_char);
    weight.setName(weight_name);
    weight.initWeight();
    model.addWeights({weight});

    // create the new link
    char link_name_char[128];
    sprintf(link_name_char, "Link_%s_to_%s@addLink:%s", source_node_name.data(), sink_node_name.data(), timestamp);
    std::string link_name(link_name_char);
    Link link(link_name, source_node_name, sink_node_name, weight_name);
    model.addLinks({link});
  }

  void ModelReplicator::copyNode(Model& model)
  {
    // [TODO: add method body]

    // pick a random node from the model
    // that is not an input or bias

    // copy the node, its links, and its bias

    // add a new link that connects the new copied node
    // to its original node

  }

  void ModelReplicator::addNode(Model& model)
  {
    // [TODO: add tests]

    // pick a random node from the model
    // that is not an input or bias    
    std::vector<NodeType> node_exclusion_list = {NodeType::bias, NodeType::input};
    std::vector<NodeType> node_inclusion_list = {};
    std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);

    // copy the node and its bias
    Node new_node = model.getNode(random_node_name);

    // select a random input link
    std::vector<std::string> input_link_names;
    for (const Link& link: model.getLinks())
    {
      if (link.getSinkNodeName() == new_node.getName())
      {
        input_link_names.push_back(link.getName());
      }
    }
    std::string input_link_name = selectRandomElement<std::string>(input_link_names);    

    // generate a current time-stamp to avoid duplicate name additions
    // [TODO: refactor to its own method for testing purposes]
    std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
    std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
    std::tm now_tm = *std::localtime(&time_now_t);
    char timestamp[64];
    std::strftime(timestamp, 64, "%Y-%m-%d-%H-%M-%S", &now_tm);
    
    // [TODO: change its iteraction probability?]
    // update the copied node name and add it to the model
    char new_node_name_char[128];
    sprintf(new_node_name_char, "%s@addNode:%s", random_node_name.data(), timestamp);
    std::string new_node_name(new_node_name_char);
    new_node.setName(new_node_name);
    
    // change the output node name of the link to the new copied node name
    Link modified_link = model.getLink(input_link_name);
    modified_link.setSinkNodeName(new_node_name);
    char modified_link_name_char[128];
    sprintf(modified_link_name_char, "Link_%s_to_%s@addNode:%s", modified_link.getSourceNodeName().data(), new_node_name, timestamp);
    std::string modified_link_name(modified_link_name_char);
    modified_link.setName(modified_link_name);
    model.addLinks({modified_link});

    // remove the unmodified link
    model.removeLinks({input_link_name});

    // add a new weight that connects the new copied node
    // to its original node
    Weight weight = model.getWeight(model.getLink(input_link_name).getWeightName()); // copy assignment
    char weight_name_char[128];
    sprintf(weight_name_char, "Weight_%s_to_%s@addNode:%s", new_node_name.data(), random_node_name.data(), timestamp);
    std::string weight_name(weight_name_char);
    weight.setName(weight_name);
    weight.initWeight();
    model.addWeights({weight});

    // add a new link that connects the new copied node
    // to its original node
    char link_name_char[128];
    sprintf(link_name_char, "Link_%s_to_%s@addNode:%s", new_node_name.data(), random_node_name.data(), timestamp);
    std::string link_name(link_name_char);
    Link link(link_name, new_node_name, random_node_name, weight_name);
    model.addLinks({link});
  }

  void ModelReplicator::deleteNode(Model& model)
  {
    // [TODO: add tests]
    // [TODO: need to change NodeType to input, bias, hidden, or output]
    // [TODO: need to add ActivationType to ReLU, Sigmoid, etc,....]

    // pick a random node from the model
    // that is not an input, bias, [TODO: nor output]
    std::vector<NodeType> node_exclusion_list = {NodeType::bias, NodeType::input};
    std::vector<NodeType> node_inclusion_list = {};
    std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);

    // delete the node, its bias, and its bias link
    model.removeNodes({random_node_name});

  }

  void ModelReplicator::deleteLink(Model& model)
  {
    // [TODO: add tests]

    // pick a random link from the model
    // that does not connect from a bias or input
    // [TODO: and does not connect to an output]
    std::vector<NodeType> source_exclusion_list = {NodeType::bias, NodeType::input};
    std::vector<NodeType> source_inclusion_list = {};
    std::vector<NodeType> sink_exclusion_list = {NodeType::bias, NodeType::input};
    std::vector<NodeType> sink_inclusion_list = {};
    std::string random_link = selectRandomLink(
      model, source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list);

    // delete the link and weight if required    
    model.removeLinks({random_link});

  }

  void ModelReplicator::modifyWeight(Model& model)
  {
    // [TODO: add method body]    
  }

  void ModelReplicator::modifyModel(Model& model)
  {
    // [TODO: add method body]
  }

  Model ModelReplicator::copyModel(const Model& model)
  {
    // [TODO: add method body]
  }
}