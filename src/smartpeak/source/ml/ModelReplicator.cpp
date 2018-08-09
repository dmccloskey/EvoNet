/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelReplicator.h>
#include <SmartPeak/core/Preprocessing.h>

#include <random> // random number generator
#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <ctime> // time format
#include <chrono> // current time
#include <set>

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

	void ModelReplicator::setNNodeActivationChanges(const int & n_node_activation_changes)
	{
		n_node_activation_changes_ = n_node_activation_changes;
	}

	void ModelReplicator::setNNodeIntegrationChanges(const int & n_node_integration_changes)
	{
		n_node_integration_changes_ = n_node_integration_changes;
	}

	void ModelReplicator::setNodeActivations(const std::vector<NodeActivation>& node_activations)
	{
		node_activations_ = node_activations;
	}

	void ModelReplicator::setNodeIntegrations(const std::vector<NodeIntegration>& node_integrations)
	{
		node_integrations_ = node_integrations;
	}

	void ModelReplicator::setNModuleAdditions(const int & n_module_additions)
	{
		n_module_additions_ = n_module_additions;
	}

	void ModelReplicator::setNModuleDeletions(const int & n_module_deletions)
	{
		n_module_deletions_ = n_module_deletions;
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

	int ModelReplicator::getNNodeActivationChanges() const
	{
		return n_node_activation_changes_;
	}

	int ModelReplicator::getNNodeIntegrationChanges() const
	{
		return n_node_integration_changes_;
	}

	std::vector<NodeActivation> ModelReplicator::getNodeActivations() const
	{
		return node_activations_;
	}

	std::vector<NodeIntegration> ModelReplicator::getNodeIntegrations() const
	{
		return node_integrations_;
	}

	int ModelReplicator::getNModuleAdditions() const
	{
		return n_module_additions_;
	}

	int ModelReplicator::getNModuleDeletions() const
	{
		return n_module_deletions_;
	}

  int ModelReplicator::getNWeightChanges() const
  {
    return n_weight_changes_;
  }

  float ModelReplicator::getWeightChangeStDev() const
  {
    return weight_change_stdev_;
  }
  
  std::string ModelReplicator::makeUniqueHash(const std::string& left_str, const std::string& right_str)
  {
    std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
    std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
    std::tm now_tm = *std::localtime(&time_now_t);
    char timestamp[64];
    std::strftime(timestamp, 64, "%Y-%m-%d-%H-%M-%S", &now_tm);

    char hash_char[512];
    sprintf(hash_char, "%s_%s_%s", left_str.data(), right_str.data(), timestamp);
    std::string hash_str(hash_char);

    return hash_str;
  }

  Model ModelReplicator::makeBaselineModel(const int& n_input_nodes, const int& n_hidden_nodes, const int& n_output_nodes,
    const NodeActivation& hidden_node_activation, const NodeIntegration& hidden_node_integration,
		const NodeActivation& output_node_activation, const NodeIntegration& output_node_integration,
    const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver,
    const ModelLossFunction& error_function, std::string unique_str)
  {
    Model model;
    model.setLossFunction(error_function);

    std::string model_name = makeUniqueHash("Model", unique_str);
    model.setName(model_name);

    // Create the input nodes
    for (int i=0; i<n_input_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Input_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, NodeType::input, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
      model.addNodes({node});
    }
    // Create the hidden nodes + biases and hidden to bias links
    for (int i=0; i<n_hidden_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Hidden_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, NodeType::hidden, NodeStatus::deactivated, hidden_node_activation, hidden_node_integration);

      char bias_name_char[64];
      sprintf(bias_name_char, "Hidden_bias_%d", i);
      std::string bias_name(bias_name_char);
      Node bias(bias_name, NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
      model.addNodes({node, bias});

      char weight_bias_name_char[64];
      sprintf(weight_bias_name_char, "Bias_%d_to_Hidden_%d", i, i);
      std::string weight_bias_name(weight_bias_name_char);

      char link_bias_name_char[64];
      sprintf(link_bias_name_char, "Bias_%d_to_Hidden_%d", i, i);
      std::string link_bias_name(link_bias_name_char);

      std::shared_ptr<WeightInitOp> bias_weight_init;
      bias_weight_init.reset(new ConstWeightInitOp(1.0));;
      std::shared_ptr<SolverOp> bias_solver = solver;
      Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
      Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);

      model.addWeights({weight_bias});
      model.addLinks({link_bias});
    }
    // Create the output nodes + biases and bias to output link
    for (int i=0; i<n_output_nodes; ++i)
    {
      char node_name_char[64];
      sprintf(node_name_char, "Output_%d", i);
      std::string node_name(node_name_char);
      Node node(node_name, NodeType::output, NodeStatus::deactivated, output_node_activation, output_node_integration);
      
      char bias_name_char[64];
      sprintf(bias_name_char, "Output_bias_%d", i);
      std::string bias_name(bias_name_char);
      Node bias(bias_name, NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
      model.addNodes({node, bias});

      char weight_bias_name_char[64];
      sprintf(weight_bias_name_char, "Bias_%d_to_Output_%d", i, i);
      std::string weight_bias_name(weight_bias_name_char);

      char link_bias_name_char[64];
      sprintf(link_bias_name_char, "Bias_%d_to_Output_%d", i, i);
      std::string link_bias_name(link_bias_name_char);

      std::shared_ptr<WeightInitOp> bias_weight_init;
      bias_weight_init.reset(new ConstWeightInitOp(1.0));
      std::shared_ptr<SolverOp> bias_solver = solver;
      Weight weight_bias(weight_bias_name, bias_weight_init, bias_solver);
      Link link_bias(link_bias_name, bias_name, node_name, weight_bias_name);

      model.addWeights({weight_bias});
      model.addLinks({link_bias});
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

        model.addWeights({weight});
        model.addLinks({link});
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

        model.addWeights({weight});
        model.addLinks({link});
      }
    }

    // Create the weights and links for input to output
    if (n_hidden_nodes == 0)
    {
      for (int i=0; i<n_input_nodes; ++i)
      {
        char input_name_char[64];
        sprintf(input_name_char, "Input_%d", i);
        std::string input_name(input_name_char);

        for (int j=0; j<n_output_nodes; ++j)
        {
          char output_name_char[64];
          sprintf(output_name_char, "Output_%d", j);
          std::string output_name(output_name_char);

          char link_name_char[64];
          sprintf(link_name_char, "Input_%d_to_Output_%d", i, j);
          std::string link_name(link_name_char);

          char weight_name_char[64];
          sprintf(weight_name_char, "Input_%d_to_Output_%d", i, j);
          std::string weight_name(weight_name_char);

          std::shared_ptr<WeightInitOp> output_weight_init = weight_init;
          std::shared_ptr<SolverOp> output_solver = solver;
          Weight weight(weight_name_char, output_weight_init, output_solver);
          Link link(link_name, input_name, output_name, weight_name);

          model.addWeights({weight});
          model.addLinks({link});
        }
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

	std::vector<std::string> ModelReplicator::selectModules(const Model & model, const std::vector<NodeType>& node_type_exclude, const std::vector<NodeType>& node_type_include)
	{
		// populate our list of modules to select from
		std::set<std::string> module_name_set;
		for (const Node& node : model.getNodes())
		{
			// check the exclusion list
			bool exclude_node = false;
			for (const NodeType& node_type : node_type_exclude)
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
				for (const NodeType& node_type : node_type_include)
				{
					if (node_type == node.getType())
					{
						include_node = true;
						break;
					}
				}
			}

			// add the node name to the list
			if (include_node && !exclude_node && !node.getModuleName().empty())
				module_name_set.insert(node.getModuleName());
		}

		std::vector<std::string> module_ids(module_name_set.begin(), module_name_set.end());
		return module_ids;
	}

  //std::string ModelReplicator::selectRandomNode(
  //  const Model& model,
  //  const std::vector<NodeType>& node_type_exclude,
  //  const std::vector<NodeType>& node_type_include,
  //  const Node& node, 
  //  const float& distance_weight,
  //  const std::string& direction)
  //{
  //  // [TODO: add method body]    
  //}

  std::string ModelReplicator::selectRandomNode(
    const Model& model,
    const std::vector<NodeType>& node_type_exclude,
    const std::vector<NodeType>& node_type_include)
  {
    std::vector<std::string> node_ids = selectNodes(model, node_type_exclude, node_type_include);

    if (node_ids.size()>0)
      return selectRandomElement<std::string>(node_ids);
    else
    {
      printf("No nodes were found that matched the inclusion/exclusion criteria.\n");
      return "";
    }
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
      printf("No source nodes were found that matched the inclusion/exclusion criteria.\n");
      return "";
    }
    std::vector<std::string> sink_node_ids = selectNodes(model, sink_node_type_exclude, sink_node_type_include);
    if (sink_node_ids.size() == 0)
    {
      printf("No sink nodes were found that matched the inclusion/exclusion criteria.\n");
      return "";
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
    {
      printf("No links were found that matched the node inclusion/exclusion criteria.\n"); 
      return "";
    }
  }

  void ModelReplicator::addLink(
    Model& model, std::string unique_str)
  {
    // define the inclusion/exclusion nodes
    const std::vector<NodeType> source_node_type_exclude = {NodeType::bias, NodeType::output}; // no output can be a source
    const std::vector<NodeType> source_node_type_include = {};
    const std::vector<NodeType> sink_node_type_exclude = {NodeType::bias, NodeType::input};  // no input can be a sink
    const std::vector<NodeType> sink_node_type_include = {};

    // select candidate source nodes
    std::vector<std::string> source_node_ids = selectNodes(model, source_node_type_exclude, source_node_type_include);
    if (source_node_ids.size() == 0)
    {
      printf("No source nodes were found that matched the inclusion/exclusion criteria.\n"); 
      return;
    }

		// select a random source node
		std::string source_node_name = selectRandomElement<std::string>(source_node_ids);

		// select candidate sink nodes
    std::vector<std::string> sink_node_ids = selectNodes(model, sink_node_type_exclude, sink_node_type_include);
    if (sink_node_ids.size() == 0)
    {
      printf("No sink nodes were found that matched the inclusion/exclusion criteria.\n"); 
      return;
    }

		// remove candidate sink nodes for which a link already exists
		

    // select a random sink node
    std::string sink_node_name = selectRandomElement<std::string>(sink_node_ids);

    // [TODO: Need a check if the link already exists...]

    // create the new weight based on a random link (this can probably be optmized...)
    std::string random_link = selectRandomLink(model, source_node_type_exclude, source_node_type_include, sink_node_type_exclude, sink_node_type_include);

    Weight weight = model.getWeight(model.getLink(random_link).getWeightName()); // copy assignment
    char weight_name_char[128];
		sprintf(weight_name_char, "Weight_%s_to_%s@addLink#", source_node_name.data(), sink_node_name.data());
    std::string weight_name = makeUniqueHash(weight_name_char, unique_str);
    weight.setName(weight_name);
    weight.initWeight();
    model.addWeights({weight});

    // create the new link
    char link_name_char[128];
		sprintf(link_name_char, "Link_%s_to_%s@addLink#", source_node_name.data(), sink_node_name.data());
    std::string link_name = makeUniqueHash(link_name_char, unique_str);
    Link link(link_name, source_node_name, sink_node_name, weight_name);
    model.addLinks({link});
  }

	void ModelReplicator::addModule(Model & model, std::string unique_str)
	{
		// pick a random module from the model
		std::vector<NodeType> node_exclusion_list = {};
		std::vector<NodeType> node_inclusion_list = {};
		std::string random_module_name = selectRandomModule(model, node_exclusion_list, node_inclusion_list);
		if (random_module_name.empty())
		{
			std::cout << "No modules were added to the model." << std::endl;
			return;
		}

		// update the module name [TODO: update the module ID]
		std::string new_name_format = "%s@addModule#";
		std::string new_module_name, module_name_prefix;
		updateName(random_module_name, new_name_format, unique_str, module_name_prefix, new_module_name);

		// copy the module and reconnect the links
		std::vector<Node> new_nodes;
		std::vector<Link> new_links;
		std::vector<Weight> new_weights;
		std::vector<Link> connecting_links;
		std::vector<Weight> connecting_weights;
		for (Link& link : model.getLinks())
		{
			if (link.getModuleName() == random_module_name)
			{ // copy the internal nodes, weights, and links, and give them a new name/id/module_name/module_id
				std::string new_link_name, link_prefix;
				updateName(link.getName(), new_name_format, unique_str, link_prefix, new_link_name);
				link.setName(new_link_name);
				link.setModuleName(new_module_name);
				new_links.push_back(link);

				Node source_node = model.getNode(link.getSourceNodeName());
				std::string new_node_name, node_prefix;
				updateName(source_node.getName(), new_name_format, unique_str, node_prefix, new_node_name);
				source_node.setName(new_node_name);
				source_node.setModuleName(new_module_name);
				new_nodes.push_back(source_node);

				Node sink_node = model.getNode(link.getSinkNodeName());
				std::string new_node_name, node_prefix;
				updateName(sink_node.getName(), new_name_format, unique_str, node_prefix, new_node_name);
				sink_node.setName(new_node_name);
				sink_node.setModuleName(new_module_name);
				new_nodes.push_back(sink_node);

				Weight weight = model.getWeight(link.getWeightName());
				std::string new_weight_name, weight_prefix;
				updateName(weight.getName(), new_name_format, unique_str, weight_prefix, new_weight_name);
				weight.setName(new_weight_name);
				weight.setModuleName(new_module_name);
				new_weights.push_back(weight);
			}
			else if (model.getNode(link.getSourceNodeName()).getModuleName() == random_module_name)
			{ // copy the connecting links and weights, and give them a new name/id
				// and update the source node name (i.e., connect to the new module)
				std::string new_link_name, link_prefix;
				updateName(link.getName(), new_name_format, unique_str, link_prefix, new_link_name);
				link.setName(new_link_name);
				std::string new_node_name, node_prefix;
				updateName(link.getSourceNodeName(), new_name_format, unique_str, node_prefix, new_node_name);
				link.setSinkNodeName(new_node_name);
				connecting_links.push_back(link);

				Weight weight = model.getWeight(link.getWeightName());
				std::string new_weight_name, weight_prefix;
				updateName(weight.getName(), new_name_format, unique_str, weight_prefix, new_weight_name);
				weight.setName(new_weight_name);
				connecting_weights.push_back(weight);
			}
			else if (model.getNode(link.getSinkNodeName()).getModuleName() == random_module_name)
			{ // copy the connecting links and weights, and give them a new name/id
				// and update the sink node name (i.e., connect to the new module)
				std::string new_link_name, link_prefix;
				updateName(link.getName(), new_name_format, unique_str, link_prefix, new_link_name);
				link.setName(new_link_name);
				std::string new_node_name, node_prefix;
				updateName(link.getSinkNodeName(), new_name_format, unique_str, node_prefix, new_node_name);
				link.setSinkNodeName(new_node_name);
				connecting_links.push_back(link);

				Weight weight = model.getWeight(link.getWeightName());
				std::string new_weight_name, weight_prefix;
				updateName(weight.getName(), new_name_format, unique_str, weight_prefix, new_weight_name);
				weight.setName(new_weight_name);
				connecting_weights.push_back(weight);
			}
		}

		// add the new nodes/links/weights to the model
		model.addNodes(new_nodes);
		model.addWeights(new_weights);
		model.addLinks(new_links);
		model.addWeights(connecting_weights);
		model.addLinks(connecting_links);
	}

	std::string ModelReplicator::selectRandomModule(const Model & model, const std::vector<NodeType>& node_type_exclude, const std::vector<NodeType>& node_type_include)
	{
		std::vector<std::string> module_ids = selectModules(model, node_type_exclude, node_type_include);

		if (module_ids.size()>0)
			return selectRandomElement<std::string>(module_ids);
		else
		{
			printf("No nodes were found that matched the inclusion/exclusion criteria.\n");
			return "";
		}
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

  void ModelReplicator::addNode(Model& model, std::string unique_str)
  {
    // pick a random node from the model
    // that is not an input or bias    
    std::vector<NodeType> node_exclusion_list = {NodeType::bias, NodeType::input};
    std::vector<NodeType> node_inclusion_list = {NodeType::hidden, NodeType::output};
    std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);
    if (random_node_name.empty() || random_node_name == "")
    {
      std::cout<<"No nodes were added to the model."<<std::endl;
      return;
    }

    // copy the node
    Node new_node = model.getNode(random_node_name);

    // select a random input link
    // [OPTIMIZATION: refactor to pass back the Link and not just the name]
    std::vector<std::string> input_link_names;
    for (const Link& link: model.getLinks())
    {
      if (link.getSinkNodeName() == random_node_name &&
        model.getNode(link.getSourceNodeName()).getType() != NodeType::bias)
      {
        input_link_names.push_back(link.getName());
      }
    }    
    if (input_link_names.size() == 0)
    {
      std::cout<<"No nodes were added to the model."<<std::endl;
      return;
    }
    std::string input_link_name = selectRandomElement<std::string>(input_link_names);

		std::string new_node_name, add_node_name;
    updateName(random_node_name, "%s@addNode#", unique_str, add_node_name, new_node_name);
    new_node.setName(new_node_name); 
		new_node.setType(NodeType::hidden); // [TODO: add test to check for the type!
    model.addNodes({new_node});

    // create a new bias
    char new_bias_name_char[128];
    sprintf(new_bias_name_char, "Bias_%s@addNode#", add_node_name.data());
    std::string new_bias_name = makeUniqueHash(new_bias_name_char, unique_str);
    Node new_bias(new_bias_name, NodeType::bias, NodeStatus::activated, NodeActivation::Linear, NodeIntegration::Sum);
    new_bias.initNode(new_node.getOutput().dimension(0), new_node.getOutput().dimension(1));
    model.addNodes({new_bias});

    // create a link from the new bias to the new node
    char weight_bias_name_char[512];
    sprintf(weight_bias_name_char, "%s_to_%s@addNode#", new_bias_name.data(), new_node_name.data());
    std::string weight_bias_name = makeUniqueHash(weight_bias_name_char, unique_str);

    char link_bias_name_char[512];
    sprintf(link_bias_name_char, "%s_to_%s@addNode#", new_bias_name.data(), new_node_name.data());
    std::string link_bias_name = makeUniqueHash(link_bias_name_char, unique_str);

    std::shared_ptr<WeightInitOp> bias_weight_init;
    bias_weight_init.reset(new ConstWeightInitOp(1.0));
    Weight weight_bias = model.getWeight(model.getLink(input_link_name).getWeightName()); // [OPTIMIZATION: use Link.getWeightName() directly]
    weight_bias.setName(weight_bias_name);
    weight_bias.setWeightInitOp(bias_weight_init);
    weight_bias.initWeight();
    Link link_bias(link_bias_name, new_bias_name, new_node_name, weight_bias_name);

    model.addWeights({weight_bias});
    model.addLinks({link_bias});
    
    // change the output node name of the link to the new copied node name
    Link modified_link = model.getLink(input_link_name);
    modified_link.setSinkNodeName(new_node_name);
    char modified_link_name_char[512];
    sprintf(modified_link_name_char, "Link_%s_to_%s@addNode#", modified_link.getSourceNodeName().data(), new_node_name.data());
    std::string modified_link_name = makeUniqueHash(modified_link_name_char, unique_str);
    modified_link.setName(modified_link_name); 
    model.addLinks({modified_link});

    // add a new weight that connects the new copied node
    // to its original node
    Weight weight = model.getWeight(model.getLink(input_link_name).getWeightName()); // copy assignment
    char weight_name_char[512];
    sprintf(weight_name_char, "Weight_%s_to_%s@addNode#", new_node_name.data(), random_node_name.data());
    std::string weight_name = makeUniqueHash(weight_name_char, unique_str);
    weight.setName(weight_name);
    weight.initWeight();
    model.addWeights({weight});

    // add a new link that connects the new copied node
    // to its original node
    char link_name_char[512];
    sprintf(link_name_char, "Link_%s_to_%s@addNode#", new_node_name.data(), random_node_name.data());
    std::string link_name = makeUniqueHash(link_name_char, unique_str);
    Link link(link_name, new_node_name, random_node_name, weight_name);
    model.addLinks({link});

    // remove the unmodified link
	// [CHECK: is this needed?  identified as a high CPU call due to prune weights]
    model.removeLinks({input_link_name});  
  }

  void ModelReplicator::deleteNode(Model& model, int prune_iterations)
  {
    // pick a random node from the model
    // that is not an input, bias, nor output
    std::vector<NodeType> node_exclusion_list = {NodeType::bias, NodeType::input, NodeType::output, NodeType::unmodifiable};
    std::vector<NodeType> node_inclusion_list = {NodeType::hidden};
    std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);

    // delete the node, its bias, and its bias link
    if (!random_node_name.empty() || random_node_name != "") // isn't this this same thing?
    {
      // std::cout<<"Random node name: "<<random_node_name<<std::endl;
      model.removeNodes({random_node_name});
      model.pruneModel(prune_iterations);  // this action can remove additional nodes including inputs, biases, and outputs
    }
  }

  void ModelReplicator::deleteLink(Model& model, int prune_iterations)
  {
    // pick a random link from the model
    // that does not connect from a bias or input
    // [TODO: need to implement a check that the deletion does not also remove an input/output node]
    std::vector<NodeType> source_exclusion_list = {NodeType::bias, NodeType::unmodifiable};
    std::vector<NodeType> source_inclusion_list = {};
    std::vector<NodeType> sink_exclusion_list = {NodeType::bias, NodeType::unmodifiable};
    std::vector<NodeType> sink_inclusion_list = {};
    std::string random_link_name = selectRandomLink(
      model, source_exclusion_list, source_inclusion_list, sink_exclusion_list, sink_inclusion_list);

    // delete the link and weight if required
    if (!random_link_name.empty() || random_link_name != "") // isn't this this same thing?
    {
      model.removeLinks({random_link_name});
      model.pruneModel(prune_iterations);  // this action can remove additional nodes including inputs, biases, and outputs
    }
  }

	void ModelReplicator::deleteModule(Model & model, int prune_iterations)
	{
		// pick a random module from the model
		std::vector<NodeType> node_exclusion_list = {};
		std::vector<NodeType> node_inclusion_list = {};
		std::string random_module_name = selectRandomModule(model, node_exclusion_list, node_inclusion_list);
		if (random_module_name.empty())
		{
			std::cout << "No modules were deleted from the model." << std::endl;
			return;
		}

		// remove nodes/link/weights from the model
		std::vector<std::string> delete_nodes;
		std::vector<std::string> delete_links;
		std::vector<std::string> delete_weights;
		for (Link& link : model.getLinks())
		{
			if (link.getModuleName() == random_module_name)
			{
				delete_links.push_back(link.getName());
				delete_nodes.push_back(link.getSourceNodeName());
				delete_nodes.push_back(link.getSinkNodeName());
				delete_weights.push_back(link.getWeightName());
			}
		}
		model.removeNodes(delete_nodes);
		model.removeLinks(delete_links);
		model.removeWeights(delete_weights);

		// prune the model
		model.pruneModel(prune_iterations);  // this action can remove additional nodes including inputs, biases, and outputs
	}

	void ModelReplicator::changeNodeActivation(Model & model, std::string unique_str)
	{
		// pick a random node from the model
		// that is not an input or bias or output
		std::vector<NodeType> node_exclusion_list = { NodeType::bias, NodeType::input, NodeType::output };
		std::vector<NodeType> node_inclusion_list = { NodeType::hidden };
		std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);
		if (random_node_name.empty() || random_node_name == "")
		{
			std::cout << "No node activations were changed in the model." << std::endl;
			return;
		}

		Node new_node = model.getNode(random_node_name); // copy the node		
		NodeActivation new_activation = selectRandomElement(node_activations_); // pick a random activation
		new_node.setActivation(new_activation); // change the activation
		model.removeNodes({new_node.getName()}); // delete the original node
		model.addNodes({new_node}); // add in the new node
	}

	void ModelReplicator::changeNodeIntegration(Model & model, std::string unique_str)
	{
		// pick a random node from the model
		// that is not an input or bias or output
		std::vector<NodeType> node_exclusion_list = { NodeType::bias, NodeType::input, NodeType::output };
		std::vector<NodeType> node_inclusion_list = { NodeType::hidden };
		std::string random_node_name = selectRandomNode(model, node_exclusion_list, node_inclusion_list);
		if (random_node_name.empty() || random_node_name == "")
		{
			std::cout << "No node activations were changed in the model." << std::endl;
			return;
		}

		Node new_node = model.getNode(random_node_name); // copy the node		
		NodeIntegration new_integration = selectRandomElement(node_integrations_); // pick a random integration
		new_node.setIntegration(new_integration); // change the integration
		model.removeNodes({ new_node.getName() }); // delete the original node
		model.addNodes({ new_node }); // add in the new node
	}

  void ModelReplicator::modifyWeight(Model& model)
  {
    // [TODO: add method body]    

    // select a random link from the model
    
    // change the weight

    // update the link's weight name

    // add the new weight back into the model
    // delete the previous weight

  }
  
	void ModelReplicator::updateName(const std::string & name, const std::string & new_name_format, std::string unique_str,
		std::string& name_prefix, std::string& new_name)
	{
		std::regex re("@");
		std::vector<std::string> str_tokens;
		name_prefix = name;
		std::copy(
			std::sregex_token_iterator(name.begin(), name.end(), re, -1),
			std::sregex_token_iterator(),
			std::back_inserter(str_tokens));
		if (str_tokens.size() > 1)
			name_prefix = str_tokens[0]; // only retain the last timestamp
																		 // printf("New node name: %s\n", add_name.data());

		char new_name_char[128];
		sprintf(new_name_char, new_name_format.data(), name_prefix.data());
		new_name = makeUniqueHash(new_name_char, unique_str);
	}

	std::vector<std::string> ModelReplicator::makeRandomModificationOrder()
  {
    // create the list of modifications
    std::vector<std::string> modifications;
		for (int i = 0; i<n_node_activation_changes_; ++i) modifications.push_back("change_node_activation");
		for (int i = 0; i<n_node_integration_changes_; ++i) modifications.push_back("change_node_integration");
    for(int i=0; i<n_node_additions_; ++i) modifications.push_back("add_node");
    for(int i=0; i<n_link_additions_; ++i) modifications.push_back("add_link");
		for (int i = 0; i<n_module_additions_; ++i) modifications.push_back("add_module");
    for(int i=0; i<n_node_deletions_; ++i) modifications.push_back("delete_node");
    for(int i=0; i<n_link_deletions_; ++i) modifications.push_back("delete_link");
		for (int i = 0; i<n_module_deletions_; ++i) modifications.push_back("delete_module");

    // // randomize
    // std::random_device seed;
    // std::mt19937 engine(seed());
    // std::shuffle(modifications.begin(), modifications.end(), engine);

    return modifications;
  }

	void ModelReplicator::setRandomModifications(
		const std::pair<int, int>& node_additions,
		const std::pair<int, int>& link_additions,
		const std::pair<int, int>& node_deletions,
		const std::pair<int, int>& link_deletions,
		const std::pair<int, int>& node_activation_changes,
		const std::pair<int, int>& node_integration_changes,
		const std::pair<int, int>& module_additions,
		const std::pair<int, int>& module_deletions)
	{
		// set 
		node_additions_ = node_additions;
		link_additions_ = link_additions;
		node_deletions_ = node_deletions;
		link_deletions_ = link_deletions;
		node_activation_changes_ = node_activation_changes;
		node_integration_changes_ = node_integration_changes;
		module_additions_ = module_additions;
		module_deletions_ = module_deletions;
	}

	void ModelReplicator::makeRandomModifications()
	{
		// random generator for model modifications
		std::random_device rd;
		std::mt19937 gen(rd());

		// set 
		std::uniform_int_distribution<> node_addition_gen(node_additions_.first, node_additions_.second);
		setNNodeAdditions(node_addition_gen(gen));
		std::uniform_int_distribution<> link_addition_gen(link_additions_.first, link_additions_.second);
		setNLinkAdditions(link_addition_gen(gen));
		std::uniform_int_distribution<> node_deletion_gen(node_deletions_.first, node_deletions_.second);
		setNNodeDeletions(node_deletion_gen(gen));
		std::uniform_int_distribution<> link_deletion_gen(link_deletions_.first, link_deletions_.second);
		setNLinkDeletions(link_deletion_gen(gen));
		std::uniform_int_distribution<> node_activation_changes_gen(node_activation_changes_.first, node_activation_changes_.second);
		setNNodeActivationChanges(node_activation_changes_gen(gen));
		std::uniform_int_distribution<> node_integration_changes_gen(node_integration_changes_.first, node_integration_changes_.second);
		setNNodeIntegrationChanges(node_integration_changes_gen(gen));
		std::uniform_int_distribution<> module_addition_gen(module_additions_.first, module_additions_.second);
		setNNodeAdditions(module_addition_gen(gen));
		std::uniform_int_distribution<> module_deletion_gen(module_deletions_.first, module_deletions_.second);
		setNNodeDeletions(module_deletion_gen(gen));
	}

  void ModelReplicator::modifyModel(Model& model, std::string unique_str)
  {
    // randomly order the modifications
    std::vector<std::string> modifications = makeRandomModificationOrder();

    // implement each modification one at a time
    // and track the counts that each modification is called
    std::map<std::string, int> modifications_counts;
    for (const std::string& modification: modifications)
      modifications_counts.emplace(modification, 0);

    const int prune_iterations = 1e6;
    for (const std::string& modification: modifications)
    {
      // [TODO: copyNode]
      if (modification == "add_node")
      {
        addNode(model, unique_str + "-" + std::to_string(modifications_counts.at(modification)));
        modifications_counts[modification] += 1;
      }
      else if (modification == "add_link")
      {
        addLink(model, unique_str + "-" + std::to_string(modifications_counts.at(modification)));
        modifications_counts[modification] += 1;
      }
      else if (modification == "delete_node")
      {
        deleteNode(model, prune_iterations);
        modifications_counts[modification] += 1;
      }
      else if (modification == "delete_link")
      {
        deleteLink(model, prune_iterations);
        modifications_counts[modification] += 1;
      }
			else if (modification == "change_node_activation")
			{
				changeNodeActivation(model);
				modifications_counts[modification] += 1;
			}
			else if (modification == "change_node_integration")
			{
				changeNodeIntegration(model);
				modifications_counts[modification] += 1;
			}
			if (modification == "add_module")
			{
				addModule(model, unique_str + "-" + std::to_string(modifications_counts.at(modification)));
				modifications_counts[modification] += 1;
			}
			else if (modification == "delete_module")
			{
				deleteModule(model, prune_iterations);
				modifications_counts[modification] += 1;
			}
      // [TODO: modifyWeight]
    }
  }
}