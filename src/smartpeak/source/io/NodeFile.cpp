/**TODO:  Add copyright*/

#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

namespace SmartPeak
{

  NodeFile::NodeFile(){}
  NodeFile::~NodeFile(){}
 
  bool NodeFile::loadNodesBinary(const std::string& filename, std::vector<Node>& nodes) { return true; }

  bool NodeFile::loadNodesCsv(const std::string& filename, std::vector<Node>& nodes)
  {
    nodes.clear();

    io::CSVReader<8> nodes_in(filename);
    nodes_in.read_header(io::ignore_extra_column, 
      "node_name", "node_type", "node_status", "node_activation", "node_activation_grad", "node_integration", "node_integration_error", "node_integration_weight_grad");
    std::string node_name, node_type_str, node_status_str, node_activation_str, node_activation_grad_str, node_integration_str, node_integration_error_str, node_integration_weight_grad_str;

    while(nodes_in.read_row(node_name, node_type_str, node_status_str, node_activation_str, node_activation_grad_str, node_integration_str, node_integration_error_str, node_integration_weight_grad_str))
    {
      // parse the node_type
      NodeType node_type;
      if (node_type_str == "hidden") node_type = NodeType::hidden;
      else if (node_type_str == "output") node_type = NodeType::output;
      else if (node_type_str == "input") node_type = NodeType::input;
      else if (node_type_str == "bias") node_type = NodeType::bias;
			else if (node_type_str == "recursive") node_type = NodeType::recursive;
      else std::cout<<"NodeType for node_name "<<node_name<<" was not recognized."<<std::endl;

      // parse the node_status
      NodeStatus node_status;
      if (node_status_str == "deactivated") node_status = NodeStatus::deactivated;
      else if (node_status_str == "initialized") node_status = NodeStatus::initialized;
      else if (node_status_str == "activated") node_status = NodeStatus::activated;
      else if (node_status_str == "corrected") node_status = NodeStatus::corrected;
      else std::cout<<"NodeStatus for node_name "<<node_name<<" was not recognized."<<std::endl;

      // parse the node_activation
			std::shared_ptr<ActivationOp<float>> node_activation;
      if (node_activation_str == "ReLUOp") node_activation.reset(new ReLUOp<float>());
      else if (node_activation_str == "ELUOp") node_activation.reset(new ELUOp<float>());
      else if (node_activation_str == "LinearOp") node_activation.reset(new LinearOp<float>());
      else if (node_activation_str == "SigmoidOp") node_activation.reset(new SigmoidOp<float>());
      else if (node_activation_str == "TanHOp") node_activation.reset(new TanHOp<float>());
      else std::cout<<"NodeActivation for node_name "<<node_name<<" was not recognized."<<std::endl;

			// parse the node_activation
			std::shared_ptr<ActivationOp<float>> node_activation_grad;
			if (node_activation_grad_str == "ReLUGradOp") node_activation_grad.reset(new ReLUGradOp<float>());
			else if (node_activation_grad_str == "ELUGradOp") node_activation_grad.reset(new ELUGradOp<float>());
			else if (node_activation_grad_str == "LinearGradOp") node_activation_grad.reset(new LinearGradOp<float>());
			else if (node_activation_grad_str == "SigmoidGradOp") node_activation_grad.reset(new SigmoidGradOp<float>());
			else if (node_activation_grad_str == "TanHGradOp") node_activation_grad.reset(new TanHGradOp<float>());
			else std::cout << "NodeActivationGrad for node_name " << node_name << " was not recognized." << std::endl;

			// parse the node_integration
			std::shared_ptr<IntegrationOp<float>> node_integration;
			if (node_integration_str == "SumOp") node_integration.reset(new SumOp<float>());
			else if (node_integration_str == "ProdOp") node_integration.reset(new ProdOp<float>());
			else if (node_integration_str == "MaxOp") node_integration.reset(new MaxOp<float>());
			else std::cout << "NodeIntegration for node_name " << node_name << " was not recognized." << std::endl;

			// parse the node_integration_error
			std::shared_ptr<IntegrationErrorOp<float>> node_integration_error;
			if (node_integration_error_str == "SumErrorOp") node_integration_error.reset(new SumErrorOp<float>());
			else if (node_integration_error_str == "ProdErrorOp") node_integration_error.reset(new ProdErrorOp<float>());
			else if (node_integration_error_str == "MaxErrorOp") node_integration_error.reset(new MaxErrorOp<float>());
			else std::cout << "NodeIntegrationError for node_name " << node_name << " was not recognized." << std::endl;

			// parse the node_integration_weight_grad
			std::shared_ptr<IntegrationWeightGradOp<float>> node_integration_weight_grad;
			if (node_integration_weight_grad_str == "SumWeightGradOp") node_integration_weight_grad.reset(new SumWeightGradOp<float>());
			else if (node_integration_weight_grad_str == "ProdWeightGradOp") node_integration_weight_grad.reset(new ProdWeightGradOp<float>());
			else if (node_integration_weight_grad_str == "MaxWeightGradOp") node_integration_weight_grad.reset(new MaxWeightGradOp<float>());
			else std::cout << "NodeIntegrationWeightGrad for node_name " << node_name << " was not recognized." << std::endl;

      Node node(node_name, node_type, node_status, node_activation, node_activation_grad, node_integration, node_integration_error, node_integration_weight_grad);
      nodes.push_back(node);
    }
	return true;
  }

  bool NodeFile::storeNodesBinary(const std::string& filename, const std::vector<Node>& nodes) { return true; }

  bool NodeFile::storeNodesCsv(const std::string& filename, const std::vector<Node>& nodes)
  {
    CSVWriter csvwriter(filename);

    // write the headers to the first line
    const std::vector<std::string> headers = {"node_name", "node_type", "node_status", "node_activation", "node_activation_grad", "node_integration", "node_integration_error", "node_integration_weight_grad" };
    csvwriter.writeDataInRow(headers.begin(), headers.end());

    for (const Node& node: nodes)
    {
      std::vector<std::string> row;
      row.push_back(node.getName());
      
      // parse the node_type
      std::string node_type_str = "";
      if (node.getType() == NodeType::hidden) node_type_str = "hidden";
      else if (node.getType() == NodeType::output) node_type_str = "output";
      else if (node.getType() == NodeType::input) node_type_str = "input";
      else if (node.getType() == NodeType::bias) node_type_str = "bias";
			else if (node.getType() == NodeType::recursive) node_type_str = "recursive";
      else std::cout<<"NodeType for node_name "<<node.getName()<<" was not recognized."<<std::endl;
      row.push_back(node_type_str);

      // parse the node_status
      std::string node_status_str = "";
      if (node.getStatus() == NodeStatus::deactivated) node_status_str = "deactivated";
      else if (node.getStatus() == NodeStatus::initialized) node_status_str = "initialized";
      else if (node.getStatus() == NodeStatus::activated) node_status_str = "activated";
      else if (node.getStatus() == NodeStatus::corrected) node_status_str = "corrected";
      else std::cout<<"NodeStatus for node_name "<<node.getName()<<" was not recognized."<<std::endl;
      row.push_back(node_status_str);

      // parse the node_activation
			std::string node_activation_str = node.getActivation()->getName();
			row.push_back(node_activation_str);
			std::string node_activation_grad_str = node.getActivationGrad()->getName();
			row.push_back(node_activation_grad_str);

			// parse the node_integration
			std::string node_integration_str = node.getIntegration()->getName();
			row.push_back(node_integration_str);
			std::string node_integration_error_str = node.getIntegrationError()->getName();
			row.push_back(node_integration_error_str);
			std::string node_integration_weight_grad_str = node.getIntegrationWeightGrad()->getName();
			row.push_back(node_integration_weight_grad_str);

      // write to file
      csvwriter.writeDataInRow(row.begin(), row.end());
    }
	return true;
  }
}