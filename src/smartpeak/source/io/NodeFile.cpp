/**TODO:  Add copyright*/

#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

namespace SmartPeak
{

  NodeFile::NodeFile(){}
  NodeFile::~NodeFile(){}
 
  bool NodeFile::loadNodesBinary(const std::string& filename, std::vector<Node>& nodes){}

  bool NodeFile::loadNodesCsv(const std::string& filename, std::vector<Node>& nodes)
  {
    nodes.clear();

    io::CSVReader<4> nodes_in(filename);
    nodes_in.read_header(io::ignore_extra_column, 
      "node_name", "node_type", "node_status", "node_activation");
    std::string node_name, node_type_str, node_status_str, node_activation_str;

    while(nodes_in.read_row(node_name, node_type_str, node_status_str,node_activation_str))
    {
      // parse the node_type
      NodeType node_type;
      if (node_type_str == "hidden") node_type = NodeType::hidden;
      else if (node_type_str == "output") node_type = NodeType::output;
      else if (node_type_str == "input") node_type = NodeType::input;
      else if (node_type_str == "bias") node_type = NodeType::bias;
      else std::cout<<"NodeType for node_name "<<node_name<<" was not recognized."<<std::endl;

      // parse the node_status
      NodeStatus node_status;
      if (node_status_str == "deactivated") node_status = NodeStatus::deactivated;
      else if (node_status_str == "initialized") node_status = NodeStatus::initialized;
      else if (node_status_str == "activated") node_status = NodeStatus::activated;
      else if (node_status_str == "corrected") node_status = NodeStatus::corrected;
      else std::cout<<"NodeStatus for node_name "<<node_name<<" was not recognized."<<std::endl;

      // parse the node_activation
      NodeActivation node_activation;
      if (node_activation_str == "ReLU") node_activation = NodeActivation::ReLU;
      else if (node_activation_str == "ELU") node_activation = NodeActivation::ELU;
      else if (node_activation_str == "Linear") node_activation = NodeActivation::Linear;
      else if (node_activation_str == "Sigmoid") node_activation = NodeActivation::Sigmoid;
      else if (node_activation_str == "TanH") node_activation = NodeActivation::TanH;
      else std::cout<<"NodeActivation for node_name "<<node_name<<" was not recognized."<<std::endl;
      
      Node node(node_name, node_type, node_status, node_activation);
      nodes.push_back(node);
    }
  }

  bool NodeFile::storeNodesBinary(const std::string& filename, const std::vector<Node>& nodes){}

  bool NodeFile::storeNodesCsv(const std::string& filename, const std::vector<Node>& nodes)
  {
    CSVWriter csvwriter(filename);

    // write the headers to the first line
    const std::vector<std::string> headers = {"node_name", "node_type", "node_status", "node_activation"};
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
      std::string node_activation_str = "";
      if (node.getActivation() == NodeActivation::ReLU) node_activation_str = "ReLU";
      else if (node.getActivation() == NodeActivation::ELU) node_activation_str = "ELU";
      else if (node.getActivation() == NodeActivation::Linear) node_activation_str = "Linear";
      else if (node.getActivation() == NodeActivation::Sigmoid) node_activation_str = "Sigmoid";
      else if (node.getActivation() == NodeActivation::TanH) node_activation_str = "TanH";
      else std::cout<<"NodeActivation for node_name "<<node.getName()<<" was not recognized."<<std::endl;
      row.push_back(node_activation_str);

      // write to file
      csvwriter.writeDataInRow(row.begin(), row.end());
    }
  }
}