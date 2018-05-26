/**TODO:  Add copyright*/

#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/csv.h>

namespace SmartPeak
{

  NodeFile::NodeFile(){}
  NodeFile::~NodeFile(){}
 
  bool NodeFile::loadNodesBinary(const std::string& filename, std::vector<Node>& nodes){}

  bool NodeFile::loadNodesCsv(const std::string& filename, std::vector<Node>& nodes)
  {
    io::CSVReader<3> nodes_in(filename);
    nodes_in.read_header(io::ignore_extra_column, 
      "node_name", "node_type", "node_status");
    std::string node_name, node_type_str, node_status_str;

    while(nodes_in.read_row(node_name, node_type_str, node_status_str))
    {
      // parse the node_type
      NodeType node_type;
      if (node_type_str == "ReLU") node_type = NodeType::ReLU;
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
  }

  bool NodeFile::storeNodesBinary(const std::string& filename, const std::vector<Node>& nodes){}
  bool NodeFile::storeNodesCsv(const std::string& filename, const std::vector<Node>& nodes){}
}