/**TODO:  Add copyright*/

#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/csv.h>

namespace SmartPeak
{

  NodeFile::NodeFile(){}
  NodeFile::~NodeFile(){}
 
  bool NodeFile::loadNodeBinary(const std::string& filename, std::vector<Node>& nodes){}
  bool NodeFile::loadNodeCsv(const std::string& filename, std::vector<Node>& nodes){}
  bool NodeFile::storeNodeBinary(const std::string& filename, const std::vector<Node>& nodes){}
  bool NodeFile::storeNodeCsv(const std::string& filename, const std::vector<Node>& nodes){}
}