/**TODO:  Add copyright*/

#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/csv.h>

namespace SmartPeak
{

  LinkFile::LinkFile(){}
  LinkFile::~LinkFile(){}
 
  bool LinkFile::loadLinksBinary(const std::string& filename, std::vector<Link>& links){}

  bool LinkFile::loadLinksCsv(const std::string& filename, std::vector<Link>& links)
  {
    io::CSVReader<4> links_in(filename);
    links_in.read_header(io::ignore_extra_column, 
      "link_name", "source_node_name", "sink_node_name", "weight_name");
    std::string link_name, source_node_name, sink_node_name, weight_name;

    while(links_in.read_row(link_name, source_node_name, sink_node_name, weight_name))
    {
      Link link(link_name, source_node_name, sink_node_name, weight_name);
      links.push_back(link);
    }
  }

  bool LinkFile::storeLinksBinary(const std::string& filename, const std::vector<Link>& links){}

  bool LinkFile::storeLinksCsv(const std::string& filename, const std::vector<Link>& links){}
}