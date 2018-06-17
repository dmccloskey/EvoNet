/**TODO:  Add copyright*/

#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

namespace SmartPeak
{

  LinkFile::LinkFile(){}
  LinkFile::~LinkFile(){}
 
  bool LinkFile::loadLinksBinary(const std::string& filename, std::vector<Link>& links) { return true; }

  bool LinkFile::loadLinksCsv(const std::string& filename, std::vector<Link>& links)
  {
    links.clear();

    io::CSVReader<4> links_in(filename);
    links_in.read_header(io::ignore_extra_column, 
      "link_name", "source_node_name", "sink_node_name", "weight_name");
    std::string link_name, source_node_name, sink_node_name, weight_name;

    while(links_in.read_row(link_name, source_node_name, sink_node_name, weight_name))
    {
      Link link(link_name, source_node_name, sink_node_name, weight_name);
      links.push_back(link);
    }
	return true;
  }

  bool LinkFile::storeLinksBinary(const std::string& filename, const std::vector<Link>& links) { return true; }

  bool LinkFile::storeLinksCsv(const std::string& filename, const std::vector<Link>& links)
  {    
    CSVWriter csvwriter(filename);

    // write the headers to the first line
    const std::vector<std::string> headers = {"link_name", "source_node_name", "sink_node_name", "weight_name"};
    csvwriter.writeDataInRow(headers.begin(), headers.end());

    for (const Link& link: links)
    {
      std::vector<std::string> row;
      row.push_back(link.getName());
      row.push_back(link.getSourceNodeName());
      row.push_back(link.getSinkNodeName());
      row.push_back(link.getWeightName());

      // write to file
      csvwriter.writeDataInRow(row.begin(), row.end());
    }
	return true;
  }
}