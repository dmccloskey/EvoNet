/**TODO:  Add copyright*/

#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/csv.h>
#include <SmartPeak/io/CSVWriter.h>

namespace SmartPeak
{ 
  bool LinkFile::loadLinksBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links) { return true; }

  bool LinkFile::loadLinksCsv(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links)
  {
    links.clear();

    io::CSVReader<5> links_in(filename);
    links_in.read_header(io::ignore_extra_column, 
      "link_name", "source_node_name", "sink_node_name", "weight_name", "module_name");
    std::string link_name, source_node_name, sink_node_name, weight_name, module_name = "";

    while(links_in.read_row(link_name, source_node_name, sink_node_name, weight_name, module_name))
    {
			std::shared_ptr<Link> link(new Link(link_name, source_node_name, sink_node_name, weight_name));
			link->setModuleName(module_name);
      links.emplace(link_name, link);
    }
	return true;
  }

  bool LinkFile::storeLinksBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links) { return true; }

  bool LinkFile::storeLinksCsv(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links)
  {    
    CSVWriter csvwriter(filename);

    // write the headers to the first line
    const std::vector<std::string> headers = {"link_name", "source_node_name", "sink_node_name", "weight_name", "module_name" };
    csvwriter.writeDataInRow(headers.begin(), headers.end());

    for (const auto& link: links)
    {
      std::vector<std::string> row;
      row.push_back(link.second->getName());
      row.push_back(link.second->getSourceNodeName());
      row.push_back(link.second->getSinkNodeName());
      row.push_back(link.second->getWeightName());
			row.push_back(link.second->getModuleName());

      // write to file
      csvwriter.writeDataInRow(row.begin(), row.end());
    }
		return true;
  }
}