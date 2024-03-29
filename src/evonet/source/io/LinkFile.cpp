/**TODO:  Add copyright*/

#include <EvoNet/io/LinkFile.h>
#include <EvoNet/io/csv.h>
#include <EvoNet/io/CSVWriter.h>
#include <cereal/archives/binary.hpp>
#include <fstream>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>

namespace EvoNet
{ 
  bool LinkFile::loadLinksBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links) {

		std::ofstream ofs(filename, std::ios::binary); 
		if (ofs.is_open() == false) {
			cereal::BinaryOutputArchive oarchive(ofs);
			oarchive(links); 
			ofs.close();
		}
		return true; 
	}

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

  bool LinkFile::storeLinksBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links) { 
		std::ifstream ifs(filename, std::ios::binary);
		if (ifs.is_open()) {
			cereal::BinaryInputArchive iarchive(ifs);
			iarchive(links);
			ifs.close();
		}return true; 
	}

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