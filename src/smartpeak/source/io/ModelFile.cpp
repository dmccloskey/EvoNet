/**TODO:  Add copyright*/

#include <SmartPeak/io/ModelFile.h>

namespace SmartPeak
{

  ModelFile::ModelFile(){}
  ModelFile::~ModelFile(){}

	bool ModelFile::storeModelDot(const std::string& filename, const Model& model)
	{
		std::fstream file;
		// Open the file in truncate mode
		file.open(filename, std::ios::out | std::ios::trunc);

		file << "digraph G {\n"; // first line

		// write each source/sink to file
		for (const Link& link : model.getLinks())
		{
			char line_char[512];
			// [TODO: check if source node is input, fill node color light grey
			// check if sink node is output, fill node color light blue
			// check if source is a bias, ignore]
			// [TODO: How to include a "" around each node name?]
			sprintf(line_char, "\t%s -> %s;\n", link.getSourceNodeName().data(), link.getSinkNodeName().data());
			// [TODO: include name of the link]
			std::string line(line_char);
			file << line;
		}

		file << "}";  // last line
		file.close();

		return true;
	}
}