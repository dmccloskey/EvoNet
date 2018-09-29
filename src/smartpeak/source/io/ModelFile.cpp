/**TODO:  Add copyright*/

#include <SmartPeak/io/ModelFile.h>
#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>

namespace SmartPeak
{

  ModelFile::ModelFile(){}
  ModelFile::~ModelFile(){}

	bool ModelFile::loadModelCsv(const std::string & filename_nodes, const std::string & filename_links, const std::string & filename_weights, Model& model)
	{
		// load the nodes
		NodeFile node_file;
		std::vector<Node> nodes;
		node_file.loadNodesCsv(filename_nodes, nodes);

		// load the links
		LinkFile link_file;
		std::vector<Link> links;
		link_file.loadLinksCsv(filename_links, links);

		// load the weights
		WeightFile weight_file;
		std::vector<Weight> weights;
		weight_file.loadWeightsCsv(filename_weights, weights);

		// make the model
		model.addNodes(nodes);
		model.addLinks(links);
		model.addWeights(weights);

		return true;
	}

	bool ModelFile::storeModelDot(const std::string& filename, const Model& model)
	{
		std::fstream file;
		// Open the file in truncate mode
		file.open(filename, std::ios::out | std::ios::trunc);

		file << "digraph G {\n"; // first line

		// write node formating to file
		for (const Node& node : model.getNodes())
		{
			if (node.getType() == NodeType::input)
			{
				char line_char[512];
				sprintf(line_char, "\t\"%s\" [shape=circle,style=filled,color=\"#D3D3D3\"];\n",	node.getName().data());
				std::string line(line_char);
				file << line;
			}
			else if (node.getType() == NodeType::output)
			{
				char line_char[512];
				sprintf(line_char, "\t\"%s\" [shape=circle,style=filled,color=\"#00FFFF\"];\n", node.getName().data());
				std::string line(line_char);
				file << line;
			}
		}

		// write each source/sink to file
		for (const Link& link : model.getLinks())
		{
			if (model.getNode(link.getSourceNodeName()).getType() != NodeType::bias)
			{
				char line_char[512];
				sprintf(line_char, "\t\"%s\" -> \"%s\";\n", link.getSourceNodeName().data(), link.getSinkNodeName().data());
				std::string line(line_char);
				file << line;
			}
		}

		file << "}";  // last line
		file.close();

		return true;
	}
}