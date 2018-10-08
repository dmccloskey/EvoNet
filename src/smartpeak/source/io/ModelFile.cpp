/**TODO:  Add copyright*/

#include <SmartPeak/io/ModelFile.h>
#include <SmartPeak/io/NodeFile.h>
#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/LinkFile.h>

//#include <filesystem> // C++ 17

namespace SmartPeak
{
	template<typename TensorT>
	bool ModelFile<TensorT>::storeModelBinary(const std::string & filename, const Model<TensorT>& model)
	{
		auto myfile = std::fstream(filename, std::ios::out | std::ios::binary);
		myfile.write((char*)&model, sizeof(model));
		myfile.close();
		return true;
	}

	template<typename TensorT>
	bool ModelFile<TensorT>::loadModelBinary(const std::string & filename, Model<TensorT>& model)
	{
		// C++17
		//std::uintmax_t file_size = std::filesystem::file_size(filename); 
		//auto myfile = std::fstream(filename, std::ios::in | std::ios::binary);
		//myfile.read((char*)&model, file_size);
		//myfile.close();

		// using the C stat header
		//struct stat results;
		//int err = stat(filename.data(), &results);
		//std::uintmax_t file_size = results.st_size;

		auto myfile = std::fstream(filename, std::ios::in | std::ios::binary | std::ios::ate);
		std::uintmax_t file_size = myfile.tellg();
		myfile.seekg(0, std::ios::beg);
		myfile.read((char*)&model, file_size);
		myfile.close();
		return true;
	}

	template<typename TensorT>
	bool ModelFile<TensorT>::storeModelCsv(const std::string & filename_nodes, const std::string & filename_links, const std::string & filename_weights, const Model<TensorT>& model)
	{
		NodeFile node_file;
		node_file.storeNodesCsv(filename_nodes, model.getNodes());
		LinkFile link_file;
		link_file.storeLinksCsv(filename_links, model.getLinks());
		WeightFile weight_file;
		weight_file.storeWeightsCsv(filename_weights, model.getWeights());
		return true;
	}

	template<typename TensorT>
	bool ModelFile<TensorT>::loadModelCsv(const std::string & filename_nodes, const std::string & filename_links, const std::string & filename_weights, Model<TensorT>& model)
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

	template<typename TensorT>
	bool ModelFile<TensorT>::storeModelDot(const std::string& filename, const Model<TensorT>& model)
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