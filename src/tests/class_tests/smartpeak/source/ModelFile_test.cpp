/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/ModelFile.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ModelFile* ptr = nullptr;
  ModelFile* nullPointer = nullptr;
  ptr = new ModelFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ModelFile* ptr = nullptr;
	ptr = new ModelFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeModelDot)
{
	ModelFile data;

	std::string filename = "ModelFileTest.gv";

	// create list of dummy links
	std::vector<Link> links;
	for (int i = 0; i<3; ++i)
	{
		const Link link(
			"Link_" + std::to_string(i),
			"Node_" + std::to_string(i),
			"Node_" + std::to_string(i + 1),
			"Weight_" + std::to_string(i));
		links.push_back(link);
	}
	data.storeModelDot(filename, links);
}

BOOST_AUTO_TEST_SUITE_END()