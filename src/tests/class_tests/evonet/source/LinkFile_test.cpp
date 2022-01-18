/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LinkFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/io/LinkFile.h>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(LinkFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  LinkFile* ptr = nullptr;
  LinkFile* nullPointer = nullptr;
  ptr = new LinkFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  LinkFile* ptr = nullptr;
	ptr = new LinkFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadCsv) 
{
  LinkFile data;

  std::string filename = "LinkFileTest.csv";

  // create list of dummy links
  std::map<std::string, std::shared_ptr<Link>> links;
  for (int i=0; i<3; ++i)
  {
    std::shared_ptr<Link> link(new Link(
      "Link_" + std::to_string(i), 
      "Source_" + std::to_string(i),
      "Sink_" + std::to_string(i),
      "Weight_" + std::to_string(i)));
		link->setModuleName(std::to_string(i));
    links.emplace("Link_" + std::to_string(i), link);
  }
  data.storeLinksCsv(filename, links);

  std::map<std::string, std::shared_ptr<Link>> links_test;
	data.loadLinksCsv(filename, links_test);

	int i = 0;
  for (auto& link_map: links_test)
  {
    BOOST_CHECK_EQUAL(link_map.second->getName(), "Link_" + std::to_string(i));
    BOOST_CHECK_EQUAL(link_map.second->getSourceNodeName(), "Source_" + std::to_string(i));
    BOOST_CHECK_EQUAL(link_map.second->getSinkNodeName(), "Sink_" + std::to_string(i));
    BOOST_CHECK_EQUAL(link_map.second->getWeightName(), "Weight_" + std::to_string(i));
		BOOST_CHECK_EQUAL(link_map.second->getModuleName(), std::to_string(i));
		++i;
  }
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary)
{
	LinkFile data;

	std::string filename = "LinkFileTest.bin";

	// create list of dummy links
	std::map<std::string, std::shared_ptr<Link>> links;
	for (int i = 0; i < 3; ++i)
	{
		std::shared_ptr<Link> link(new Link(
			"Link_" + std::to_string(i),
			"Source_" + std::to_string(i),
			"Sink_" + std::to_string(i),
			"Weight_" + std::to_string(i)));
		link->setModuleName(std::to_string(i));
		links.emplace("Link_" + std::to_string(i), link);
	}
	data.storeLinksBinary(filename, links);

	std::map<std::string, std::shared_ptr<Link>> links_test;
	data.loadLinksBinary(filename, links_test);

	int i = 0;
	for (auto& link_map : links_test)
	{
		BOOST_CHECK_EQUAL(link_map.second->getName(), "Link_" + std::to_string(i));
		BOOST_CHECK_EQUAL(link_map.second->getSourceNodeName(), "Source_" + std::to_string(i));
		BOOST_CHECK_EQUAL(link_map.second->getSinkNodeName(), "Sink_" + std::to_string(i));
		BOOST_CHECK_EQUAL(link_map.second->getWeightName(), "Weight_" + std::to_string(i));
		BOOST_CHECK_EQUAL(link_map.second->getModuleName(), std::to_string(i));
		++i;
	}
}

BOOST_AUTO_TEST_SUITE_END()