/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LinkFile test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/io/LinkFile.h>
// #include <filesystem> C++ 17

using namespace SmartPeak;
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
  std::vector<Link> links;
  for (int i=0; i<3; ++i)
  {
    const Link link(
      "Link_" + std::to_string(i), 
      "Source_" + std::to_string(i),
      "Sink_" + std::to_string(i),
      "Weight_" + std::to_string(i));
  }
  data.storeLinksCsv<float, 3>(filename, links);

	data.loadLinksCsv<float, 3>(filename, links);

  for (int i=0; i<3; ++i)
  {
    BOOST_CHECK_EQUAL(link.getName(), "Link_" + std::to_string(i));
    BOOST_CHECK_EQUAL(link.getSourceNodeName(), "Source_" + std::to_string(i));
    BOOST_CHECK_EQUAL(link.getSinkNodeName(), "Sink_" + std::to_string(i));
    BOOST_CHECK_EQUAL(link.getWeightName(), "Weight_" + std::to_string(i));
  }
}

BOOST_AUTO_TEST_SUITE_END()