/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Node test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(node)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Node* ptr = nullptr;
  Node* nullPointer = nullptr;
	ptr = new Node();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Node* ptr = nullptr;
	ptr = new Node();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Node node(1.0, 2.0, 3.0, 4.0);

  BOOST_CHECK_EQUAL(node.getH(), 1.0);
  BOOST_CHECK_EQUAL(node.getTau(), 2.0);
  BOOST_CHECK_EQUAL(node.getMu(), 3.0);
  BOOST_CHECK_EQUAL(node.getSigma(), 4.0);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Node node;
  node.setH(1.0);
  node.setTau(2.0);
  node.setMu(3.0);
  node.setSigma(4.0);

  BOOST_CHECK_EQUAL(node.getH(), 1.0);
  BOOST_CHECK_EQUAL(node.getTau(), 2.0);
  BOOST_CHECK_EQUAL(node.getMu(), 3.0);
  BOOST_CHECK_EQUAL(node.getSigma(), 4.0);
}

BOOST_AUTO_TEST_SUITE_END()