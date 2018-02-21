/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Link test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Link.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(link)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Link* ptr = nullptr;
  Link* nullPointer = nullptr;
	ptr = new Link();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Link* ptr = nullptr;
	ptr = new Link();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Link link(1.0, 2.0, 3.0, 4.0);

  BOOST_CHECK_EQUAL(link.getH(), 1.0);
  BOOST_CHECK_EQUAL(link.getTau(), 2.0);
  BOOST_CHECK_EQUAL(link.getMu(), 3.0);
  BOOST_CHECK_EQUAL(link.getSigma(), 4.0);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Link link;
  link.setH(1.0);
  link.setTau(2.0);
  link.setMu(3.0);
  link.setSigma(4.0);

  BOOST_CHECK_EQUAL(link.getH(), 1.0);
  BOOST_CHECK_EQUAL(link.getTau(), 2.0);
  BOOST_CHECK_EQUAL(link.getMu(), 3.0);
  BOOST_CHECK_EQUAL(link.getSigma(), 4.0);
}

BOOST_AUTO_TEST_SUITE_END()