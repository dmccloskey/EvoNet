/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE EMGModel test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/algorithm/EMGModel.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(emgmodel)

BOOST_AUTO_TEST_CASE(constructor) 
{
  EMGModel* ptr = nullptr;
  EMGModel* nullPointer = nullptr;
	ptr = new EMGModel();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  EMGModel* ptr = nullptr;
	ptr = new EMGModel();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  EMGModel emg;
  emg.setH(1.0);
  emg.setTau(2.0);
  emg.setMu(3.0);
  emg.setSigma(4.0);

  BOOST_CHECK_EQUAL(emg.getH(), 1.0);
  BOOST_CHECK_EQUAL(emg.getTau(), 2.0);
  BOOST_CHECK_EQUAL(emg.getMu(), 3.0);
  BOOST_CHECK_EQUAL(emg.getSigma(), 4.0);
}

BOOST_AUTO_TEST_SUITE_END()