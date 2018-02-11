/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PeakSimulator test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/algorithm/PeakSimulator.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(peaksimulator)

// class PeakSimulator_test: public PeakSimulator
// {
// public:
// };

BOOST_AUTO_TEST_CASE(constructor) 
{
  PeakSimulator* ptr = nullptr;
  PeakSimulator* nullPointer = nullptr;
	ptr = new PeakSimulator();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  PeakSimulator* ptr = nullptr;
	ptr = new PeakSimulator();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  PeakSimulator psim;
  psim.setStepSizeMu(500.0);
  psim.setStepSizeSigma(1.0);
  psim.setWindowStart(0.0);
  psim.setWindowEnd(10.0);
  psim.setNoiseMu(2.0);
  psim.setNoiseSimga(1.0);
  psim.setBaselineLeft(5.0);
  psim.setBaselineRight(10.0);
  psim.setSaturationLimit(1e6);

  BOOST_CHECK_EQUAL(psim.getStepSizeMu(), 500.0);
  BOOST_CHECK_EQUAL(psim.getStepSizeSigma(), 1.0);
  BOOST_CHECK_EQUAL(psim.getWindowStart(), 0.0);
  BOOST_CHECK_EQUAL(psim.getWindowEnd(), 10.0);
  BOOST_CHECK_EQUAL(psim.getNoiseMu(), 2.0);
  BOOST_CHECK_EQUAL(psim.getNoiseSigma(), 1.0);
  BOOST_CHECK_EQUAL(psim.getBaselineLeft(), 5.0);
  BOOST_CHECK_EQUAL(psim.getBaselineRight(), 10.0);
  BOOST_CHECK_EQUAL(psim.getSaturationLimit(), 1e6);
}

BOOST_AUTO_TEST_CASE(addNoise) 
{
  PeakSimulator psim;
}

BOOST_AUTO_TEST_SUITE_END()