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
  PeakSimulator emg;
  emg.setStepSizeMu(500.0);
  emg.setStepSizeSigma(1.0);
  emg.setWindowStart(0.0);
  emg.setWindowEnd(10.0);
  emg.setNoiseMu(2.0);
  emg.setNoiseSimga(1.0);

  BOOST_CHECK_EQUAL(emg.getStepSizeMu(), 500.0);
  BOOST_CHECK_EQUAL(emg.getStepSizeSigma(), 1.0);
  BOOST_CHECK_EQUAL(emg.getWindowStart(), 0.0);
  BOOST_CHECK_EQUAL(emg.getWindowEnd(), 10.0);
  BOOST_CHECK_EQUAL(emg.getNoiseMu(), 2.0);
  BOOST_CHECK_EQUAL(emg.getNoiseSigma(), 1.0);
}

BOOST_AUTO_TEST_SUITE_END()